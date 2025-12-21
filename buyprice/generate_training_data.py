#!/usr/bin/env python3
"""
generate_training_data.py
-------------------------
Build a time-ordered training dataset from cached/IBKR OHLCV, compute indicators,
and label rows as `strong_buy` if price rises by >= GAIN_THR within the next N days.

Key features:
- TZ-safe: normalizes all timestamps to tz-naive (UTC clock) to avoid tz-aware/naive comparisons.
- Uses your project's `fetching.fetch_data_cached` and `compute.compute_indicators`.
- Chronological ordering to prevent leakage.
- Saves:
    - train_data.csv
    - model_features.txt (names of numeric feature columns used by the trainer)

Usage examples:
    python generate_training_data.py
    python generate_training_data.py --symbols AAPL,MSFT,NVDA --gain-thr 0.10 --horizon 90
    python generate_training_data.py --out my_train.csv --limit 50

"""
import argparse
import sys
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd

# Project modules
from fetching import fetch_data_cached
from compute import compute_indicators
from eps_features import fetch_quarterly_eps, eps_growth_flags, fetch_market_sentiment


# Try to import symbol universe from config.py (be flexible on variable names)
DEFAULT_SYMBOLS: List[str] = []
try:
    from config import SYMBOLS as _S1  # common name
    DEFAULT_SYMBOLS = list(dict.fromkeys([s.strip().upper() for s in _S1 if isinstance(s, str)]))
except Exception:
    try:
        from config import symbols as _S2  # alternate
        DEFAULT_SYMBOLS = list(dict.fromkeys([s.strip().upper() for s in _S2 if isinstance(s, str)]))
    except Exception:
        # Fallback: empty list; user can pass --symbols
        DEFAULT_SYMBOLS = []

def tz_naive_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Force 'date' to tz-naive (UTC clock) for safe comparisons/merges."""
    if df is not None and not df.empty and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
    return df

def label_strong_buy_forward_max(df: pd.DataFrame, horizon: int = 90, gain_thr: float = 0.08,
                                 use_close: bool = True) -> pd.Series:
    """
    Vectorized "hit within next N days" using a reverse-rolling max.

    If use_close=True: uses future max of 'close'; else uses future max of 'high'.
    strong_buy = 1 if future_max >= (current_close * (1 + gain_thr)) else 0
    """
    if df.empty:
        return pd.Series([], dtype=int)
    price_col = "close" if use_close else "high"
    px = pd.to_numeric(df[price_col], errors="coerce")
    # Reverse, rolling max, reverse back. Rolling includes current row, so shift(-1) to exclude today.
    fut_max = px[::-1].rolling(window=horizon, min_periods=1).max()[::-1].shift(-1)
    target_lvl = pd.to_numeric(df["close"], errors="coerce") * (1.0 + float(gain_thr))
    strong = (fut_max >= target_lvl).astype(int)
    strong = strong.fillna(0).astype(int)
    return strong

def parse_symbols(arg: Optional[str]) -> List[str]:
    if arg is None or arg.strip() == "":
        return DEFAULT_SYMBOLS
    # allow comma/space separated
    toks = [t.strip().upper() for t in arg.replace(";", ",").replace("|", ",").replace(" ", ",").split(",")]
    return [t for t in toks if t]

def build_training_rows(symbol: str, bar_spec: str, bar_size: str,
                        horizon: int, gain_thr: float, min_rows: int, debug: bool = True) -> Optional[pd.DataFrame]:
    try:
        df = fetch_data_cached(symbol, bar_spec=bar_spec, bar_size=bar_size)
        df = tz_naive_utc(df)
        if df is None or df.empty:
            raise ValueError("empty dataframe after fetch")

        # Sort by time & drop dupes to guarantee chronology
        if "date" in df.columns:
            df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

        # Compute indicators (also normalizes dates internally by our patched code)
        df = compute_indicators(df, symbol=symbol)
        df["regime_combo"] = (
                df["sym_vol_regime"].fillna(0) * 2 +
                df["VIX_vol_regime"].fillna(0)
        )

        # === Add Alpha Vantage fundamentals & sentiment ===
        # 1) EPS growth flags (static per symbol, broadcast over all rows)
        try:
            eps_df = fetch_quarterly_eps(symbol)
            eps_flags = eps_growth_flags(eps_df)
        except Exception as _e:
            if debug:
                print(f"⚠️ {symbol}: EPS enrichment failed ({_e})")
            eps_flags = {}

        def _flag_to_float(v):
            # True → 1.0, False → 0.0, None/missing → 0.0
            if v is True:
                return 1.0
            if v is False:
                return 0.0
            return 0.0

        df["eps_inc_2q"] = _flag_to_float(eps_flags.get("EPS Increase 2Q"))
        df["eps_inc_3q"] = _flag_to_float(eps_flags.get("EPS Increase 3Q"))
        df["eps_inc_4q"] = _flag_to_float(eps_flags.get("EPS Increase 4Q"))

        # 2) Recent news sentiment features
        try:
            sent = fetch_market_sentiment(symbol)
        except Exception as _e:
            if debug:
                print(f"⚠️ {symbol}: sentiment enrichment failed ({_e})")
            sent = {}

        df["news_sentiment_score"] = float(sent.get("news_sentiment_score") or 0.0)
        df["news_positive_ratio"] = float(sent.get("news_positive_ratio") or 0.0)
        df["news_article_count"] = float(sent.get("news_article_count") or 0.0)

        # Minimal required columns
        needed = ["date", "open", "high", "low", "close", "volume"]
        for c in needed:
            if c not in df.columns:
                raise KeyError(f"missing required column: {c}")

        # Label target
        df["strong_buy"] = label_strong_buy_forward_max(df, horizon=horizon, gain_thr=gain_thr, use_close=True)

        # Drop early NaNs (indicator warmups)
        df = df.dropna(subset=["close"]).reset_index(drop=True)
        if min_rows and len(df) < min_rows:
            raise ValueError(f"insufficient rows ({len(df)} < {min_rows})")

        # Keep last 3 years worth (already controlled by fetch), but ensure fully valid numeric columns
        return df
    except Exception as e:
        if debug:
            print(f"❌ {symbol}: fetch/compute failed ({e}); skipping")
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=str, default="", help="Comma/space-separated symbols; defaults to config.SYMBOLS")
    ap.add_argument("--bar-spec", type=str, default="10 Y", help='IBKR duration string (default: "10 Y")')
    ap.add_argument("--bar-size", type=str, default="1 day", help='IBKR bar size (default: "1 day")')
    ap.add_argument("--horizon", type=int, default=90, help="Forward-looking window in days (default: 90)")
    ap.add_argument("--gain-thr", type=float, default=0.08, help="Gain threshold (e.g., 0.08 = +8%)")
    ap.add_argument("--min-rows", type=int, default=150, help="Minimum rows per symbol to keep")
    ap.add_argument("--out", type=str, default="train_data.csv", help="Output CSV path")
    ap.add_argument("--limit", type=int, default=0, help="Optional cap on number of symbols processed")
    ap.add_argument("--quiet", action="store_true", help="Reduce console logging")
    args = ap.parse_args()

    symbols = parse_symbols(args.symbols)
    if args.limit and args.limit > 0:
        symbols = symbols[: args.limit]

    if not symbols:
        print("⚠️ No symbols provided and none found in config. Use --symbols AAPL,MSFT or add SYMBOLS in config.py")
        sys.exit(2)

    all_rows: List[pd.DataFrame] = []
    for i, sym in enumerate(symbols, 1):
        if not args.quiet:
            print(f"[{i}/{len(symbols)}] {sym} …")
        df = build_training_rows(
            sym, bar_spec=args.bar_spec, bar_size=args.bar_size,
            horizon=args.horizon, gain_thr=args.gain_thr, min_rows=args.min_rows, debug=not args.quiet
        )
        if df is not None and not df.empty:
            # Only keep fully numeric features + date + target
            # (Trainer will drop non-numeric; we still keep date for chronological checks)
            all_rows.append(df)

    if not all_rows:
        print("❌ No training rows were generated. Check data fetch, cache freshness, or symbol list.")
        sys.exit(1)

    full = pd.concat(all_rows, ignore_index=True)
    # Enforce time order across the dataset
    if "date" in full.columns:
        full["date"] = pd.to_datetime(full["date"], errors="coerce")
        full = full.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Save feature name list for downstream inference alignment
    numeric_cols = full.drop(columns=["strong_buy"], errors="ignore").select_dtypes(include="number").columns.tolist()
    with open("model_features.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(numeric_cols))

    # Save dataset
    full.to_csv(args.out, index=False)

    # Quick label distribution
    y = full["strong_buy"].astype(int)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    total = int(len(y))
    pos_pct = (pos / max(total, 1)) * 100.0

    print(f"✅ Saved {args.out} with {total} rows ({pos} positives, {neg} negatives; positives={pos_pct:.2f}%).")
    print(f"📝 Saved model_features.txt with {len(numeric_cols)} numeric feature names.")

if __name__ == "__main__":
    main()
