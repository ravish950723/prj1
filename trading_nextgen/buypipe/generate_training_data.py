from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .fetching import fetch_data_cached
from .compute import compute_indicators
from .eps_features import compute_eps_features, fetch_market_sentiment
from .train_dataset import load_quant_features
from .eps_cache import get_eps_data, get_sentiment_data

DEFAULT_SYMBOLS: List[str] = []
try:
    from .config import SYMBOLS as _S1
    DEFAULT_SYMBOLS = list(dict.fromkeys([s.strip().upper() for s in _S1 if isinstance(s, str)]))
except Exception:
    try:
        from .config import symbols as _S2
        DEFAULT_SYMBOLS = list(dict.fromkeys([s.strip().upper() for s in _S2 if isinstance(s, str)]))
    except Exception:
        DEFAULT_SYMBOLS = []


def tz_naive_utc(df: pd.DataFrame) -> pd.DataFrame:
    if df is not None and not df.empty and "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
    return df


def parse_symbols(arg: Optional[str]) -> List[str]:
    if arg is None or arg.strip() == "":
        return DEFAULT_SYMBOLS
    toks = [t.strip().upper() for t in arg.replace(";", ",").replace("|", ",").replace(" ", ",").split(",")]
    return [t for t in toks if t]


def _fetch_eps_payload(symbol: str) -> Dict:
    return compute_eps_features(symbol, None) or {}


def _fetch_sentiment_payload(symbol: str) -> Dict:
    return fetch_market_sentiment(symbol) or {}


def add_fundamental_and_sentiment_features(df: pd.DataFrame, symbol: str, debug: bool = True) -> pd.DataFrame:
    df = df.copy()

    try:
        eps_feats = get_eps_data(symbol, _fetch_eps_payload)
        new_cols = {}
        for k, v in eps_feats.items():
            try:
                new_cols[k] = float(v)
            except:
                new_cols[k] = 0.0

        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    except Exception as e:
        if debug:
            print(f"⚠️ {symbol}: EPS enrichment failed ({e})")

    asset_type = pd.Series(df.get("ASSET_TYPE", ""), index=df.index).astype(str).str.upper()
    df["ASSET_TYPE_STOCK"] = asset_type.eq("STOCK").astype(int)
    df["ASSET_TYPE_ETF"] = asset_type.eq("ETF").astype(int)

    eps_q = series_or_default(df, "eps_quality_score", 0.0).fillna(0.0)

    etf_q = series_or_default(df, "ETF_PROXY_GROWTH_SCORE", 0.0).fillna(0.0)

    df["FUNDAMENTAL_BOOST"] = np.where(df["ASSET_TYPE_STOCK"] == 1, 0.08 * eps_q, 0.05 * etf_q)

    try:
        sent = get_sentiment_data(symbol, _fetch_sentiment_payload)
    except Exception as e:
        if debug:
            print(f"⚠️ {symbol}: sentiment enrichment failed ({e})")
        sent = {}

    df["news_sentiment_score"] = float(sent.get("news_sentiment_score") or 0.0)
    df["news_positive_ratio"] = float(sent.get("news_positive_ratio") or 0.0)
    df["news_article_count"] = float(sent.get("news_article_count") or 0.0)
    return df


def compute_forward_label_columns(df: pd.DataFrame, horizon: int = 60, gain_thr: float = 0.10,
                                  max_dd_thr: float = -0.07) -> pd.DataFrame:
    df = df.copy()
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")

    fut_max_high = high[::-1].rolling(window=horizon, min_periods=1).max()[::-1].shift(-1)
    fut_min_low = low[::-1].rolling(window=min(20, horizon), min_periods=1).min()[::-1].shift(-1)
    future_return = (fut_max_high / close) - 1.0
    future_drawdown = (fut_min_low / close) - 1.0

    trend_ok = (
        pd.to_numeric(df.get("EMA_21", np.nan), errors="coerce") >
        pd.to_numeric(df.get("EMA_50", np.nan), errors="coerce")
    )

    adx_ok = pd.to_numeric(
        df["ADX_14"] if "ADX_14" in df.columns else pd.Series(0.0, index=df.index),
        errors="coerce"
    ).fillna(0.0) >= 16  # relaxed from 18

    volume_ok = pd.to_numeric(
        df["VOL_SURGE_RATIO"] if "VOL_SURGE_RATIO" in df.columns else pd.Series(0.0, index=df.index),
        errors="coerce"
    ).fillna(0.0) >= 1.00  # relaxed from 1.05

    breakout_ok = pd.to_numeric(
        df["breakout_strength"] if "breakout_strength" in df.columns else pd.Series(0.0, index=df.index),
        errors="coerce"
    ).fillna(0.0) >= -0.40  # relaxed from -0.25

    sentiment_ok = pd.to_numeric(
        df["news_sentiment_score"] if "news_sentiment_score" in df.columns else pd.Series(0.0, index=df.index),
        errors="coerce"
    ).fillna(0.0) >= -0.35

    fundamental_ok = pd.to_numeric(
        df["FUNDAMENTAL_BOOST"] if "FUNDAMENTAL_BOOST" in df.columns else pd.Series(0.0, index=df.index),
        errors="coerce"
    ).fillna(0.0) >= -1.0


    strong_buy = (
        (future_return >= float(gain_thr)) &
        (future_drawdown >= float(max_dd_thr)) &
        trend_ok &
        adx_ok &
        volume_ok &
        breakout_ok &
        sentiment_ok &
        fundamental_ok
    ).fillna(False).astype(int)

    buy = (
        (future_return >= max(0.06, float(gain_thr) * 0.75)) &
        (future_drawdown >= min(-0.10, float(max_dd_thr) * 1.25))
    ).fillna(False).astype(int)

    avoid = ((future_return <= 0.02) | (future_drawdown <= -0.12)).fillna(False).astype(int)

    df["future_return_max"] = future_return.replace([np.inf, -np.inf], np.nan)
    df["future_drawdown_min"] = future_drawdown.replace([np.inf, -np.inf], np.nan)
    df["strong_buy"] = strong_buy
    df["buy"] = buy
    df["avoid"] = avoid
    return df


def build_training_rows(symbol: str, bar_spec: str, bar_size: str, horizon: int, gain_thr: float,
                        min_rows: int, debug: bool = True) -> Optional[pd.DataFrame]:
    try:
        df = fetch_data_cached(symbol, bar_spec=bar_spec, bar_size=bar_size)
        df = tz_naive_utc(df)
        if df is None or df.empty:
            raise ValueError("empty dataframe after fetch")

        if "date" in df.columns:
            df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

        df = compute_indicators(df, symbol=symbol)
        new_cols = {}
        new_cols["regime_combo"] = df["sym_vol_regime"].fillna(0) * 2 + df["VIX_vol_regime"].fillna(0)

        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        df = add_fundamental_and_sentiment_features(df, symbol=symbol, debug=debug)

        needed = ["date", "open", "high", "low", "close", "volume"]
        for c in needed:
            if c not in df.columns:
                raise KeyError(f"missing required column: {c}")

        df = compute_forward_label_columns(df, horizon=horizon, gain_thr=gain_thr, max_dd_thr=-0.07)
        df = df.dropna(subset=["close"]).reset_index(drop=True)
        if min_rows and len(df) < min_rows:
            raise ValueError(f"insufficient rows ({len(df)} < {min_rows})")
        return df
    except Exception as e:
        if debug:
            print(f" {symbol}: fetch/compute failed ({e}); skipping")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=str, default="", help="Comma/space-separated symbols; defaults to config.SYMBOLS")
    ap.add_argument("--bar-spec", type=str, default="10 Y", help='IBKR duration string (default: "10 Y")')
    ap.add_argument("--bar-size", type=str, default="1 day", help='IBKR bar size (default: "1 day")')
    ap.add_argument("--horizon", type=int, default=60, help="Forward-looking window in days (default: 60)")
    ap.add_argument("--gain-thr", type=float, default=0.10, help="Gain threshold (e.g., 0.10 = +10%)")
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
        df = build_training_rows(sym, bar_spec=args.bar_spec, bar_size=args.bar_size,
                                 horizon=args.horizon, gain_thr=args.gain_thr,
                                 min_rows=args.min_rows, debug=not args.quiet)
        if df is not None and not df.empty:
            all_rows.append(df)

    if not all_rows:
        print(" No training rows were generated. Check data fetch, cache freshness, or symbol list.")
        sys.exit(1)

    full = pd.concat(all_rows, ignore_index=True)
    if "date" in full.columns:
        full["date"] = pd.to_datetime(full["date"], errors="coerce")
        full = full.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    feature_names = load_quant_features()
    with open("model_features.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(feature_names) + "\n")

    full.to_csv(args.out, index=False)

    y = pd.to_numeric(full["strong_buy"], errors="coerce").fillna(0).astype(int)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    total = int(len(y))
    pos_pct = (pos / max(total, 1)) * 100.0
    avg_ret = float(series_or_default(full, "future_return_max", 0.0).fillna(0.0).mean())
    avg_dd = float(series_or_default(full, "future_drawdown_min", 0.0).fillna(0.0).mean())
    print(f" Saved {args.out} with {total} rows ({pos} positives, {neg} negatives; positives={pos_pct:.2f}%; avg_future_return={avg_ret:.3f}; avg_future_drawdown={avg_dd:.3f}).")



def series_or_default(df: pd.DataFrame, col: str, default=0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")

if __name__ == "__main__":
    main()
