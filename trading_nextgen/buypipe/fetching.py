import os
import json
import time
import shutil
import tempfile
import contextlib
from datetime import datetime, timezone
from typing import Tuple, Optional

import pandas as pd

try:
    from ib_insync import Stock, Index, IB
except Exception as _e:
    IB = None
    Stock = None
    Index = None


from .config import CACHE_DIR, IB_CLIENT_ID, IB_HOST, IB_PORT

os.makedirs(CACHE_DIR, exist_ok=True)

# In-process (run-level) cache to avoid repeated parquet reads / network pulls in the same run
_RUN_DF_CACHE = {}

# Preferred exchanges to try when qualifying contracts
_EXCHANGE_TRIES = [
    dict(exchange="SMART", primaryExchange="ARCA"),
    dict(exchange="ARCA"),
    dict(exchange="SMART"),
    dict(exchange="ISLAND", primaryExchange="NASDAQ"),
]


# ---------------------------- Helpers ----------------------------

def _tz_naive_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Force df['date'] to tz-naive (UTC clock)."""
    if df is not None and not df.empty and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
    return df


def _slugify(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in str(s))


def get_cache_paths(symbol: str, bar_spec: str = "10Y-1D") -> Tuple[str, str]:
    slug = f"{_slugify(symbol)}__{_slugify(bar_spec)}"
    data_path = os.path.join(CACHE_DIR, f"{slug}.parquet")
    meta_path = os.path.join(CACHE_DIR, f"{slug}.meta.json")
    return data_path, meta_path


def write_cache_atomic(df: pd.DataFrame, data_path: str, meta_path: str) -> None:
    """Write parquet + meta atomically."""
    tmp_dir = tempfile.mkdtemp(prefix="cachewrite_")
    try:
        # Data
        df = _tz_naive_utc(df.copy())
        tmp_file = os.path.join(tmp_dir, os.path.basename(data_path) + ".tmp.parquet")
        df.to_parquet(tmp_file, index=False)
        os.replace(tmp_file, data_path)

        # Meta (store date-only to avoid tz confusion)
        latest_date = None
        for col in ("date", "Date", "datetime", "time", "timestamp"):
            if col in df.columns:
                ld = pd.to_datetime(df[col], errors="coerce", utc=True).max()
                latest_date = ld.tz_localize(None).date().isoformat() if pd.notna(ld) else None
                break

        meta = {
            "rows": int(len(df)),
            "latest_date": latest_date,
            "written_at": datetime.now(timezone.utc).isoformat(),
            "format": "parquet",
        }
        tmp_meta = os.path.join(tmp_dir, os.path.basename(meta_path) + ".tmp.json")
        with open(tmp_meta, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        os.replace(tmp_meta, meta_path)
    finally:
        with contextlib.suppress(Exception):
            shutil.rmtree(tmp_dir, ignore_errors=True)


def read_cache_if_exists(symbol: str, bar_spec: str) -> Tuple[Optional[pd.DataFrame], Optional[dict], str, str]:
    data_path, meta_path = get_cache_paths(symbol, bar_spec)
    if not (os.path.exists(data_path) and os.path.exists(meta_path)):
        return None, None, data_path, meta_path

    try:
        df = pd.read_parquet(data_path)
        df = _tz_naive_utc(df)
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        return df, meta, data_path, meta_path
    except Exception:
        # Corrupt cache → delete both
        with contextlib.suppress(Exception):
            os.remove(data_path)
        with contextlib.suppress(Exception):
            os.remove(meta_path)
        return None, None, data_path, meta_path


def is_cache_stale(meta: dict, ttl_minutes: int = 360, require_today: bool = True) -> bool:
    """Return True if cache is considered stale."""
    if not meta:
        return True
    try:
        written_at = datetime.fromisoformat(meta.get("written_at"))
    except Exception:
        return True

    # Age check (tolerate naive written_at by attaching UTC)
    now_utc = datetime.now(timezone.utc)
    if written_at.tzinfo is None:
        written_at = written_at.replace(tzinfo=timezone.utc)
    age_minutes = (now_utc - written_at).total_seconds() / 60.0
    if age_minutes > ttl_minutes:
        return True

    if require_today:
        latest = meta.get("latest_date")
        if latest is None:
            return True
        try:
            latest_date = pd.to_datetime(latest, errors="coerce").date()
        except Exception:
            return True
        today_date = now_utc.date()
        if latest_date < today_date:
            return True

    return False


# ------------------------- IBKR Fetching -------------------------

def _resolve_contract(ib: "IB", symbol: str):
    """Try multiple exchanges to qualify a contract quickly.

    Special-case index-style macro symbols like VIX.
    """
    symbol_up = symbol.upper()
    last_exc = None

    # --- Index contracts (VIX etc.) ---
    if symbol_up == "VIX" and "Index" in globals() and Index is not None:
        try:
            # CBOE VIX index
            c = Index("VIX", "CBOE")
            qc = ib.qualifyContracts(c)
            if qc:
                return qc[0]
        except Exception as e:
            last_exc = e  # fall through to stock attempts if needed

    # --- Default: treat as stock / ETF ---
    for ex in _EXCHANGE_TRIES:
        try:
            c = Stock(symbol, **ex, currency="USD")
            qc = ib.qualifyContracts(c)
            if qc:
                return qc[0]
        except Exception as e:
            last_exc = e
            continue

    if last_exc:
        raise last_exc
    raise RuntimeError(f"Could not qualify contract for {symbol}")


def _fetch_ib_bars(ib: "IB", symbol: str, bar_spec: str, bar_size: str) -> pd.DataFrame:
    """
    Fetch historical bars using ib_insync.
    bar_spec examples: '3 Y', '1 Y', '180 D'
    bar_size examples: '1 day', '1 hour'
    """
    contract = _resolve_contract(ib, symbol)
    #  WhatToShow: TRADES for stocks/ETFs; use RTH=0 for full data
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=bar_spec,
        barSizeSetting=bar_size,
        whatToShow="TRADES",
        useRTH=False,
        formatDate=1,
        keepUpToDate=False,
    )
    if not bars:
        return pd.DataFrame()

    # Convert to DataFrame
    recs = [dict(date=b.date, open=b.open, high=b.high, low=b.low, close=b.close, volume=b.volume) for b in bars]
    df = pd.DataFrame.from_records(recs)
    # Normalize date and basic schema
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    # Ensure numeric
    for c in ("open", "high", "low", "close", "volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    # Deduplicate/sort
    df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    return df


# --------------------------- Public API --------------------------

def fetch_data_cached(
    symbol: str,
    bar_spec: str = "10 Y",
    bar_size: str = "1 day",
    ttl_minutes: int = 360,
    require_today: bool = True,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return OHLCV DataFrame for `symbol` with columns: date, open, high, low, close, volume.

    Improvements vs prior version:
      - Run-level in-memory cache avoids repeated parquet reads / repeated network pulls in the same run.
      - "Refresh in place": if cache exists and is stale, fetch only the missing tail (small duration)
        and append/dedupe, instead of re-downloading the full history every time.

    Cache behavior:
      - If run-cache has it => return immediately (even if force_refresh was used earlier in this run).
      - Else if disk cache exists and is not stale (and not force_refresh) => return cached.
      - Else attempt live fetch, write cache, and return.
      - On fetch errors, fall back to any existing cache; else empty DataFrame.
    """
    # Compose a compact bar_spec for cache key
    bar_key = f"{bar_spec.strip().replace(' ', '')}-{bar_size.strip().replace(' ', '')}"
    run_key = (symbol.upper().strip(), bar_key)

    # 0) Run-level cache: if we already fetched/loaded this exact key in this run, reuse it.
    if run_key in _RUN_DF_CACHE:
        return _RUN_DF_CACHE[run_key]

    df_cache, meta, data_path, meta_path = read_cache_if_exists(symbol, bar_key)

    # 1) Fast path: disk cache is fresh and refresh not forced
    if df_cache is not None and not force_refresh and not is_cache_stale(meta, ttl_minutes, require_today):
        _RUN_DF_CACHE[run_key] = df_cache
        return df_cache

    # 2) Live fetch (IBKR)
    if IB is None:
        print("⚠️ ib_insync not available; returning cache if present.")
        out = df_cache if df_cache is not None else pd.DataFrame()
        _RUN_DF_CACHE[run_key] = out
        return out

    ib = IB()
    try:
        ib.connect(IB_HOST, IB_PORT, clientId=int(IB_CLIENT_ID), readonly=True, timeout=10.0)

        # --- Refresh in place (incremental tail fetch) ---
        # If we already have cache, and we're only missing a small tail, fetch just that tail.
        # Works best for daily bars; for weekly/other bar sizes we keep full fetch.
        if df_cache is not None and not df_cache.empty and str(bar_size).strip().lower() in {"1 day", "1day"}:
            try:
                last_dt = pd.to_datetime(df_cache["date"], errors="coerce").max()
                if pd.notna(last_dt):
                    last_date = last_dt.date()
                    today_utc = datetime.now(timezone.utc).date()
                    missing_days = (today_utc - last_date).days

                    # If missing is small, do a small duration fetch and append.
                    # Add a small buffer to handle holidays/timezone boundary.
                    if 0 < missing_days <= 30:
                        inc_spec = f"{min(missing_days + 5, 35)} D"
                        df_inc = _fetch_ib_bars(ib, symbol, bar_spec=inc_spec, bar_size=bar_size)

                        if df_inc is not None and not df_inc.empty:
                            df_combined = (
                                pd.concat([df_cache, df_inc], ignore_index=True)
                                .drop_duplicates(subset=["date"], keep="last")
                                .sort_values("date")
                                .reset_index(drop=True)
                            )
                            write_cache_atomic(df_combined, data_path, meta_path)
                            _RUN_DF_CACHE[run_key] = df_combined
                            return df_combined
            except Exception:
                # Fall through to full fetch if incremental refresh fails
                pass

        # Full fetch (either no cache, or cache too old / non-daily)
        df_new = _fetch_ib_bars(ib, symbol, bar_spec=bar_spec, bar_size=bar_size)
        if df_new is None or df_new.empty:
            # Nothing new; return cache if exists
            out = df_cache if df_cache is not None else pd.DataFrame()
            _RUN_DF_CACHE[run_key] = out
            return out

        write_cache_atomic(df_new, data_path, meta_path)
        _RUN_DF_CACHE[run_key] = df_new
        return df_new

    except Exception as e:
        print(f" Error fetching {symbol}: {e}")
        # Fallback to cache if available
        out = df_cache if df_cache is not None else pd.DataFrame()
        _RUN_DF_CACHE[run_key] = out
        return out
    finally:
        with contextlib.suppress(Exception):
            ib.disconnect()


if __name__ == "__main__":
    # Tiny smoke test (won't run if ib_insync isn't configured)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", type=str, default="AAPL")
    ap.add_argument("--bar-spec", type=str, default="10 Y")
    ap.add_argument("--bar-size", type=str, default="1 day")
    ap.add_argument("--require-today", action="store_true")
    args = ap.parse_args()

    df = fetch_data_cached(args.symbol, bar_spec=args.bar_spec, bar_size=args.bar_size, require_today=args.require_today)
    print(f"Rows: {len(df)}")
    if not df.empty and "date" in df.columns:
        print("Last date:", df['date'].max())


# ==================== IBKR -> Alpha Vantage fallback (Daily OHLCV) ====================
# Used by buy_srs_update_excel.py
try:
    import requests
except Exception:
    requests = None


def _fetch_alpha_vantage_daily(symbol: str, outputsize: str = "full") -> pd.DataFrame:
    """Alpha Vantage daily OHLCV fallback.
    Returns a DataFrame with columns: date, open, high, low, close, volume (sorted ascending).
    """
    if requests is None:
        raise RuntimeError("Missing dependency: requests. Install via: pip install requests")

    api_key = os.getenv("ALPHAVANTAGE_API_KEY") or os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("Alpha Vantage API key not found. Set env var ALPHAVANTAGE_API_KEY")

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "outputsize": outputsize,
        "apikey": api_key,
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    if "Note" in data:
        raise RuntimeError(f"AlphaVantage throttled: {data['Note']}")
    if "Error Message" in data:
        raise RuntimeError(f"AlphaVantage error: {data['Error Message']}")

    ts = data.get("Time Series (Daily)")
    if not ts:
        return pd.DataFrame()

    rows = []
    for dt, v in ts.items():
        rows.append({
            "date": dt,
            "open": float(v.get("1. open", "nan")),
            "high": float(v.get("2. high", "nan")),
            "low": float(v.get("3. low", "nan")),
            "close": float(v.get("4. close", "nan")),
            "volume": float(v.get("6. volume", "nan")),
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # basic AV rate-limit friendliness
    time.sleep(0.9)
    return df


def fetch_data_daily_with_fallback(
    symbol: str,
    bar_spec: str = "10 Y",
    bar_size: str = "1 day",
    ttl_minutes: int = 360,
    require_today: bool = False,
    force_refresh: bool = False,
):
    """Fetch daily OHLCV. Returns (df, source, err).

    source: 'IBKR' | 'ALPHAVANTAGE' | 'ERROR'
    err: string message (or None)

    Uses your existing fetch_data_cached() (IBKR + cache). If it fails/empty, falls back to Alpha Vantage.
    """
    # 1) IBKR
    try:
        df = fetch_data_cached(
            symbol=symbol,
            bar_spec=bar_spec,
            bar_size=bar_size,
            ttl_minutes=ttl_minutes,
            require_today=require_today,
            force_refresh=force_refresh,
        )
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df, "IBKR", None
    except Exception as e:
        ib_err = str(e)
    else:
        ib_err = "IBKR returned empty"

    # 2) Alpha Vantage
    try:
        df2 = _fetch_alpha_vantage_daily(symbol, outputsize="full")
        if isinstance(df2, pd.DataFrame) and not df2.empty:
            return df2, "ALPHAVANTAGE", None
        return pd.DataFrame(), "ERROR", f"IBKR failed/empty ({ib_err}); AlphaVantage returned empty"
    except Exception as e:
        return pd.DataFrame(), "ERROR", f"IBKR failed/empty ({ib_err}); AlphaVantage failed ({e})"
