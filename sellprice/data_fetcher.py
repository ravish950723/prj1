# data_fetcher.py
from __future__ import annotations

from datetime import datetime, timezone
import pandas as pd
from ib_insync import Stock, util

from ib_connection import get_ib
from cache_utils import load_cache, save_cache

# Map IB barSizeSetting -> our cache timeframe label
_BARSIZE_TO_TF = {
    "1 day": "daily",
    "1 week": "weekly",
    "1 hour": "hourly",
    # extend as needed
}


def _tf_from_bar_size(bar_size: str) -> str:
    if not isinstance(bar_size, str):
        return "daily"
    return _BARSIZE_TO_TF.get(bar_size.strip().lower(), bar_size.strip().lower())


def _empty_frame(symbol: str) -> pd.DataFrame:
    """Return an empty but schema-correct frame so downstream code won't KeyError."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime([], errors="coerce"),
            "open": pd.Series([], dtype="float64"),
            "high": pd.Series([], dtype="float64"),
            "low": pd.Series([], dtype="float64"),
            "close": pd.Series([], dtype="float64"),
            "volume": pd.Series([], dtype="float64"),
            "symbol": pd.Series([], dtype="object"),
        }
    ).assign(symbol=symbol)


def _normalize_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Ensure a consistent schema with 'date' datetime column (ascending),
    and standard OHLCV column names if present. Add/overwrite `symbol`.
    """
    out = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(df)

    # unify date-like column to 'date'
    rename = {}
    for c in ("date", "Date", "datetime", "time", "timestamp"):
        if c in out.columns:
            if c != "date":
                rename[c] = "date"
            break
    if rename:
        out = out.rename(columns=rename)
    if "date" not in out.columns:
        out.insert(0, "date", pd.NaT)

    # drop IB's barCount if present
    if "barCount" in out.columns:
        out = out.drop(columns=["barCount"])

    # Ensure expected columns exist
    for col in ("open", "high", "low", "close", "volume"):
        if col not in out.columns:
            out[col] = pd.Series([None] * len(out), dtype="float64")

    # Parse & sort
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # attach symbol
    out["symbol"] = symbol
    return out


def _merge_and_cache(symbol: str, timeframe: str, cached_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge cached_df + new_df on 'date', keep latest row per date, save to cache, return merged.
    """
    if cached_df is None or cached_df.empty:
        merged = new_df.copy()
    else:
        merged = pd.concat([cached_df, new_df], ignore_index=True)
        merged.drop_duplicates(subset=["date"], keep="last", inplace=True)
        merged.sort_values("date", inplace=True)

    save_cache(symbol, timeframe, merged)
    return merged


def fetch_historical_data(
    symbol: str,
    duration: str = "3 Y",
    bar_size: str = "1 day",
    what_to_show: str = "TRADES",
    use_rth: bool = True,
) -> pd.DataFrame:
    """
    Fetch OHLCV data for `symbol` using cached data + delta refresh.

    Steps:
      1. Load cache (if any).
      2. If cache is recent (<1 day gap), just use it.
      3. If cache exists but is old, request ONLY the missing days from IB.
      4. If no cache, request full history from IB.
      5. Normalize columns and return.
    """
    timeframe = _tf_from_bar_size(bar_size)

    # Try load cache
    try:
        cached_df = load_cache(symbol, timeframe)
    except Exception as e:
        print(f"[CACHE] Read error for {symbol} ({timeframe}): {e}")
        cached_df = None

    ib = get_ib()
    contract = Stock(symbol, "SMART", "USD")

    # --- Case 1: cache exists ---
    if cached_df is not None and not cached_df.empty:
        cached_df = _normalize_df(cached_df, symbol)

        # last_ts may be tz-naive; normalize both sides to naive-UTC for subtraction
        last_ts = cached_df["date"].iloc[-1]
        if last_ts.tzinfo is not None:
            last_ts_naive = last_ts.tz_convert("UTC").tz_localize(None)
        else:
            last_ts_naive = last_ts  # assume already UTC-like from IB/CSV

        now_utc = datetime.now(timezone.utc)
        now_naive = now_utc.replace(tzinfo=None)

        delta_days = (now_naive - last_ts_naive).days

        # If cache is basically up to date (<1 day gap) or in the future (clock skew), reuse cache
        if delta_days < 1:
            return cached_df

        # Otherwise, incremental fetch: just the missing tail
        durationStr = f"{delta_days + 2} D"  # safety buffer
        try:
            bars = ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=durationStr,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=1 if use_rth else 0,
                formatDate=1,
                keepUpToDate=False,
            )
            if bars:
                new_df = util.df(bars)
                new_df = _normalize_df(new_df, symbol)
                merged = _merge_and_cache(symbol, timeframe, cached_df, new_df)
                return merged
            else:
                # IB gave nothing new → still return cache
                return cached_df
        except Exception as e:
            print(f"[IB] Delta fetch failed for {symbol}: {e}")
            return cached_df

    # --- Case 2: no cache → full fetch ---
    try:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=1 if use_rth else 0,
            formatDate=1,
            keepUpToDate=False,
        )
    except Exception as e:
        print(f"[IB] Full fetch failed for {symbol}: {e}")
        return _empty_frame(symbol)

    if not bars:
        return _empty_frame(symbol)

    df = util.df(bars)
    df = _normalize_df(df, symbol)

    # write new cache
    try:
        save_cache(symbol, timeframe, df)
    except Exception as e:
        print(f"[CACHE] Write error for {symbol} ({timeframe}): {e}")

    return df





def _tf_from_bar_size(bar_size: str) -> str:
    if not isinstance(bar_size, str):
        return "daily"
    return _BARSIZE_TO_TF.get(bar_size.strip().lower(), bar_size.strip().lower())


def _empty_frame(symbol: str) -> pd.DataFrame:
    """Return an empty but schema-correct frame so downstream code won't KeyError."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime([], errors="coerce"),
            "open": pd.Series([], dtype="float64"),
            "high": pd.Series([], dtype="float64"),
            "low": pd.Series([], dtype="float64"),
            "close": pd.Series([], dtype="float64"),
            "volume": pd.Series([], dtype="float64"),
            "symbol": pd.Series([], dtype="object"),
        }
    ).assign(symbol=symbol)


# def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Ensure a consistent schema with 'date' datetime column (ascending),
#     and standard OHLCV column names if present.
#     """
#     out = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
#
#     # unify date-like column to 'date'
#     rename = {}
#     for c in ("date", "Date", "datetime", "time", "timestamp"):
#         if c in out.columns:
#             if c != "date":
#                 rename[c] = "date"
#             break
#     if rename:
#         out = out.rename(columns=rename)
#     if "date" not in out.columns:
#         out.insert(0, "date", pd.NaT)
#
#     # drop IB's barCount if present
#     if "barCount" in out.columns:
#         out = out.drop(columns=["barCount"])
#
#     # Ensure expected columns exist
#     for col in ("open", "high", "low", "close", "volume"):
#         if col not in out.columns:
#             out[col] = pd.Series([None] * len(out), dtype="float64")
#
#     # Parse & sort
#     out["date"] = pd.to_datetime(out["date"], errors="coerce")
#     out = out.sort_values("date").reset_index(drop=True)
#
#     return out


def _ensure_symbol_column(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    out = df.copy()
    out["symbol"] = symbol
    return out


# def fetch_historical_data(
#     symbol: str,
#     duration: str = "1 Y",
#     bar_size: str = "1 day",
#     what_to_show: str = "TRADES",
#     use_rth: bool = True,
# ) -> pd.DataFrame:
#     """
#     Fetch OHLCV data for `symbol` at the specified bar size.
#
#     Steps:
#       1) Read from cache (CACHE_DIR) via cache_utils if present.
#       2) If cache missing/empty, fetch from IB.
#       3) Normalize schema, attach `symbol`, cache successful fetch, and return.
#
#     Returns a DataFrame with columns at least:
#       ['date','open','high','low','close','volume','symbol']
#     May be empty, but columns will always exist (so callers won't KeyError).
#     """
#     timeframe = _tf_from_bar_size(bar_size)
#
#     # 1) Try cache
#     try:
#         cached = load_cache(symbol, timeframe)
#     except Exception as e:
#         print(f"[CACHE] Read error for {symbol} ({timeframe}): {e}")
#         cached = None
#
#     if isinstance(cached, pd.DataFrame) and not cached.empty:
#         cached = _normalize_df(cached)
#         return _ensure_symbol_column(cached, symbol)
#
#     # 2) Fetch from IB
#     ib = get_ib()
#     contract = Stock(symbol, "SMART", "USD")
#
#     try:
#         bars = ib.reqHistoricalData(
#             contract,
#             endDateTime="",
#             durationStr=duration,
#             barSizeSetting=bar_size,
#             whatToShow=what_to_show,
#             useRTH=1 if use_rth else 0,
#             formatDate=1,
#             keepUpToDate=False,
#         )
#     except Exception as e:
#         # Typical: Error 200 "No security definition..." etc.
#         print(f"[IB] Error for {symbol} ({timeframe}): {e}")
#         return _empty_frame(symbol)
#
#     # Convert to DataFrame
#     try:
#         df = util.df(bars)
#     except Exception as e:
#         print(f"[IB] Failed to convert bars to DataFrame for {symbol}: {e}")
#         return _empty_frame(symbol)
#
#     # Normalize schema & attach symbol
#     df = _normalize_df(df)
#     df = _ensure_symbol_column(df, symbol)
#
#     if df.empty:
#         print(f"[IB] No data returned for {symbol} ({timeframe}).")
#         return _empty_frame(symbol)
#
#     # 3) Save to cache (only when we have actual rows)
#     try:
#         save_cache(symbol, timeframe, df)
#     except Exception as e:
#         print(f"[CACHE] Save error for {symbol} ({timeframe}): {e}")
#
#     return df
