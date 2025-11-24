# macro_features.py
import numpy as np
import pandas as pd
from fetching import fetch_data_cached  # you already have this

# Simple cache so we don't hit IB for each symbol repeatedly
_MACRO_CACHE = {}


def _get_macro_series(symbol: str, bar_spec: str = "10 Y", bar_size: str = "1 day") -> pd.DataFrame:
    """
    Fetch a macro symbol once and cache. Assumes same 'date' column structure as your other data.
    """
    key = (symbol, bar_spec, bar_size)
    if key in _MACRO_CACHE:
        return _MACRO_CACHE[key]

    df = fetch_data_cached(symbol, bar_spec, bar_size)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    _MACRO_CACHE[key] = df
    return df


def enrich_with_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add macro context:
      - VIX level / regime
      - QQQ trend
      - TLT as rates proxy
      - symbol-level volatility regime
    """
    if "date" not in df.columns:
        return df

    base = df.copy()
    base["date"] = pd.to_datetime(base["date"])

    # --- VIX (market volatility proxy) ---
    try:
        vix = _get_macro_series("VIX")  # adjust if your IB symbol differs
        vix = vix[["date", "close"]].rename(columns={"close": "VIX_close"})
        base = base.merge(vix, on="date", how="left")
    except Exception:
        base["VIX_close"] = np.nan

    # --- QQQ (growth / tech trend proxy) ---
    try:
        qqq = _get_macro_series("QQQ")
        qqq = qqq[["date", "close"]].rename(columns={"close": "QQQ_close"})
        base = base.merge(qqq, on="date", how="left")
    except Exception:
        base["QQQ_close"] = np.nan

    # --- TLT (rates proxy) ---
    try:
        tlt = _get_macro_series("TLT")
        tlt = tlt[["date", "close"]].rename(columns={"close": "TLT_close"})
        base = base.merge(tlt, on="date", how="left")
    except Exception:
        base["TLT_close"] = np.nan

    # --- Symbol & macro vol / regimes ---
    # symbol realized vol
    sym_ret = base["close"].pct_change(fill_method=None)
    sym_vol_20 = sym_ret.rolling(20).std()
    sym_vol_250 = sym_ret.rolling(250).std()
    base["sym_vol_ratio"] = sym_vol_20 / (sym_vol_250 + 1e-9)

    # VIX vol ratio (using VIX levels as a proxy)
    vix_ret = base["VIX_close"].pct_change(fill_method=None)
    vix_vol_20 = vix_ret.rolling(20).std()
    vix_vol_250 = vix_ret.rolling(250).std()
    base["VIX_vol_ratio"] = vix_vol_20 / (vix_vol_250 + 1e-9)

    def _regime(x):
        if not np.isfinite(x):
            return 0
        if x >= 1.5:
            return 1   # high vol
        if x <= 0.7:
            return -1  # low vol
        return 0       # normal

    base["sym_vol_regime"] = base["sym_vol_ratio"].apply(_regime)
    base["VIX_vol_regime"] = base["VIX_vol_ratio"].apply(_regime)

    # --- QQQ trend features ---
    base["QQQ_ma_50"] = base["QQQ_close"].rolling(50).mean()
    base["QQQ_ma_200"] = base["QQQ_close"].rolling(200).mean()
    base["QQQ_trend_50_200"] = np.where(
        base["QQQ_ma_50"] > base["QQQ_ma_200"], 1,
        np.where(base["QQQ_ma_50"] < base["QQQ_ma_200"], -1, 0)
    )
    base["QQQ_ret_20"] = base["QQQ_close"].pct_change(20)
    base["TLT_ret_20"] = base["TLT_close"].pct_change(20)

    return base
