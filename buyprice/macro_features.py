# macro_features.py
import numpy as np
import pandas as pd
from fetching import fetch_data_cached  # your cached IBKR fetch

# Simple cache so we don't hit IB for each symbol repeatedly
_MACRO_CACHE = {}


def _get_macro_series(symbol: str, bar_spec: str = "10 Y", bar_size: str = "1 day") -> pd.DataFrame:
    """
    Fetch a macro symbol once and cache.
    IMPORTANT:
      - require_today=False so weekends/holidays don't break macro enrichment
      - date normalized via fetching.py (tz-naive); we still coerce defensively here
    """
    key = (symbol, bar_spec, bar_size)
    if key in _MACRO_CACHE:
        return _MACRO_CACHE[key]

    df = fetch_data_cached(symbol, bar_spec, bar_size, require_today=False)
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    _MACRO_CACHE[key] = df
    return df


def enrich_with_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add macro context:
      - VIX level / regime
      - QQQ trend
      - TLT as rates proxy
      - symbol-level volatility regime

    This function is BULLETPROOF:
      - It will NEVER raise KeyError for missing macro columns.
      - If any macro series fails to fetch, the corresponding columns are created as NaN.
    """
    if df is None or df.empty or "date" not in df.columns:
        return df

    base = df.copy()
    base["date"] = pd.to_datetime(base["date"], errors="coerce")

    # --- VIX (market volatility proxy) ---
    try:
        vix = _get_macro_series("VIX")  # adjust if your IB symbol differs
        if vix is not None and not vix.empty and {"date", "close"}.issubset(vix.columns):
            vix = vix[["date", "close"]].rename(columns={"close": "VIX_close"})
            base = base.merge(vix, on="date", how="left")
    except Exception:
        pass

    # --- QQQ (growth / tech trend proxy) ---
    try:
        qqq = _get_macro_series("QQQ")
        if qqq is not None and not qqq.empty and {"date", "close"}.issubset(qqq.columns):
            qqq = qqq[["date", "close"]].rename(columns={"close": "QQQ_close"})
            base = base.merge(qqq, on="date", how="left")
    except Exception:
        pass

    # --- TLT (rates proxy) ---
    try:
        tlt = _get_macro_series("TLT")
        if tlt is not None and not tlt.empty and {"date", "close"}.issubset(tlt.columns):
            tlt = tlt[["date", "close"]].rename(columns={"close": "TLT_close"})
            base = base.merge(tlt, on="date", how="left")
    except Exception:
        pass

    # Ensure macro columns exist even if merges failed (prevents KeyError downstream)
    for c in ("VIX_close", "QQQ_close", "TLT_close"):
        if c not in base.columns:
            base[c] = np.nan

    # --- Symbol & macro vol / regimes ---
    # symbol realized vol
    sym_ret = pd.to_numeric(base.get("close", np.nan), errors="coerce").pct_change(fill_method=None)
    sym_vol_20 = sym_ret.rolling(20).std()
    sym_vol_250 = sym_ret.rolling(250).std()
    base["sym_vol_ratio"] = sym_vol_20 / (sym_vol_250 + 1e-9)

    # VIX vol ratio (using VIX levels as a proxy)
    vix_series = pd.to_numeric(base["VIX_close"], errors="coerce")
    vix_ret = vix_series.pct_change(fill_method=None)
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
    qqq_series = pd.to_numeric(base["QQQ_close"], errors="coerce")
    base["QQQ_ma_50"] = qqq_series.rolling(50).mean()
    base["QQQ_ma_200"] = qqq_series.rolling(200).mean()
    base["QQQ_trend_50_200"] = np.where(
        base["QQQ_ma_50"] > base["QQQ_ma_200"], 1,
        np.where(base["QQQ_ma_50"] < base["QQQ_ma_200"], -1, 0)
    )
    base["QQQ_ret_20"] = qqq_series.pct_change(20, fill_method=None)

    tlt_series = pd.to_numeric(base["TLT_close"], errors="coerce")
    base["TLT_ret_20"] = tlt_series.pct_change(20, fill_method=None)

    return base
