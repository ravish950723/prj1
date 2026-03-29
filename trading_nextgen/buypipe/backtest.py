# backtest.py

import math
import numpy as np
import pandas as pd


def _to_float(x):
    try:
        if x is None:
            return float("nan")
        if isinstance(x, str):
            xs = x.strip()
            if xs == "" or xs.upper() in {"N/A", "NA", "NONE", "NULL"}:
                return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def evaluate_backtest_accuracy(
    symbol: str,
    df: pd.DataFrame,
    buy_price,
    gain_threshold: float = 0.10,
    gain_thresh: float | None = None,
    use_close: bool = False,
    lookahead_days: int = 90,
):
    """
    Forward backtest proxy over the most recent `lookahead_days` bars.

    Returns (hit: bool, max_gain_pct: float, days_to_peak: int)

    - hit: any day where gain_pct >= threshold
    - max_gain_pct: max gain over period in percentage points (e.g. 12.3 means +12.3%)
    - days_to_peak: index within the lookahead window (0..N-1) where max gain occurs, else -1

    Robust to string placeholders (e.g., 'N/A') and mixed dtypes in OHLC.
    """
    try:
        if gain_thresh is not None:
            gain_threshold = float(gain_thresh)
        gain_threshold = float(gain_threshold)

        if df is None or df.empty:
            return (False, 0.0, -1)

        d = df.copy()

        # Normalize column names
        cols = {c.lower(): c for c in d.columns}
        close_col = cols.get("close", "close") if "close" in cols else None
        if close_col is None:
            return (False, 0.0, -1)

        # Coerce close to numeric
        d[close_col] = pd.to_numeric(d[close_col], errors="coerce")

        # Prefer current close as entry when requested
        if use_close:
            try:
                buy_price = d[close_col].iloc[-1]
            except Exception:
                pass

        bp = _to_float(buy_price)
        if not np.isfinite(bp) or bp <= 0:
            return (False, 0.0, -1)

        tail = d.tail(int(lookahead_days)).reset_index(drop=True)
        px = tail[close_col].astype(float)

        # gain series
        gain = (px - bp) / bp
        gain = pd.to_numeric(gain, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if gain.empty:
            return (False, 0.0, -1)

        hit = bool((gain >= gain_threshold).any())
        idx = int(gain.values.argmax())
        max_gain_pct = float(gain.iloc[idx] * 100.0)
        return (hit, max_gain_pct, idx)
    except Exception:
        return (False, 0.0, -1)
