# exit_signals.py
import numpy as np
import pandas as pd


# ================================
# Scalar/boolean safety helpers
# Avoid: "The truth value of a Series is ambiguous"
# ================================
def _as_scalar(x, default=None):
    """Convert Series/array/scalar to a float scalar (last element for Series)."""
    try:
        import numpy as _np
        if default is None:
            default = _np.nan
        import pandas as _pd
        if isinstance(x, _pd.Series):
            if x.empty:
                return default
            x = x.iloc[-1]
        elif isinstance(x, _np.ndarray):
            if x.size == 0:
                return default
            x = x.reshape(-1)[-1]
        elif isinstance(x, (list, tuple)):
            if len(x) == 0:
                return default
            x = x[-1]
        if x is None:
            return default
        v = float(x)
        if _np.isfinite(v):
            return v
        return default
    except Exception:
        return default

def _as_bool(x, default=False):
    """Convert Series/array/scalar to bool (last element for Series)."""
    try:
        import numpy as _np
        import pandas as _pd
        if isinstance(x, _pd.Series):
            if x.empty:
                return default
            x = x.iloc[-1]
        elif isinstance(x, _np.ndarray):
            if x.size == 0:
                return default
            x = x.reshape(-1)[-1]
        elif isinstance(x, (list, tuple)):
            if len(x) == 0:
                return default
            x = x[-1]
        if x is None:
            return default
        if isinstance(x, (float, _np.floating)) and _np.isnan(x):
            return default
        return bool(x)
    except Exception:
        return default


def compute_atr_trailing_stop(df: pd.DataFrame, entry_price: float, atr_mult: float = 2.0) -> float:
    """
    Long-only ATR trailing stop from latest bar.
    If ATR is missing, falls back to a simple 5% stop.
    """
    if "ATR_14" not in df.columns or len(df) == 0:
        return round(entry_price * 0.95, 2)

    atr = float(df["ATR_14"].iloc[-1])
    if not np.isfinite(atr) or atr <= 0:
        return round(entry_price * 0.95, 2)

    stop = entry_price - atr_mult * atr
    return round(stop, 2)


def detect_trend_break_exit(df: pd.DataFrame) -> bool:
    """
    Exit if:
      - EMA_21 crosses below EMA_50, OR
      - HTF trend flips UP -> DOWN.
    """
    if len(df) < 2:
        return False

    ema21 = df["EMA_21"]
    ema50 = df["EMA_50"]
    cross_down = (ema21.iloc[-1] < ema50.iloc[-1]) and (ema21.iloc[-2] >= ema50.iloc[-2])

    if "HTF_Trend" in df.columns:
        htf_now = str(df["HTF_Trend"].iloc[-1])
        htf_prev = str(df["HTF_Trend"].iloc[-2])
        trend_flip = (htf_prev == "UP" and htf_now == "DOWN")
    else:
        trend_flip = False

    return bool(cross_down or trend_flip)


def detect_ema21_loss_exit(df: pd.DataFrame, max_closes_below: int = 2, tolerance: float = 0.003) -> bool:
    """
    Exit if price has been below EMA_21 (with small tolerance) for N consecutive bars.
    """
    if "EMA_21" not in df.columns or len(df) < max_closes_below:
        return False

    closes = df["close"].tail(max_closes_below)
    ema21 = df["EMA_21"].tail(max_closes_below)
    below = closes < (ema21 * (1 - tolerance))
    return bool(below.all())


def detect_volatility_expansion_exit(df: pd.DataFrame, lookback: int = 20, threshold: float = 1.8) -> bool:
    """
    Exit when ATR suddenly spikes and the current bar is a large red candle.
    threshold is expansion factor vs median ATR over lookback.
    """
    if "ATR_14" not in df.columns or len(df) < lookback + 1:
        return False

    atr = df["ATR_14"].tail(lookback + 1)
    cur = float(atr.iloc[-1])
    base = float(atr.iloc[:-1].median())
    if not np.isfinite(cur) or base <= 0:
        return False

    expansion = cur / base
    last = df.iloc[-1]
    body = last["open"] - last["close"]
    big_red = (last["close"] < last["open"]) and (body > 0.5 * cur)

    return bool(expansion >= threshold and big_red)


def compute_exit_signals(df: pd.DataFrame, entry_price: float, atr_mult: float = 2.0) -> dict:
    """
    Aggregate exits into a single decision record for the latest bar.
    """
    atr_stop = compute_atr_trailing_stop(df, entry_price, atr_mult=atr_mult)
    trend_break = detect_trend_break_exit(df)
    ema_loss = detect_ema21_loss_exit(df)
    vol_exit = detect_volatility_expansion_exit(df)

    exit_now = _as_bool(trend_break) or _as_bool(ema_loss) or _as_bool(vol_exit)
    reasons = []
    if trend_break:
        reasons.append("Trend break (EMA21<EMA50 or HTF DOWN)")
    if ema_loss:
        reasons.append("Close below EMA21 for consecutive bars")
    if vol_exit:
        reasons.append("Volatility expansion red bar")

    return {
        "AtrTrailingStop": atr_stop,
        "ExitNow": bool(exit_now),
        "ExitReasons": "; ".join(reasons) if reasons else "",
    }