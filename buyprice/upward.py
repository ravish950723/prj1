import pandas as pd
import numpy as np


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

def detect_smc_accumulation_breakout(df):
    recent = df.tail(20)
    if len(recent) < 20:
        return False
    close_last = _as_scalar(recent['close'].iloc[-1], default=np.nan)
    hi = _as_scalar(recent['high'].max(), default=np.nan)
    lo = _as_scalar(recent['low'].min(), default=np.nan)
    tight_range = (np.isfinite(hi) and np.isfinite(lo) and np.isfinite(close_last) and (hi - lo) < 0.05 * close_last)
    breakout = _as_scalar(df.iloc[-1].get('close', np.nan), default=np.nan) > _as_scalar(recent['high'].max(), default=np.nan)
    volume_spike = _as_scalar(df.iloc[-1].get('volume', np.nan), default=np.nan) > 1.5 * _as_scalar(recent['volume'].mean(), default=np.nan)
    return tight_range and breakout and volume_spike

def detect_mean_reversion_buy(df):
    if df.empty:
        return False
    last = df.iloc[-1]
    return (
        last.get('close', np.nan) <= last.get('BB_lower', np.nan) and
        last.get('RSI_14', 100) < 40 and
        last.get('MACD_hist', 1) < 0
    )

def detect_bullish_engulfing(df):
    if len(df) < 2:
        return False
    prev, curr = df.iloc[-2], df.iloc[-1]
    return (
        prev['close'] < prev['open'] and
        curr['close'] > curr['open'] and
        curr['close'] > prev['open'] and
        curr['open'] < prev['close']
    )

def detect_hammer(df):
    if len(df) < 1:
        return False
    last = df.iloc[-1]
    body = abs(last['close'] - last['open'])
    lower_wick = last['open'] - last['low'] if last['open'] > last['close'] else last['close'] - last['low']
    upper_wick = last['high'] - last['close'] if last['close'] > last['open'] else last['high'] - last['open']
    return (
        lower_wick > 2 * body and
        upper_wick < body
    )

def compute_upward_trend(df):
    df["trend_strength"] = (
        (df["EMA_20"] > df["EMA_50"]) &
        (df["MACD"] > 0) &
        (df["RSI_14"] > 50)
    ).astype(int)
    return df

def compute_signal_score(df):
    score = 0.0
    signal_count = 0

    if detect_smc_accumulation_breakout(df):
        score += 0.05
        signal_count += 1

    if detect_mean_reversion_buy(df):
        score += 0.03
        signal_count += 1

    if detect_bullish_engulfing(df):
        score += 0.03
        signal_count += 1

    if detect_hammer(df):
        score += 0.03
        signal_count += 1

    df = compute_upward_trend(df)
    if df.iloc[-1].get("trend_strength", 0) == 1:
        score += 0.01
        signal_count += 1

    recommendation = "BUY" if score > 0.08 and signal_count >= 2 else "HOLD"
    return score, signal_count, recommendation