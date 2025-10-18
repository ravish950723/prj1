
"""
selling_price.py

Compute recommended SELL (take-profit / exit) prices from weekly candle structure
over configurable windows (6, 8, 12, 18, 30 weeks). The heuristic is simple,
transparent, and robust:

1) Resample to weekly OHLCV (W-FRI).
2) Consider the last N *completed* weeks (excludes the in-progress week).
3) Find the most recent pivot high (a local maximum). If unavailable, fallback to
   the highest close in the window.
4) Use weekly ATR as a buffer to avoid missing by a few cents:
   - 6w/8w: buffer = 0.15 * weekly_ATR
   - 12w/18w/30w: buffer = 0.25 * weekly_ATR
5) If a clear bearish reversal prints within ~2% of the pivot resistance
   (bearish engulfing or a long upper-wick "shooting star"), tighten the target.

Returns a float (recommended exit price), or None if not computable.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

# ---------- Helpers ----------

def _to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DatetimeIndex, normalize column names, and resample to weekly (W-FRI).
    Required columns: open, high, low, close, volume
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        else:
            raise ValueError("DataFrame must be indexed by DatetimeIndex or have a 'date' column")

    # Normalize to lowercase columns
    df = df.rename(columns={c: c.lower() for c in df.columns})

    needed = {'open', 'high', 'low', 'close', 'volume'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    wk = pd.DataFrame({
        'open': df['open'].resample('W-FRI').first(),
        'high': df['high'].resample('W-FRI').max(),
        'low': df['low'].resample('W-FRI').min(),
        'close': df['close'].resample('W-FRI').last(),
        'volume': df['volume'].resample('W-FRI').sum(),
    }).dropna()
    return wk

def _true_range(h, l, c_prev):
    return np.maximum(h - l, np.maximum(abs(h - c_prev), abs(l - c_prev)))

def _weekly_atr(dfw: pd.DataFrame, period: int = 14) -> pd.Series:
    c_prev = dfw['close'].shift(1)
    tr = _true_range(dfw['high'], dfw['low'], c_prev)
    atr = pd.Series(tr, index=dfw.index).rolling(
        window=period, min_periods=max(2, period // 2)
    ).mean()
    return atr

def _pivot_highs(dfw: pd.DataFrame, left: int = 1, right: int = 1) -> pd.Series:
    """
    Mark pivot-highs where high[i] is greater than highs of left/right neighbors.
    """
    highs = dfw['high'].to_numpy()
    piv = np.zeros_like(highs, dtype=bool)
    for i in range(left, len(highs) - right):
        left_blk = highs[i-left:i]
        right_blk = highs[i+1:i+1+right]
        if left_blk.size and right_blk.size and highs[i] > left_blk.max() and highs[i] > right_blk.max():
            piv[i] = True
    return pd.Series(piv, index=dfw.index)

def _bearish_reversal_last_week(dfw: pd.DataFrame) -> bool:
    """
    Detect a basic bearish reversal on the most recent *completed* week:
     - Bearish Engulfing
     - Shooting Star (long upper wick, small body near low)
    """
    if len(dfw) < 3:
        return False
    w = dfw.iloc[-1]
    p = dfw.iloc[-2]

    bearish_engulf = (
        (w['close'] < w['open']) and
        (p['close'] > p['open']) and
        (w['open'] >= p['close']) and
        (w['close'] <= p['open'])
    )

    body = abs(w['close'] - w['open'])
    upper_wick = w['high'] - max(w['close'], w['open'])
    lower_wick = min(w['close'], w['open']) - w['low']
    denom = body if body != 0 else 1e-9
    shooting_star = (upper_wick / denom >= 2.0) and (lower_wick <= body * 0.5) and (
        w['close'] < (w['low'] + 0.35 * (w['high'] - w['low']))
    )

    return bool(bearish_engulf or shooting_star)

# ---------- Core ----------

def get_selling_price_by_weeks(dfd: pd.DataFrame, weeks: int) -> float | None:
    """
    Compute a recommended SELL price using the last `weeks` completed weekly candles.
    `dfd` must include columns: ['open','high','low','close','volume'] and be daily or intraday with DatetimeIndex.
    """
    if weeks < 3:
        raise ValueError("weeks must be >= 3")

    dfw = _to_weekly(dfd)
    if len(dfw) < weeks + 2:  # need context for reversals/ATR
        return None

    # Exclude in-progress week; use fully closed weeks
    today = pd.Timestamp.today().normalize()
    dfw_closed = dfw.iloc[:-1] if dfw.index[-1] >= today else dfw
    if len(dfw_closed) < weeks:
        dfw_closed = dfw

    window = dfw_closed.iloc[-weeks:]

    # Weekly ATR buffer
    atr_series = _weekly_atr(dfw_closed, period=14)
    atr_recent = float(atr_series.iloc[-1]) if pd.notna(atr_series.iloc[-1]) else float(window['high'].iloc[-1] - window['low'].iloc[-1])
    if atr_recent <= 0:
        atr_recent = float((window['high'].max() - window['low'].min()) / max(weeks, 1))

    # Resistance from most-recent pivot high (fallback: highest close)
    piv = _pivot_highs(window, left=1, right=1)
    if piv.any():
        resistance = float(window.loc[piv, 'high'].iloc[-1])
    else:
        resistance = float(window['close'].max())

    # Buffer: smaller for 6/8w, larger for longer windows
    buf = 0.15 * atr_recent if weeks <= 8 else 0.25 * atr_recent
    target = resistance - buf

    # Tighten if bearish reversal near resistance
    if _bearish_reversal_last_week(window):
        last_close = float(window['close'].iloc[-1])
        if abs(last_close - resistance) / max(resistance, 1e-9) <= 0.02:
            tight = min(last_close + 0.5 * atr_recent, resistance - 0.1 * atr_recent)
            target = min(target, tight)

    # Guardrail: ensure target above last close
    last_close = float(window['close'].iloc[-1])
    if target <= last_close:
        target = max(resistance - 0.1 * atr_recent, last_close * 1.01)

    return float(round(target, 4))

# Convenience wrappers
def get_selling_price_6w(dfd):  return get_selling_price_by_weeks(dfd, 6)
def get_selling_price_8w(dfd):  return get_selling_price_by_weeks(dfd, 8)
def get_selling_price_12w(dfd): return get_selling_price_by_weeks(dfd, 12)
def get_selling_price_18w(dfd): return get_selling_price_by_weeks(dfd, 18)
def get_selling_price_30w(dfd): return get_selling_price_by_weeks(dfd, 30)
