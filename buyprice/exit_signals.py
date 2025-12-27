# exit_signals.py
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _safe_float(x: Any, default: float = 0.0) -> float:
    """Convert to finite float, else default."""
    try:
        if x is None:
            return default
        v = float(x)
        if not np.isfinite(v):
            return default
        return v
    except Exception:
        return default


def compute_atr_trailing_stop(df: pd.DataFrame, entry_price: float, atr_mult: float = 2.0) -> float:
    """
    Long-only ATR trailing stop computed from the latest bar.

    - Uses ATR_14 if present and valid.
    - Falls back to a simple 5% stop if ATR is missing/invalid or df empty.
    """
    entry_price = _safe_float(entry_price, default=np.nan)
    if not np.isfinite(entry_price) or entry_price <= 0:
        return float("nan")

    if df is None or df.empty or "ATR_14" not in df.columns:
        return round(entry_price * 0.95, 2)

    atr = _safe_float(df["ATR_14"].iloc[-1], default=np.nan)
    if not np.isfinite(atr) or atr <= 0:
        return round(entry_price * 0.95, 2)

    stop = entry_price - float(atr_mult) * atr
    return round(stop, 2)


def detect_trend_break_exit(df: pd.DataFrame) -> bool:
    """
    Exit if:
      - EMA_21 crosses below EMA_50, OR
      - HTF trend flips UP -> DOWN (if HTF_Trend column exists).

    All checks are scalar-safe (use iloc[-1]/iloc[-2]) to avoid pandas "truth value of a Series" errors.
    """
    if df is None or len(df) < 2:
        return False

    if "EMA_21" not in df.columns or "EMA_50" not in df.columns:
        cross_down = False
    else:
        ema21_now = _safe_float(df["EMA_21"].iloc[-1], default=np.nan)
        ema50_now = _safe_float(df["EMA_50"].iloc[-1], default=np.nan)
        ema21_prev = _safe_float(df["EMA_21"].iloc[-2], default=np.nan)
        ema50_prev = _safe_float(df["EMA_50"].iloc[-2], default=np.nan)

        cross_down = (
            np.isfinite(ema21_now) and np.isfinite(ema50_now) and np.isfinite(ema21_prev) and np.isfinite(ema50_prev)
            and (ema21_now < ema50_now) and (ema21_prev >= ema50_prev)
        )

    trend_flip = False
    if "HTF_Trend" in df.columns:
        htf_now = str(df["HTF_Trend"].iloc[-1]).strip().upper()
        htf_prev = str(df["HTF_Trend"].iloc[-2]).strip().upper()
        trend_flip = (htf_prev == "UP" and htf_now == "DOWN")

    return bool(cross_down or trend_flip)


def detect_ema21_loss_exit(df: pd.DataFrame, max_closes_below: int = 2, tolerance: float = 0.003) -> bool:
    """
    Exit if price has been below EMA_21 (with small tolerance) for N consecutive bars.

    All checks are vector-safe and end in a Python bool.
    """
    if df is None or df.empty or "EMA_21" not in df.columns or "close" not in df.columns:
        return False

    n = int(max_closes_below)
    if n <= 0 or len(df) < n:
        return False

    closes = pd.to_numeric(df["close"].tail(n), errors="coerce")
    ema21 = pd.to_numeric(df["EMA_21"].tail(n), errors="coerce")

    # if we have NaNs, can't assert consecutive condition
    if closes.isna().any() or ema21.isna().any():
        return False

    below = closes < (ema21 * (1.0 - float(tolerance)))
    return bool(below.all())


def detect_volatility_expansion_exit(df: pd.DataFrame, lookback: int = 20, threshold: float = 1.8) -> bool:
    """
    Exit when ATR suddenly spikes and the current bar is a large red candle.
    threshold is expansion factor vs median ATR over lookback.

    All checks are scalar-safe (use iloc) to avoid ambiguous Series truth.
    """
    if df is None or df.empty or "ATR_14" not in df.columns or "open" not in df.columns or "close" not in df.columns:
        return False

    lb = int(lookback)
    if len(df) < lb + 1:
        return False

    atr = pd.to_numeric(df["ATR_14"].tail(lb + 1), errors="coerce").replace([np.inf, -np.inf], np.nan)
    cur = _safe_float(atr.iloc[-1], default=np.nan)
    base = _safe_float(atr.iloc[:-1].median(skipna=True), default=np.nan)
    if not np.isfinite(cur) or not np.isfinite(base) or base <= 0:
        return False

    expansion = cur / base

    last = df.iloc[-1]
    o = _safe_float(last.get("open"), default=np.nan)
    c = _safe_float(last.get("close"), default=np.nan)
    if not np.isfinite(o) or not np.isfinite(c):
        return False

    body = o - c
    big_red = (c < o) and (body > 0.5 * cur)

    return bool(expansion >= float(threshold) and big_red)


def compute_exit_signals(df: pd.DataFrame, entry_price: float, atr_mult: float = 2.0) -> Dict[str, Any]:
    """
    Aggregate exits into a single decision record for the latest bar.

    Returns keys your Excel pipeline expects:
      - AtrTrailingStop
      - ExitNow
      - ExitReasons
    """
    atr_stop = compute_atr_trailing_stop(df, entry_price, atr_mult=atr_mult)
    trend_break = detect_trend_break_exit(df)
    ema_loss = detect_ema21_loss_exit(df)
    vol_exit = detect_volatility_expansion_exit(df)

    exit_now = bool(trend_break or ema_loss or vol_exit)

    reasons: List[str] = []
    if trend_break:
        reasons.append("Trend break (EMA21<EMA50 or HTF DOWN)")
    if ema_loss:
        reasons.append("Close below EMA21 for consecutive bars")
    if vol_exit:
        reasons.append("Volatility expansion red bar")

    return {
        "AtrTrailingStop": atr_stop,
        "ExitNow": exit_now,
        "ExitReasons": "; ".join(reasons) if reasons else "",
    }
