import pandas as pd
import numpy as np

from config import UPWARD_SIGNAL_WEIGHTS, VIX_ADAPTIVE_WEIGHTS

def detect_smc_accumulation_breakout(df):
    recent = df.tail(20)
    if len(recent) < 20:
        return False
    tight_range = recent['high'].max() - recent['low'].min() < 0.05 * recent['close'].iloc[-1]
    breakout = df.iloc[-1]['close'] > recent['high'].max()
    volume_spike = df.iloc[-1]['volume'] > 1.5 * recent['volume'].mean()
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

def _vix_bucket(vix_regime: str | None) -> str:
    if not vix_regime:
        return "NORMAL"
    vr = str(vix_regime).upper()
    # match common labels like HIGH / MEDIUM / LOW
    for k in ("HIGH", "MEDIUM", "LOW", "NORMAL"):
        if k in vr:
            return k
    return "NORMAL"

def _apply_vix_adaptation(weights: dict, vix_regime: str | None) -> dict:
    """Return a copy of weights with VIX-adaptive adjustments applied."""
    bucket = _vix_bucket(vix_regime)
    adapt = VIX_ADAPTIVE_WEIGHTS.get(bucket, VIX_ADAPTIVE_WEIGHTS.get("NORMAL", {}))
    wmult = float(adapt.get("weight_mult", 1.0))
    bt_mult = float(adapt.get("buy_threshold_mult", 1.0))
    ms_add = int(adapt.get("min_signals_add", 0))

    out = dict(weights)
    # scale pattern weights (not thresholds)
    for k in ("smc", "mean_reversion", "bullish_engulfing", "hammer", "trend_strength"):
        if k in out:
            out[k] = float(out.get(k, 0.0)) * wmult

    # thresholds become stricter in higher vol (bt_mult > 1)
    out["buy_threshold"] = float(out.get("buy_threshold", 0.0)) * bt_mult
    out["min_signals"] = int(out.get("min_signals", 1)) + ms_add
    return out

def compute_signal_score(signals: dict, vix_regime: str | None = None) -> float:
    """
    Computes composite upward signal score using configurable weights.
    Optionally applies VIX-adaptive adjustments when vix_regime is provided.
    """
    weights = _apply_vix_adaptation(UPWARD_SIGNAL_WEIGHTS, vix_regime) if vix_regime else dict(UPWARD_SIGNAL_WEIGHTS)

    score = 0.0
    signal_count = 0

    signal_map = {
        "smc": bool(signals.get("smc_breakout", False)),
        "mean_reversion": bool(signals.get("mean_reversion", False)),
        "bullish_engulfing": bool(signals.get("bullish_engulfing", False)),
        "hammer": bool(signals.get("hammer", False)),
    }

    for key, fired in signal_map.items():
        if fired:
            score += float(weights.get(key, 0.0))
            signal_count += 1

    trend_strength = float(signals.get("trend_strength", 0.0) or 0.0)
    if trend_strength > 0:
        score += float(weights.get("trend_strength", 0.0))

    # Minimum confirmation gate
    if signal_count < int(weights.get("min_signals", 1)):
        return 0.0

    return round(float(score), 3)
