"""rank_long_short.py

Long/Short ranking + entry/targets for your Excel pipeline.

Design goals:
  - Deterministic, explainable scoring (0..100)
  - Works even when some fields are missing
  - Consumes daily OHLCV with indicator columns (or raw OHLCV; it will compute what it needs)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import re


# ---------------------------
# Utilities
# ---------------------------

def _f(x, default=np.nan) -> float:
    try:
        if x is None:
            return float(default)
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _b(x, default: bool = False) -> bool:
    try:
        if x is None:
            return default
        if isinstance(x, (bool, np.bool_)):
            return bool(x)
        if isinstance(x, (int, float, np.integer, np.floating)):
            if not np.isfinite(float(x)):
                return default
            return bool(int(float(x)) != 0)
        s = str(x).strip().lower()
        if s in {"true", "t", "yes", "y", "1", "✅"}:
            return True
        if s in {"false", "f", "no", "n", "0", "❌"}:
            return False
        return default
    except Exception:
        return default


def sma(series: pd.Series, n: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").rolling(n, min_periods=n).mean()


def pct_from(price: float, ma: float) -> float:
    if not np.isfinite(price) or not np.isfinite(ma) or ma == 0:
        return np.nan
    return (price - ma) / ma * 100.0


def dma_stack(d20: float, d50: float, d100: float, d200: float) -> str:
    if not all(np.isfinite(x) for x in (d20, d50, d100, d200)):
        return "MIXED"
    if d20 > d50 > d100 > d200:
        return "BULL_STACKED"
    if d20 < d50 < d100 < d200:
        return "BEAR_STACKED"
    return "MIXED"


def volume_metrics(df: pd.DataFrame) -> Tuple[float, float, float, float]:
    vol = pd.to_numeric(df.get("volume"), errors="coerce")
    if vol is None or vol.empty:
        return np.nan, np.nan, np.nan, np.nan
    v_today = _f(vol.iloc[-1])
    avg20 = _f(vol.rolling(20).mean().iloc[-1])
    surge = (v_today / avg20) if (np.isfinite(v_today) and np.isfinite(avg20) and avg20 > 0) else np.nan
    v5 = _f(vol.tail(5).mean())
    v20 = _f(vol.tail(20).mean())
    trend5 = (v5 / v20) if (np.isfinite(v5) and np.isfinite(v20) and v20 > 0) else np.nan
    return v_today, avg20, surge, trend5


def gap_pct(df: pd.DataFrame) -> float:
    if df is None or len(df) < 2:
        return np.nan
    o = _f(df.get("open", pd.Series([np.nan])).iloc[-1])
    prev_c = _f(df.get("close", pd.Series([np.nan])).iloc[-2])
    if not np.isfinite(o) or not np.isfinite(prev_c) or prev_c == 0:
        return np.nan
    return (o - prev_c) / prev_c * 100.0


def rel_strength_20d_vs_qqq(sym_df: pd.DataFrame, qqq_df: Optional[pd.DataFrame]) -> float:
    """20D relative strength proxy: (sym 20D return - qqq 20D return) in %."""
    try:
        if sym_df is None or sym_df.empty or qqq_df is None or qqq_df.empty:
            return np.nan

        s = pd.to_numeric(sym_df["close"], errors="coerce").dropna()
        q = pd.to_numeric(qqq_df["close"], errors="coerce").dropna()
        if len(s) < 21 or len(q) < 21:
            return np.nan

        s_ret = (s.iloc[-1] / s.iloc[-21] - 1.0) * 100.0
        q_ret = (q.iloc[-1] / q.iloc[-21] - 1.0) * 100.0
        return float(s_ret - q_ret)
    except Exception:
        return np.nan


@dataclass
class RankConfig:
    # thresholds
    vol_surge_breakout: float = 2.0
    vol_surge_reversal: float = 1.5
    vol_pullback_max: float = 1.2
    rsi_breakout_min: float = 55
    rsi_pullback_min: float = 45
    rsi_short_max: float = 45
    gap_down_pct: float = -3.0
    overext_dma20_pct: float = 12.0
    long_score_buy_now: float = 70.0
    long_score_wait_dip: float = 60.0
    short_score_short_now: float = 70.0
    short_score_wait_bounce: float = 60.0
    borrow_fee_max: float = 20.0

    # weights (must sum ~1.0; we normalize anyway)
    w_trend: float = 0.25
    w_volume: float = 0.20
    w_momentum: float = 0.20
    w_volatility: float = 0.10
    w_rel_strength: float = 0.15
    w_liquidity: float = 0.10


def _normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    s = sum(max(0.0, float(v)) for v in w.values())
    if s <= 0:
        return {k: 1.0 / len(w) for k in w}
    return {k: max(0.0, float(v)) / s for k, v in w.items()}


def _score_0_100(x: float, lo: float, hi: float) -> float:
    """Clamp + scale to 0..100."""
    if not np.isfinite(x):
        return np.nan
    if hi == lo:
        return 50.0
    v = (x - lo) / (hi - lo)
    return float(np.clip(v, 0.0, 1.0) * 100.0)


def compute_common_indicators_rls(df: pd.DataFrame, qqq_df: Optional[pd.DataFrame] = None) -> Dict[str, object]:
    """Compute the columns required by the SRS (DMA, volume analytics, distances, trend flags, etc.)."""
    out: Dict[str, object] = {}

    if df is None or df.empty:
        return out

    # ---------------------------
    # Normalize schema (IBKR vs AlphaVantage)
    # ---------------------------
    df = df.copy()
    # If date is index, reset so downstream code can use columns safely.
    if "date" not in df.columns and df.index is not None:
        try:
            df = df.reset_index()
        except Exception:
            pass

    # Ensure required OHLCV columns exist and are numeric.
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    # Guard: after coercion/dropna we may end up with an empty frame (e.g., bad fetch/cache or all-NaN close).
    if df.empty:
        return out


    # Helper indicator fallbacks (only used if compute.py didn't precompute)
    def _ema(s: pd.Series, span: int) -> pd.Series:
        return pd.to_numeric(s, errors="coerce").ewm(span=span, adjust=False, min_periods=span).mean()

    def _rsi_wilder(close_s: pd.Series, period: int = 14) -> pd.Series:
        c = pd.to_numeric(close_s, errors="coerce")
        delta = c.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    def _atr(df_in: pd.DataFrame, period: int = 14) -> pd.Series:
        h = pd.to_numeric(df_in["high"], errors="coerce")
        l = pd.to_numeric(df_in["low"], errors="coerce")
        c = pd.to_numeric(df_in["close"], errors="coerce")
        prev_c = c.shift(1)
        tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        return tr.rolling(period, min_periods=period).mean()

    close = pd.to_numeric(df.get("close"), errors="coerce").dropna()
    if close.empty:
        return out
    price = _f(close.iloc[-1])
    out["PRICE"] = price

    d20 = _f(sma(close, 20).iloc[-1])
    d50 = _f(sma(close, 50).iloc[-1])
    d100 = _f(sma(close, 100).iloc[-1])
    d150 = _f(sma(close, 150).iloc[-1])
    d200 = _f(sma(close, 200).iloc[-1])

    out.update({
        "DMA20": d20,
        "DMA50": d50,
        "DMA100": d100,
        "DMA150": d150,
        "DMA200": d200,
        "PCT_FROM_DMA20": pct_from(price, d20),
        "PCT_FROM_DMA50": pct_from(price, d50),
        "PCT_FROM_DMA200": pct_from(price, d200),
        "DMA_STACK": dma_stack(d20, d50, d100, d200),
        "ABOVE_DMA20": bool(np.isfinite(price) and np.isfinite(d20) and price > d20),
        "ABOVE_DMA50": bool(np.isfinite(price) and np.isfinite(d50) and price > d50),
        "ABOVE_DMA200": bool(np.isfinite(price) and np.isfinite(d200) and price > d200),
    })

    v_today, avg20, surge, trend5 = volume_metrics(df)
    out.update({
        "VOL_TODAY": v_today,
        "AVG_VOL_20D": avg20,
        "VOL_SURGE_RATIO": surge,
        "VOL_TREND_5D": trend5,
        "GAP_PCT": gap_pct(df),
    })

    # ---------------------------
    # Momentum + volatility indicators
    # Prefer precomputed columns from compute.py, else compute locally.
    # ---------------------------

    # RSI14
    if "RSI_14" in df.columns:
        out["RSI14"] = _f(df["RSI_14"].iloc[-1])
    else:
        out["RSI14"] = _f(_rsi_wilder(close, 14).iloc[-1])

    # MACD (12,26,9)
    if "MACD" in df.columns and "MACD_signal" in df.columns and "MACD_hist" in df.columns:
        out["MACD"] = _f(df["MACD"].iloc[-1])
        out["MACD_SIGNAL"] = _f(df["MACD_signal"].iloc[-1])
        out["MACD_HIST"] = _f(df["MACD_hist"].iloc[-1])
    else:
        ema12 = _ema(close, 12)
        ema26 = _ema(close, 26)
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
        hist = macd - signal
        out["MACD"] = _f(macd.iloc[-1])
        out["MACD_SIGNAL"] = _f(signal.iloc[-1])
        out["MACD_HIST"] = _f(hist.iloc[-1])

    # ATR14
    if "ATR_14" in df.columns:
        out["ATR14"] = _f(df["ATR_14"].iloc[-1])
    else:
        out["ATR14"] = _f(_atr(df, 14).iloc[-1])

    out["ATR14_PCT"] = (out["ATR14"] / price * 100.0) if (np.isfinite(out["ATR14"]) and np.isfinite(price) and price > 0) else np.nan

    # Daily VWAP proxy (20D rolling: sum(tp*vol)/sum(vol))
    try:
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        vol = df["volume"].replace(0.0, np.nan)
        vwap20 = (tp * vol).rolling(20, min_periods=10).sum() / vol.rolling(20, min_periods=10).sum()
        vwap_val = _f(vwap20.iloc[-1])
    except Exception:
        vwap_val = np.nan
    out["VWAP"] = vwap_val
    out["VWAP_DISTANCE_PCT"] = pct_from(price, vwap_val)

    # Distribution / Accumulation days (20D): high-volume red/green days
    try:
        c = df["close"]
        prev_c = c.shift(1)
        day_ret = (c - prev_c)
        vol = df["volume"]
        avg20 = vol.rolling(20, min_periods=10).mean()
        high_vol = vol >= (1.2 * avg20)
        dist_days = int(((day_ret < 0) & high_vol).tail(20).sum())
        accu_days = int(((day_ret > 0) & high_vol).tail(20).sum())
        out["DISTRIBUTION_DAYS_20D"] = dist_days
        out["ACCUMULATION_DAYS_20D"] = accu_days
        if accu_days > dist_days + 1:
            out["VOLUME_SIGNAL"] = "ACCUMULATION"
        elif dist_days > accu_days + 1:
            out["VOLUME_SIGNAL"] = "DISTRIBUTION"
        else:
            out["VOLUME_SIGNAL"] = "NEUTRAL"
    except Exception:
        out["DISTRIBUTION_DAYS_20D"] = np.nan
        out["ACCUMULATION_DAYS_20D"] = np.nan
        out["VOLUME_SIGNAL"] = "NEUTRAL"

    out["REL_STRENGTH_20D_VS_QQQ"] = rel_strength_20d_vs_qqq(df, qqq_df)

    return out


# ---------------------------
# Setup detection
# ---------------------------

def detect_long_setup(common: Dict[str, object], df: pd.DataFrame, cfg: RankConfig) -> Optional[str]:
    """Return L1/L2/L3 or None."""
    price = _f(common.get("PRICE"))
    d20 = _f(common.get("DMA20"))
    d50 = _f(common.get("DMA50"))
    d200 = _f(common.get("DMA200"))
    surge = _f(common.get("VOL_SURGE_RATIO"))
    rsi = _f(common.get("RSI14"))
    macdh = _f(common.get("MACD_HIST"))
    above50 = _b(common.get("ABOVE_DMA50"))
    above200 = _b(common.get("ABOVE_DMA200"))
    stack = str(common.get("DMA_STACK") or "MIXED")

    # Need at least a small history for pivot logic
    if df is None or len(df) < 60:
        return None
    highs = pd.to_numeric(df.get("high"), errors="coerce")
    closes = pd.to_numeric(df.get("close"), errors="coerce")
    if highs.isna().all() or closes.isna().all():
        return None

    # Pivot/resistance: 20-day prior high (excluding today)
    prior_res = _f(highs.shift(1).rolling(20).max().iloc[-1])

    # L1 Breakout + volume
    l1 = (
        np.isfinite(prior_res) and np.isfinite(price) and price > prior_res and
        np.isfinite(surge) and surge >= cfg.vol_surge_breakout and
        np.isfinite(rsi) and rsi >= cfg.rsi_breakout_min and
        np.isfinite(macdh) and macdh >= 0 and
        (stack in {"BULL_STACKED", "MIXED"}) and above50
    )
    if l1:
        return "L1"

    # L2 Pullback into 20-50 band, low volume
    in_20_50_band = (np.isfinite(price) and np.isfinite(d20) and np.isfinite(d50) and (min(d20, d50) <= price <= max(d20, d50)))
    l2 = (
        above200 and in_20_50_band and
        np.isfinite(surge) and surge < cfg.vol_pullback_max and
        np.isfinite(rsi) and rsi >= cfg.rsi_pullback_min
    )
    if l2:
        return "L2"

    # L3 Reversal: cross above DMA50 with volume
    prev_close = _f(closes.iloc[-2])
    l3 = (
        np.isfinite(d50) and np.isfinite(prev_close) and np.isfinite(price) and
        (prev_close < d50 <= price) and
        np.isfinite(surge) and surge >= cfg.vol_surge_reversal and
        np.isfinite(macdh) and macdh >= 0
    )
    if l3:
        return "L3"

    return None


def detect_short_setup(common: Dict[str, object], df: pd.DataFrame, cfg: RankConfig) -> Optional[str]:
    price = _f(common.get("PRICE"))
    d20 = _f(common.get("DMA20"))
    d50 = _f(common.get("DMA50"))
    surge = _f(common.get("VOL_SURGE_RATIO"))
    rsi = _f(common.get("RSI14"))
    macdh = _f(common.get("MACD_HIST"))
    above50 = _b(common.get("ABOVE_DMA50"))
    gap = _f(common.get("GAP_PCT"))
    pct_dma20 = _f(common.get("PCT_FROM_DMA20"))

    if df is None or len(df) < 60:
        return None

    lows = pd.to_numeric(df.get("low"), errors="coerce")
    closes = pd.to_numeric(df.get("close"), errors="coerce")
    if lows.isna().all() or closes.isna().all():
        return None

    # S1 Breakdown below DMA50 + distribution
    s1 = (
        (not above50) and
        np.isfinite(surge) and surge >= 1.8 and
        np.isfinite(rsi) and rsi <= cfg.rsi_short_max and
        np.isfinite(macdh) and macdh < 0
    )
    if s1 or (np.isfinite(gap) and gap <= cfg.gap_down_pct and s1):
        return "S1"

    # S2 Failed breakout / bull trap: close back below prior 20D high
    prior_res = _f(pd.to_numeric(df.get("high"), errors="coerce").shift(1).rolling(20).max().iloc[-1])
    was_above = bool(np.isfinite(prior_res) and np.isfinite(closes.iloc[-2]) and closes.iloc[-2] > prior_res)
    fell_back = bool(np.isfinite(prior_res) and np.isfinite(price) and price < prior_res)
    s2 = (
        was_above and fell_back and
        np.isfinite(surge) and surge >= 1.5 and
        np.isfinite(rsi) and rsi < 50 and
        np.isfinite(macdh) and macdh < 0
    )
    if s2:
        return "S2"

    # S3 Overextended from DMA20 and momentum fading
    s3 = (
        np.isfinite(pct_dma20) and pct_dma20 > cfg.overext_dma20_pct and
        np.isfinite(rsi) and rsi >= 70 and
        np.isfinite(macdh)
    )
    if s3:
        return "S3"

    return None


# ---------------------------
# Scoring + trade plan strings
# ---------------------------

def _liquidity_score(avg_vol20: float, spread_pct: float) -> float:
    """0..100; higher is better."""
    # volume: 100k -> 20, 500k -> 50, 2M -> 80, 10M -> 100
    v = _f(avg_vol20)
    sp = _f(spread_pct)
    vol_score = np.nan
    if np.isfinite(v):
        vol_score = float(np.clip(np.log10(max(v, 1.0)) - 5.0, 0.0, 2.0) / 2.0 * 100.0)  # 1e5..1e7
    # spread: 0.05% -> 100, 0.20% -> 70, 0.50% -> 40, 1% -> 20
    spread_score = np.nan
    if np.isfinite(sp):
        spread_score = float(np.clip(0.5 - sp, 0.0, 0.5) / 0.5 * 100.0)
    if not np.isfinite(vol_score) and not np.isfinite(spread_score):
        return np.nan
    if not np.isfinite(vol_score):
        return spread_score
    if not np.isfinite(spread_score):
        return vol_score
    return float(0.6 * vol_score + 0.4 * spread_score)


def liquidity_score(avg_vol20: float, spread_pct: float = np.nan) -> float:
    """Public wrapper for liquidity scoring (0..100; higher is better)."""
    return _liquidity_score(avg_vol20, spread_pct)


def score_long(common: Dict[str, object], cfg: RankConfig, bid_ask_spread_pct: float = np.nan) -> float:
    w = _normalize_weights({
        "trend": cfg.w_trend,
        "volume": cfg.w_volume,
        "momentum": cfg.w_momentum,
        "vol": cfg.w_volatility,
        "rs": cfg.w_rel_strength,
        "liq": cfg.w_liquidity,
    })

    stack = str(common.get("DMA_STACK") or "MIXED")
    above200 = _b(common.get("ABOVE_DMA200"))
    trend_raw = 0.0
    if stack == "BULL_STACKED":
        trend_raw = 1.0
    elif stack == "MIXED":
        trend_raw = 0.6
    else:
        trend_raw = 0.2
    if above200:
        trend_raw = min(1.0, trend_raw + 0.15)
    trend = trend_raw * 100.0

    surge = _f(common.get("VOL_SURGE_RATIO"))
    volume = _score_0_100(surge, lo=0.8, hi=2.5)  # reward 1.0..2.5

    rsi = _f(common.get("RSI14"))
    macdh = _f(common.get("MACD_HIST"))
    momentum = np.nan
    if np.isfinite(rsi) and np.isfinite(macdh):
        momentum = float(np.clip((rsi - 40.0) / 30.0, 0.0, 1.0) * 60.0 + (np.clip(macdh, -0.5, 0.5) + 0.5) * 40.0)
    elif np.isfinite(rsi):
        momentum = float(np.clip((rsi - 40.0) / 30.0, 0.0, 1.0) * 100.0)

    atrp = _f(common.get("ATR14_PCT"))
    # Prefer moderate ATR% (2..6). Too low = sleepy, too high = whipsaw.
    vol_suit = np.nan
    if np.isfinite(atrp):
        vol_suit = 100.0 - abs(atrp - 4.0) * 18.0
        vol_suit = float(np.clip(vol_suit, 0.0, 100.0))

    rs = _f(common.get("REL_STRENGTH_20D_VS_QQQ"))
    rs_score = _score_0_100(rs, lo=-10.0, hi=10.0)

    liq = _liquidity_score(_f(common.get("AVG_VOL_20D")), bid_ask_spread_pct)

    parts = {
        "trend": trend,
        "volume": volume,
        "momentum": momentum,
        "vol": vol_suit,
        "rs": rs_score,
        "liq": liq,
    }
    total = 0.0
    wsum = 0.0
    for k, v in parts.items():
        if np.isfinite(_f(v)):
            total += w[k] * float(v)
            wsum += w[k]
    if wsum <= 0:
        return np.nan
    return float(np.clip(total / wsum, 0.0, 100.0))


def score_short(common: Dict[str, object], cfg: RankConfig, bid_ask_spread_pct: float = np.nan,
                shortable_flag: Optional[bool] = None, borrow_fee_pct: float = np.nan) -> float:
    # Use same weights but interpret trend inversely + RS inversely
    w = _normalize_weights({
        "trend": cfg.w_trend,
        "volume": cfg.w_volume,
        "momentum": cfg.w_momentum,
        "vol": cfg.w_volatility,
        "rs": cfg.w_rel_strength,
        "liq": cfg.w_liquidity,
    })

    stack = str(common.get("DMA_STACK") or "MIXED")
    above200 = _b(common.get("ABOVE_DMA200"))
    trend_raw = 0.0
    if stack == "BEAR_STACKED":
        trend_raw = 1.0
    elif stack == "MIXED":
        trend_raw = 0.6
    else:
        trend_raw = 0.2
    if not above200:
        trend_raw = min(1.0, trend_raw + 0.15)
    trend = trend_raw * 100.0

    surge = _f(common.get("VOL_SURGE_RATIO"))
    volume = _score_0_100(surge, lo=0.8, hi=2.5)

    rsi = _f(common.get("RSI14"))
    macdh = _f(common.get("MACD_HIST"))
    momentum = np.nan
    if np.isfinite(rsi) and np.isfinite(macdh):
        # lower RSI better; negative MACD hist better
        r = float(np.clip((60.0 - rsi) / 30.0, 0.0, 1.0))
        m = float(np.clip((-macdh + 0.5) / 1.0, 0.0, 1.0))
        momentum = (0.6 * r + 0.4 * m) * 100.0
    elif np.isfinite(rsi):
        momentum = float(np.clip((60.0 - rsi) / 30.0, 0.0, 1.0) * 100.0)

    atrp = _f(common.get("ATR14_PCT"))
    vol_suit = np.nan
    if np.isfinite(atrp):
        vol_suit = 100.0 - abs(atrp - 5.0) * 16.0
        vol_suit = float(np.clip(vol_suit, 0.0, 100.0))

    rs = _f(common.get("REL_STRENGTH_20D_VS_QQQ"))
    rs_score = _score_0_100(-rs, lo=-10.0, hi=10.0)  # weaker than QQQ is better

    liq = _liquidity_score(_f(common.get("AVG_VOL_20D")), bid_ask_spread_pct)

    total = 0.0
    wsum = 0.0
    for k, v in {
        "trend": trend,
        "volume": volume,
        "momentum": momentum,
        "vol": vol_suit,
        "rs": rs_score,
        "liq": liq,
    }.items():
        if np.isfinite(_f(v)):
            total += w[k] * float(v)
            wsum += w[k]

    if wsum <= 0:
        return np.nan
    base = float(np.clip(total / wsum, 0.0, 100.0))

    # Feasibility penalty
    feas_pen = 0.0
    if shortable_flag is False:
        feas_pen += 25.0
    if np.isfinite(_f(borrow_fee_pct)) and borrow_fee_pct > cfg.borrow_fee_max:
        feas_pen += min(25.0, (borrow_fee_pct - cfg.borrow_fee_max))
    return float(np.clip(base - feas_pen, 0.0, 100.0))


def build_long_plan(common: Dict[str, object], df: pd.DataFrame, setup: Optional[str], cfg: RankConfig) -> Dict[str, str]:
    price = _f(common.get("PRICE"))
    d20 = _f(common.get("DMA20"))
    d50 = _f(common.get("DMA50"))
    atr = _f(common.get("ATR14"))

    highs = pd.to_numeric(df.get("high"), errors="coerce")
    lows = pd.to_numeric(df.get("low"), errors="coerce")
    prior_high = _f(highs.shift(1).rolling(20).max().iloc[-1]) if highs is not None and not highs.empty else np.nan
    swing_low = _f(lows.shift(1).rolling(10).min().iloc[-1]) if lows is not None and not lows.empty else np.nan

    entry = "N/A"
    inval = "N/A"
    t1 = "N/A"
    t2 = "N/A"

    if setup == "L1" and np.isfinite(prior_high):
        entry = f"Above pivot {prior_high:.2f} (or retest)"
        inv = min(x for x in [d50, swing_low] if np.isfinite(x)) if any(np.isfinite(x) for x in [d50, swing_low]) else np.nan
        if np.isfinite(inv):
            inval = f"Below {inv:.2f}"
        if np.isfinite(atr) and np.isfinite(prior_high):
            t1 = f"{prior_high + 2*atr:.2f}"
            t2 = f"{prior_high + 3*atr:.2f}"
    elif setup == "L2" and np.isfinite(d20) and np.isfinite(d50):
        lo = min(d20, d50)
        hi = max(d20, d50)
        entry = f"DMA band {lo:.2f}–{hi:.2f}"
        inv = min(x for x in [d50, swing_low] if np.isfinite(x)) if any(np.isfinite(x) for x in [d50, swing_low]) else np.nan
        if np.isfinite(inv):
            inval = f"Below {inv:.2f}"
        if np.isfinite(atr) and np.isfinite(price):
            t1 = f"{price + 2*atr:.2f}"
            t2 = f"{price + 3*atr:.2f}"
    elif setup == "L3" and np.isfinite(d50):
        entry = f"Reclaim DMA50 {d50:.2f}"
        inv = min(x for x in [d50, swing_low] if np.isfinite(x)) if any(np.isfinite(x) for x in [d50, swing_low]) else np.nan
        if np.isfinite(inv):
            inval = f"Below {inv:.2f}"
        if np.isfinite(atr) and np.isfinite(price):
            t1 = f"{price + 2*atr:.2f}"
            t2 = f"{price + 3*atr:.2f}"
    else:
        # fallback
        if np.isfinite(d20):
            entry = f"Near DMA20 {d20:.2f}"
        if np.isfinite(d50):
            inval = f"Below DMA50 {d50:.2f}"

    # ---------------------------
    # Numeric zones + RR (for Excel columns)
    # ---------------------------
    entry_low = np.nan
    entry_high = np.nan
    inval_px = np.nan
    t1_px = np.nan

    # Parse numeric invalidation when possible
    try:
        m = re.search(r"([-+]?[0-9]*\.?[0-9]+)", str(inval))
        inval_px = float(m.group(1)) if m else np.nan
    except Exception:
        inval_px = np.nan

    # Setup-specific zone bounds (fallback to "near" bands)
    if setup == "L2" and np.isfinite(d20) and np.isfinite(d50):
        entry_low, entry_high = float(min(d20, d50)), float(max(d20, d50))
    elif setup == "L1" and np.isfinite(prior_high):
        entry_low, entry_high = float(prior_high * 0.995), float(prior_high * 1.005)
    elif setup == "L3" and np.isfinite(d50):
        entry_low, entry_high = float(d50 * 0.997), float(d50 * 1.003)
    elif np.isfinite(d20):
        entry_low, entry_high = float(d20 * 0.995), float(d20 * 1.005)

    # Target-1 numeric
    try:
        t1_px = float(t1) if str(t1).strip() else np.nan
    except Exception:
        t1_px = np.nan

    entry_mid = np.nan
    if np.isfinite(entry_low) and np.isfinite(entry_high):
        entry_mid = 0.5 * (entry_low + entry_high)

    rr = np.nan
    if np.isfinite(entry_mid) and np.isfinite(inval_px) and np.isfinite(t1_px):
        risk = max(entry_mid - inval_px, 1e-9)
        reward = t1_px - entry_mid
        if reward > 0:
            rr = reward / risk

    verdict = "AVOID"
    return {
        "LONG_ENTRY_ZONE": entry,
        "LONG_ENTRY_ZONE_LOW": entry_low,
        "LONG_ENTRY_ZONE_HIGH": entry_high,
        "LONG_INVALIDATION": inval_px if np.isfinite(inval_px) else inval,
        "LONG_TARGET_1": t1_px if np.isfinite(t1_px) else t1,
        "LONG_TARGET_2": t2,
        "LONG_RR_RATIO": rr,
        "LONG_VERDICT": verdict,
    }


def build_short_plan(common: Dict[str, object], df: pd.DataFrame, setup: Optional[str], cfg: RankConfig) -> Dict[str, str]:
    price = _f(common.get("PRICE"))
    d20 = _f(common.get("DMA20"))
    d50 = _f(common.get("DMA50"))
    d200 = _f(common.get("DMA200"))
    atr = _f(common.get("ATR14"))

    lows = pd.to_numeric(df.get("low"), errors="coerce")
    prior_low = _f(lows.shift(1).rolling(20).min().iloc[-1]) if lows is not None and not lows.empty else np.nan
    recent_high = _f(pd.to_numeric(df.get("high"), errors="coerce").shift(1).rolling(10).max().iloc[-1])

    entry = "N/A"
    inval = "N/A"
    t1 = "N/A"
    t2 = "N/A"

    if setup in {"S1", "S2"}:
        if np.isfinite(d20) and np.isfinite(d50):
            entry = f"Weak bounce into {min(d20, d50):.2f}–{max(d20, d50):.2f}"
        else:
            entry = "On breakdown / weak bounce"
        inv = max(x for x in [d50, recent_high] if np.isfinite(x)) if any(np.isfinite(x) for x in [d50, recent_high]) else np.nan
        if np.isfinite(inv):
            inval = f"Above {inv:.2f}"
        # targets: prior low, then DMA200
        if np.isfinite(prior_low):
            t1 = f"{prior_low:.2f}"
        if np.isfinite(d200):
            t2 = f"{d200:.2f}"
        elif np.isfinite(atr) and np.isfinite(price):
            t2 = f"{price - 3*atr:.2f}"
    elif setup == "S3":
        entry = "Fade overextension (tight risk)"
        inv = recent_high if np.isfinite(recent_high) else np.nan
        if np.isfinite(inv):
            inval = f"Above {inv:.2f}"
        if np.isfinite(d20):
            t1 = f"{d20:.2f}"
        if np.isfinite(d50):
            t2 = f"{d50:.2f}"
    else:
        if np.isfinite(d20):
            entry = f"Weak bounce near DMA20 {d20:.2f}"
        if np.isfinite(d50):
            inval = f"Above DMA50 {d50:.2f}"

    # ---------------------------
    # Numeric zones + RR (for Excel columns)
    # ---------------------------
    entry_low = np.nan
    entry_high = np.nan
    inval_px = np.nan
    t1_px = np.nan

    try:
        m = re.search(r"([-+]?[0-9]*\.?[0-9]+)", str(inval))
        inval_px = float(m.group(1)) if m else np.nan
    except Exception:
        inval_px = np.nan

    if np.isfinite(d20) and np.isfinite(d50):
        entry_low, entry_high = float(min(d20, d50)), float(max(d20, d50))
    elif np.isfinite(d50):
        entry_low, entry_high = float(d50 * 0.997), float(d50 * 1.003)
    elif np.isfinite(d200):
        entry_low, entry_high = float(d200 * 0.997), float(d200 * 1.003)

    try:
        t1_px = float(t1) if str(t1).strip() else np.nan
    except Exception:
        t1_px = np.nan

    entry_mid = np.nan
    if np.isfinite(entry_low) and np.isfinite(entry_high):
        entry_mid = 0.5 * (entry_low + entry_high)

    rr = np.nan
    # For shorts: reward = entry_mid - target1, risk = invalidation - entry_mid
    if np.isfinite(entry_mid) and np.isfinite(inval_px) and np.isfinite(t1_px):
        risk = max(inval_px - entry_mid, 1e-9)
        reward = entry_mid - t1_px
        if reward > 0:
            rr = reward / risk

    verdict = "AVOID"
    return {
        "SHORT_ENTRY_ZONE": entry,
        "SHORT_ENTRY_ZONE_LOW": entry_low,
        "SHORT_ENTRY_ZONE_HIGH": entry_high,
        "SHORT_INVALIDATION": inval_px if np.isfinite(inval_px) else inval,
        "SHORT_TARGET_1": t1_px if np.isfinite(t1_px) else t1,
        "SHORT_TARGET_2": t2,
        "SHORT_RR_RATIO": rr,
        "SHORT_VERDICT": verdict,
    }


def explain_spike_drop(common: Dict[str, object], long_setup: Optional[str], short_setup: Optional[str]) -> Tuple[str, str]:
    surge = _f(common.get("VOL_SURGE_RATIO"))
    gap = _f(common.get("GAP_PCT"))
    rsi = _f(common.get("RSI14"))
    stack = str(common.get("DMA_STACK") or "MIXED")

    spike = "N/A"
    drop = "N/A"

    if long_setup == "L1":
        spike = "Breakout above resistance with heavy volume"
    elif long_setup == "L2":
        spike = "Institutional-style pullback into 20–50 DMA band"
    elif long_setup == "L3":
        spike = "Reversal reclaiming DMA50 with volume"
    else:
        if np.isfinite(surge) and surge >= 2.0:
            spike = "Volume surge (possible catalyst/news)"

    if short_setup == "S1":
        drop = "Breakdown below DMA50 with distribution volume"
    elif short_setup == "S2":
        drop = "Failed breakout / bull trap with selling pressure"
    elif short_setup == "S3":
        drop = "Overextended move; momentum exhaustion / mean reversion risk"
    else:
        if np.isfinite(gap) and gap <= -3.0:
            drop = "Gap down (news/earnings shock likely)"
        elif stack == "BEAR_STACKED" and np.isfinite(rsi) and rsi < 45:
            drop = "Weak trend + bearish momentum"

    return spike, drop


def verdicts(long_score: float, short_score: float, long_setup: Optional[str], short_setup: Optional[str],
             common: Dict[str, object], cfg: RankConfig) -> Tuple[str, str]:
    # Long
    pct_dma20 = _f(common.get("PCT_FROM_DMA20"))
    if np.isfinite(long_score) and long_setup and long_score >= cfg.long_score_buy_now:
        if np.isfinite(pct_dma20) and pct_dma20 > 8:
            long_v = "WAIT_FOR_DIP"
        else:
            long_v = "BUY_NOW"
    elif np.isfinite(long_score) and long_setup and long_score >= cfg.long_score_wait_dip:
        long_v = "WATCH"
    else:
        long_v = "AVOID"

    # Short
    if np.isfinite(short_score) and short_setup and short_score >= cfg.short_score_short_now:
        short_v = "SHORT_NOW"
    elif np.isfinite(short_score) and short_setup and short_score >= cfg.short_score_wait_bounce:
        short_v = "WATCH"
    else:
        short_v = "AVOID"

    return long_v, short_v


# ---------------------------
# Required API for buy.py
# ---------------------------

def load_rank_config(path="rank_config.json") -> RankConfig:
    """
    Load RankConfig from JSON.

    Supports BOTH formats:
      (A) Legacy flat keys (vol_surge_breakout, w_trend, ...)
      (B) Nested config used by your pipeline:
          {
            "version": "...",
            "scoring": {"weights": {...}},
            "thresholds": {"long": {...}, "short": {...}},
            "modes": {"breakout": {...}, "pullback": {...}},
            "active_mode_default": "pullback"
          }

    Unknown keys are ignored. If file missing/invalid -> RankConfig defaults.
    """
    import json
    import os

    # Resolve path
    try:
        p = str(path)
    except Exception:
        p = "rank_config.json"

    cfg = RankConfig()
    if not p or not os.path.exists(p):
        return cfg

    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
    except Exception:
        return cfg

    # 1) Backward-compatible: load any flat keys at top-level (older configs)
    try:
        for k, v in (data or {}).items():
            if hasattr(cfg, k):
                try:
                    setattr(cfg, k, type(getattr(cfg, k))(v))
                except Exception:
                    setattr(cfg, k, v)
    except Exception:
        pass

    # 2) New format: select thresholds (mode-aware)
    thresholds = None
    try:
        modes = data.get("modes")
        active_mode = data.get("active_mode_default") or data.get("active_mode") or None
        if isinstance(modes, dict) and active_mode and active_mode in modes:
            thresholds = modes[active_mode]
        else:
            thresholds = data.get("thresholds")
    except Exception:
        thresholds = None

    # Apply nested thresholds if present
    if isinstance(thresholds, dict):
        long_t = thresholds.get("long") or {}
        short_t = thresholds.get("short") or {}

        # Long
        if "vol_surge_breakout" in long_t: cfg.vol_surge_breakout = float(long_t["vol_surge_breakout"])
        if "vol_surge_reversal" in long_t: cfg.vol_surge_reversal = float(long_t["vol_surge_reversal"])
        if "vol_pullback_max" in long_t: cfg.vol_pullback_max = float(long_t["vol_pullback_max"])
        if "rsi_breakout_min" in long_t: cfg.rsi_breakout_min = float(long_t["rsi_breakout_min"])
        if "rsi_pullback_min" in long_t: cfg.rsi_pullback_min = float(long_t["rsi_pullback_min"])
        if "long_score_buy_now" in long_t: cfg.long_score_buy_now = float(long_t["long_score_buy_now"])
        if "long_score_wait_dip" in long_t: cfg.long_score_wait_dip = float(long_t["long_score_wait_dip"])

        # Short
        if "rsi_short_max" in short_t: cfg.rsi_short_max = float(short_t["rsi_short_max"])
        if "gap_down_pct" in short_t: cfg.gap_down_pct = float(short_t["gap_down_pct"])
        if "overext_dma20_pct" in short_t: cfg.overext_dma20_pct = float(short_t["overext_dma20_pct"])
        if "borrow_fee_max" in short_t: cfg.borrow_fee_max = float(short_t["borrow_fee_max"])
        if "short_score_short_now" in short_t: cfg.short_score_short_now = float(short_t["short_score_short_now"])
        if "short_score_wait_bounce" in short_t: cfg.short_score_wait_bounce = float(short_t["short_score_wait_bounce"])

    # 3) New format: weights under scoring.weights (if present)
    try:
        scoring = data.get("scoring") or {}
        weights = scoring.get("weights") or {}
        if isinstance(weights, dict):
            # Accept either pipeline names or cfg names
            if "trend" in weights: cfg.w_trend = float(weights["trend"])
            if "volume" in weights: cfg.w_volume = float(weights["volume"])
            if "momentum" in weights: cfg.w_momentum = float(weights["momentum"])
            if "volatility" in weights: cfg.w_volatility = float(weights["volatility"])
            if "relative_strength" in weights: cfg.w_rel_strength = float(weights["relative_strength"])
            if "liquidity" in weights: cfg.w_liquidity = float(weights["liquidity"])
    except Exception:
        pass

    # Normalize weights (safe even if partially missing)
    w = {
        "w_trend": cfg.w_trend,
        "w_volume": cfg.w_volume,
        "w_momentum": cfg.w_momentum,
        "w_volatility": cfg.w_volatility,
        "w_rel_strength": cfg.w_rel_strength,
        "w_liquidity": cfg.w_liquidity,
    }
    nw = _normalize_weights(w)
    cfg.w_trend = nw["w_trend"]
    cfg.w_volume = nw["w_volume"]
    cfg.w_momentum = nw["w_momentum"]
    cfg.w_volatility = nw["w_volatility"]
    cfg.w_rel_strength = nw["w_rel_strength"]
    cfg.w_liquidity = nw["w_liquidity"]
    return cfg

    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        for k, v in data.items():
            if hasattr(cfg, k):
                try:
                    setattr(cfg, k, type(getattr(cfg, k))(v))
                except Exception:
                    setattr(cfg, k, v)
    except Exception:
        return cfg

    # Normalize weights if provided oddly
    w = {
        "w_trend": cfg.w_trend,
        "w_volume": cfg.w_volume,
        "w_momentum": cfg.w_momentum,
        "w_volatility": cfg.w_volatility,
        "w_rel_strength": cfg.w_rel_strength,
        "w_liquidity": cfg.w_liquidity,
    }
    nw = _normalize_weights(w)
    cfg.w_trend = nw["w_trend"]
    cfg.w_volume = nw["w_volume"]
    cfg.w_momentum = nw["w_momentum"]
    cfg.w_volatility = nw["w_volatility"]
    cfg.w_rel_strength = nw["w_rel_strength"]
    cfg.w_liquidity = nw["w_liquidity"]
    return cfg


def feasibility_label(shortable_flag, borrow_fee_pct, cfg: RankConfig) -> str:
    # Short feasibility label used by buy.py / Excel.
    # Returns: OK / HARD_TO_BORROW / UNKNOWN
    sflag = shortable_flag
    try:
        if isinstance(sflag, str):
            sflag = sflag.strip().lower() in ("true", "t", "yes", "y", "1")
        elif sflag is None:
            sflag = None
        else:
            sflag = bool(sflag)
    except Exception:
        sflag = None

    if sflag is False:
        return "HARD_TO_BORROW"

    try:
        bf = float(borrow_fee_pct) if borrow_fee_pct is not None else None
    except Exception:
        bf = None

    if bf is None:
        return "UNKNOWN"

    if bf > float(cfg.borrow_fee_max):
        return "HARD_TO_BORROW"

    return "OK"
