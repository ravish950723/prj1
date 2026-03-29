from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from .compute import compute_indicators
from .config import sector_etfs, symbol_to_sector
from .entry_prices import compute_entry_prices
from .fetching import fetch_data_cached
from .macro_features import enrich_with_macro_features
from .train_dataset import build_training_frame, DEFAULT_FEATURES
from .ml_models import predict_ml_probability
from .entry_prices import candle_entries_multi


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_THRESHOLDS: Dict[str, Any] = {
    "model_thresholds": {"watch": 0.30, "buy": 0.45, "strong_buy": 0.60},
    "boosters": {
        "min_signal_score_for_boost": 0.08,
        "min_institutional_for_boost": 0.70,
        "min_entry_quality_for_buy": 0.45,
        "min_entry_quality_for_strong_buy": 0.60,
    },
    "guards": {
        "cap_if_macro_warn": True,
        "cap_if_distribution": True,
        "cap_if_markdown": True,
    },
    "training_reference": {},
}



def _execution_action(
        *,
        recommendation: str,
        avoid_new_entry: bool,
        current_price: float,
        refined_buy_price: float,
        entry_quality_score: float,
) -> str:
    rec = safe_str(recommendation, "WATCH").upper()

    if rec in {"AVOID", "WATCH"}:
        return rec

    if avoid_new_entry:
        return "AVOID"

    if rec not in {"BUY", "STRONG_BUY"}:
        return "WAIT"

    if not np.isfinite(current_price) or not np.isfinite(refined_buy_price) or refined_buy_price <= 0:
        return "WAIT"

    premium = (current_price / refined_buy_price) - 1.0

    if premium <= 0.03 and entry_quality_score >= 0.60:
        return "BUY_NOW"
    if premium <= 0.08 and entry_quality_score >= 0.48:
        return "BUY_ON_PULLBACK"
    if premium <= 0.12 and entry_quality_score >= 0.40:
        return "SCALE_IN"
    return "WAIT"

    if not np.isfinite(current_price) or not np.isfinite(refined_buy_price) or refined_buy_price <= 0:
        return "WAIT"

    premium = (current_price / refined_buy_price) - 1.0

    if premium <= 0.03 and entry_quality_score >= 0.55:
        return "BUY_NOW"

    if premium <= 0.08 and entry_quality_score >= 0.45:
        return "BUY_ON_PULLBACK"

    if premium <= 0.06 and entry_quality_score >= 0.45:
        return "BUY_ON_PULLBACK"

    return "WAIT"


def _risk_state(
        *,
        market_stage: str,
        macro_warn: bool,
        adx: float,
        institutional_score: float,
) -> str:
    stage_key = _normalize_stage(market_stage)

    if macro_warn or stage_key == "distribution":
        return "HIGH_RISK"

    if stage_key == "markdown":
        return "MEDIUM_RISK"
    if adx >= 25 and institutional_score >= 0.70 and stage_key in {"markup", "accumulation"}:
        return "LOW_RISK"
    return "MEDIUM_RISK"


def _compute_trade_levels(
        *,
        current_price: float,
        refined_buy_price: float,
        candle_entry_4w: float,
        candle_entry_8w: float,
        atr: float,
        market_stage: str,
        recommendation: str,
        avoid_new_entry: bool,
) -> Dict[str, float]:
    out = {
        "Stop Loss": np.nan,
        "Target 1": np.nan,
        "Target 2": np.nan,
        "Risk/Reward T1": np.nan,
        "Risk/Reward T2": np.nan,
    }

    if avoid_new_entry or recommendation not in {"BUY", "STRONG_BUY"}:
        return out

    entry = refined_buy_price if np.isfinite(refined_buy_price) and refined_buy_price > 0 else current_price
    if not np.isfinite(entry) or entry <= 0:
        return out

    stage_key = _normalize_stage(market_stage)

    support_floor = np.nanmean([
        candle_entry_4w if np.isfinite(candle_entry_4w) else np.nan,
        candle_entry_8w if np.isfinite(candle_entry_8w) else np.nan,
    ])

    if not np.isfinite(atr) or atr <= 0:
        atr = entry * 0.03

    if stage_key == "markup":
        stop_loss = min(entry - 1.20 * atr, support_floor if np.isfinite(support_floor) else entry - 1.20 * atr)
        target_1 = entry + 2.0 * atr
        target_2 = entry + 4.0 * atr
    elif stage_key == "accumulation":
        stop_loss = min(entry - 1.50 * atr, support_floor if np.isfinite(support_floor) else entry - 1.50 * atr)
        target_1 = entry + 2.5 * atr
        target_2 = entry + 5.0 * atr
    else:
        stop_loss = entry - 1.50 * atr
        target_1 = entry + 1.5 * atr
        target_2 = entry + 3.0 * atr

    risk = entry - stop_loss
    rr1 = ((target_1 - entry) / risk) if np.isfinite(risk) and risk > 0 else np.nan
    rr2 = ((target_2 - entry) / risk) if np.isfinite(risk) and risk > 0 else np.nan

    out["Stop Loss"] = round(stop_loss, 4) if np.isfinite(stop_loss) else np.nan
    out["Target 1"] = round(target_1, 4) if np.isfinite(target_1) else np.nan
    out["Target 2"] = round(target_2, 4) if np.isfinite(target_2) else np.nan
    out["Risk/Reward T1"] = round(rr1, 4) if np.isfinite(rr1) else np.nan
    out["Risk/Reward T2"] = round(rr2, 4) if np.isfinite(rr2) else np.nan
    return out


def _recommendation_rank(
        *,
        recommendation: str,
        confidence_grade: str,
        model_probability: float,
        entry_quality_score: float,
        signal_score: float,
        institutional_score: float,
        avoid_new_entry: bool,
) -> float:
    if avoid_new_entry:
        return 0.0

    base = 0.0
    if recommendation == "STRONG_BUY":
        base = 80.0
    elif recommendation == "BUY":
        base = 60.0
    elif recommendation == "WATCH":
        base = 35.0

    grade_bonus = {"A": 10.0, "B": 5.0, "C": 0.0}.get(confidence_grade, 0.0)

    score = (
            base
            + grade_bonus
            + 10.0 * np.clip(model_probability, 0.0, 1.0)
            + 8.0 * np.clip(entry_quality_score, 0.0, 1.0)
            + 6.0 * np.clip(signal_score, 0.0, 1.0)
            + 6.0 * np.clip(institutional_score, 0.0, 1.0)
    )
    return round(float(score), 4)


def _load_thresholds() -> Dict[str, Any]:
    path = os.path.join(_THIS_DIR, "strong_buy_thresholds.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        out = dict(DEFAULT_THRESHOLDS)
        out.update(data or {})
        out["model_thresholds"] = {
            **DEFAULT_THRESHOLDS["model_thresholds"],
            **out.get("model_thresholds", {}),
        }
        out["boosters"] = {
            **DEFAULT_THRESHOLDS["boosters"],
            **out.get("boosters", {}),
        }
        out["guards"] = {
            **DEFAULT_THRESHOLDS["guards"],
            **out.get("guards", {}),
        }
        return out
    except Exception:
        return dict(DEFAULT_THRESHOLDS)


THRESHOLDS = _load_thresholds()


def _load_model() -> Optional[Any]:
    candidates = [
        os.path.join(_THIS_DIR, "strong_buy_xgb_model_calibrated.pkl"),
        os.path.join(_THIS_DIR, "strong_buy_xgb_model.pkl"),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return joblib.load(p)
            except Exception:
                pass
    return None


MODEL = _load_model()


def _as_scalar(x: Any, default: Optional[float] = None) -> float:
    try:
        if default is None:
            default = np.nan
        if isinstance(x, pd.Series):
            if x.empty:
                return default
            x = x.iloc[-1]
        elif isinstance(x, np.ndarray):
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
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _as_bool(x: Any, default: bool = False) -> bool:
    try:
        if isinstance(x, pd.Series):
            if x.empty:
                return default
            x = x.iloc[-1]
        elif isinstance(x, np.ndarray):
            if x.size == 0:
                return default
            x = x.reshape(-1)[-1]
        elif isinstance(x, (list, tuple)):
            if len(x) == 0:
                return default
            x = x[-1]
        if x is None:
            return default
        if isinstance(x, (float, np.floating)) and np.isnan(x):
            return default
        return bool(x)
    except Exception:
        return default


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def safe_str(x: Any, default: str = "") -> str:
    try:
        if x is None:
            return default
        s = str(x).strip()
        return s if s else default
    except Exception:
        return default


def _trend_from_emas(close: float, ema20: float, ema50: float) -> str:
    if np.isfinite(close) and np.isfinite(ema20) and np.isfinite(ema50):
        if close >= ema20 >= ema50:
            return "Bullish"
        if close <= ema20 <= ema50:
            return "Bearish"
    return "Neutral"


def _normalize_stage(stage: str) -> str:
    s = safe_str(stage, "Neutral/Transition").strip().lower()
    if "mark-up" in s or "markup" in s:
        return "markup"
    if "accum" in s:
        return "accumulation"
    if "distribution" in s:
        return "distribution"
    if "mark-down" in s or "markdown" in s:
        return "markdown"
    return "neutral"


def _mtf_alignment_score(trend: str, trend_itf: str, trend_ltf: str) -> float:
    vals = [safe_str(trend).lower(), safe_str(trend_itf).lower(), safe_str(trend_ltf).lower()]
    bull = sum(v.startswith("bull") or v == "up" for v in vals)
    bear = sum(v.startswith("bear") or v == "down" for v in vals)

    if bull == 3:
        return 1.0
    if bull == 2:
        return 0.75
    if bear >= 2:
        return 0.20
    return 0.50


def _v4_composite_score(
        *,
        model_probability: float,
        entry_quality_score: float,
        signal_score: float,
        institutional_score: float,
        mtf_alignment_score: float,
        substage_confidence: float,
        regime_quality_score: float,
) -> float:
    score = (
            0.28 * np.clip(model_probability, 0.0, 1.0)
            + 0.18 * np.clip(entry_quality_score, 0.0, 1.0)
            + 0.14 * np.clip(signal_score, 0.0, 1.0)
            + 0.12 * np.clip(institutional_score, 0.0, 1.0)
            + 0.10 * np.clip(mtf_alignment_score, 0.0, 1.0)
            + 0.08 * np.clip(substage_confidence, 0.0, 1.0)
            + 0.10 * np.clip(regime_quality_score, 0.0, 1.0)
    )
    return float(np.clip(score, 0.0, 1.0))


def _calc_signal_score(df: pd.DataFrame) -> float:
    score = 0.0
    total = 0.0

    def add(cond: bool, w: float) -> None:
        nonlocal score, total
        total += w
        if cond:
            score += w

    if df is None or df.empty:
        return 0.0

    last = df.iloc[-1]
    vwap_sup = _as_scalar(last.get("vwap_support", np.nan), default=np.nan)
    close_v = _as_scalar(last.get("close", last.get("Close", np.nan)), default=np.nan)
    macd_cross = bool(last.get("MACD_Crossover", last.get("MACD_crossover", last.get("macd_cross", False))))
    rsi_val = safe_float(last.get("RSI_14", np.nan), np.nan)
    adx_val = safe_float(last.get("ADX_14", np.nan), np.nan)
    institutional = safe_float(last.get("institutional_score", 0.0), 0.0)
    volume_weight = safe_float(last.get("volume_weight", 0.0), 0.0)
    ema_up = _as_bool(last.get("EMA_uptrend", last.get("ema_uptrend", False)), False)
    near_support = _as_bool(last.get("near_support", False), False)
    darvas_signal = int(safe_float(last.get("darvas_signal", 0), 0.0))
    smc_breakout = _as_bool(last.get("SMC_Breakout", last.get("smc_breakout", False)), False)
    vol_surge = safe_float(last.get("VOL_SURGE_RATIO", last.get("volume_surge", 0.0)), 0.0)

    rsi_bull = np.isfinite(rsi_val) and rsi_val >= 55
    adx_good = np.isfinite(adx_val) and adx_val >= 20
    price_above_vwap = np.isfinite(vwap_sup) and np.isfinite(close_v) and close_v >= vwap_sup

    add(price_above_vwap, 0.14)
    add(ema_up, 0.14)
    add(macd_cross, 0.10)
    add(rsi_bull, 0.10)
    add(near_support, 0.10)
    add(vol_surge >= 1.5, 0.10)
    add(darvas_signal == 1, 0.14)
    add(smc_breakout, 0.08)
    add(adx_good, 0.05)
    add(institutional >= 0.70, 0.03)
    add(volume_weight >= 1.10, 0.02)

    if total <= 0:
        return 0.0
    return float(np.clip(score / total, 0.0, 1.0))


def _macro_flags(df: pd.DataFrame) -> Tuple[bool, bool]:
    macro_warn = False
    try:
        vix_regime = str(df.get("VIX_regime", pd.Series([""])).iloc[-1]).upper()
        if "HIGH" in vix_regime:
            macro_warn = True
        market_regime = str(df.get("Market_regime", pd.Series([""])).iloc[-1]).upper()
        if any(k in market_regime for k in ("RISK_OFF", "BEAR", "DOWNTREND", "HIGH_VOL")):
            macro_warn = True
    except Exception:
        macro_warn = False
    return (not macro_warn), macro_warn


def _entry_quality_score(
        *,
        current_price: float,
        refined_buy_price: float,
        market_stage: str,
        market_substage: str,
        substage_confidence: float,
        adx: float,
        institutional_score: float,
        signal_score: float,
) -> float:
    if not np.isfinite(current_price) or current_price <= 0:
        return 0.0
    if not np.isfinite(refined_buy_price) or refined_buy_price <= 0:
        return 0.0

    stage_key = _normalize_stage(market_stage)
    substage_u = safe_str(market_substage, "NEUTRAL_RANGE").upper()

    distance_pct = (current_price / refined_buy_price) - 1.0

    if distance_pct <= -0.03:
        price_fit = 0.85
    elif distance_pct <= 0.02:
        price_fit = 1.00
    elif distance_pct <= 0.05:
        price_fit = 0.80
    elif distance_pct <= 0.10:
        price_fit = 0.45
    else:
        price_fit = 0.15

    stage_bonus = 0.0
    if stage_key == "markup":
        stage_bonus = 0.15
    elif stage_key == "accumulation":
        stage_bonus = 0.18
    elif stage_key == "distribution":
        stage_bonus = -0.18
    elif stage_key == "markdown":
        stage_bonus = -0.25

    substage_bonus = 0.0
    if "BREAKOUT" in substage_u or "VWAP_RECLAIM" in substage_u:
        substage_bonus += 0.12
    if "PULLBACK" in substage_u or "SMART_MONEY_ENTRY" in substage_u:
        substage_bonus += 0.10
    if "BASE" in substage_u or "SPRING" in substage_u:
        substage_bonus += 0.10
    if "FAILED_BREAKOUT" in substage_u or "UTAD" in substage_u or "UPTHRUST" in substage_u:
        substage_bonus -= 0.18
    if "PANIC" in substage_u or "MOMENTUM_SELLOFF" in substage_u:
        substage_bonus -= 0.20

    adx_component = 0.0
    if np.isfinite(adx):
        if adx >= 30:
            adx_component = 0.10
        elif adx >= 20:
            adx_component = 0.06
        elif adx < 15:
            adx_component = -0.04

    inst_component = 0.12 * np.clip(institutional_score, 0.0, 1.0)
    signal_component = 0.18 * np.clip(signal_score, 0.0, 1.0)

    substage_conf_component = 0.10 * np.clip(substage_confidence, 0.0, 1.0)
    raw = price_fit + stage_bonus + substage_bonus + adx_component + inst_component + signal_component + substage_conf_component
    return float(np.clip(raw, 0.0, 1.0))






def final_recommendation(
    *,
    model_prob,
    signal_score,
    institutional_score,
    entry_quality_score,
    market_stage,
    market_substage,
    substage_confidence,
    mtf_alignment_score,
    regime_quality_score,
    macro_ok,
    macro_warn,
    thr,
):
    stage_key = _normalize_stage(market_stage)
    sub_u = safe_str(market_substage).upper()

    composite = _v4_composite_score(
        model_probability=model_prob,
        entry_quality_score=entry_quality_score,
        signal_score=signal_score,
        institutional_score=institutional_score,
        mtf_alignment_score=mtf_alignment_score,
        substage_confidence=substage_confidence,
        regime_quality_score=regime_quality_score,
    )

    market_mode = "NEUTRAL"
    if regime_quality_score >= 0.60 and mtf_alignment_score >= 0.60:
        market_mode = "BULL"
    elif regime_quality_score < 0.45:
        market_mode = "BEAR"

    if market_mode == "BULL" and mtf_alignment_score < 0.25:
        composite *= 0.85
    elif market_mode == "BEAR" and mtf_alignment_score < 0.40:
        composite *= 0.65

    if macro_warn:
        composite = min(composite, 0.45)

    if stage_key == "markdown":
        composite *= 0.55
    elif stage_key == "distribution":
        composite *= 0.70
    elif stage_key == "accumulation":
        composite *= 1.05
    elif stage_key == "markup":
        composite *= 1.10

    if stage_key in {"distribution", "markdown"}:
        reversal_ok = (
            entry_quality_score >= 0.80
            and substage_confidence >= 0.70
            and signal_score >= 0.50
            and institutional_score >= 0.60
        )
        composite = min(composite, 0.65 if reversal_ok else 0.35)

    if any(x in sub_u for x in ["FAILED_BREAKOUT", "UTAD", "UPTHRUST", "PANIC", "MOMENTUM_SELLOFF"]):
        composite = min(composite, 0.30)

    if substage_confidence < 0.40:
        composite *= 0.75

    if model_prob >= 0.50:
        composite += 0.05
    if entry_quality_score >= 0.75:
        composite += 0.05
    if substage_confidence >= 0.80:
        composite += 0.04
    if mtf_alignment_score >= 0.70:
        composite += 0.04
    if any(x in sub_u for x in ["SPRING", "TEST_OF_SPRING", "BREAKOUT_CONFIRMATION", "HIGH_VOLUME_BREAKOUT", "VWAP_RECLAIM"]):
        composite += 0.06

    composite = float(np.clip(composite, 0.0, 1.0))

    if stage_key in {"distribution", "markdown"} and entry_quality_score < 0.80:
        return "WATCH"
    if composite >= 0.75:
        return "STRONG_BUY"
    if composite >= 0.58:
        return "BUY"
    return "WATCH"




def compute_confidence_grade( *, recommendation: str, model_prob: float, signal_score: float,
        institutional_score: float, entry_quality_score: float, macro_warn: bool,adx: float,
        trend_htf: str, trend_itf: str, trend_ltf: str, market_stage: str,
) -> str:
    if macro_warn:
        return "C"

    stage_key = _normalize_stage(market_stage)
    trend_bullish = (trend_htf == "Bullish") and (trend_itf == "Bullish" or trend_ltf == "Bullish")

    if recommendation == "STRONG_BUY":
        if (
                model_prob >= 0.70
                and signal_score >= 0.10
                and institutional_score >= 0.80
                and entry_quality_score >= 0.65
                and adx >= 25
                and trend_bullish
                and stage_key in {"markup", "accumulation"}
        ):
            return "A"
        return "B"

    if recommendation == "BUY":
        if (
                model_prob >= 0.55
                and (signal_score >= 0.08 or institutional_score >= 0.75)
                and entry_quality_score >= 0.45
                and adx >= 20
                and trend_bullish
                and stage_key in {"markup", "accumulation", "neutral"}
        ):
            return "B"
        return "C"

    return "C"


def expected_holding_period(
        *,
        market_stage: str,
        market_substage: str,
        adx: float,
        recommendation: str,
) -> str:
    stage = _normalize_stage(market_stage)
    sub = safe_str(market_substage, "NEUTRAL_RANGE").upper()

    if recommendation not in {"BUY", "STRONG_BUY"}:
        return "N/A"

    if stage == "markup":
        if "BREAKOUT" in sub or "VWAP_RECLAIM" in sub:
            return "4-12 weeks"
        return "6-16 weeks"

    if stage == "accumulation":
        return "8-20 weeks"

    if stage in {"distribution", "markdown"}:
        return "Short-term only"

    if np.isfinite(adx) and adx >= 25:
        return "4-10 weeks"

    return "4-12 weeks"


def decision_reason(
        *,
        rec: str,
        model_prob: float,
        signal_score: float,
        institutional_score: float,
        entry_quality_score: float,
        market_stage: str,
        market_substage: str,
        macro_warn: bool,
) -> str:
    parts = [f"model={model_prob:.2f}", f"signal={signal_score:.2f}", f"entry={entry_quality_score:.2f}"]

    if np.isfinite(institutional_score):
        parts.append(f"institutional={institutional_score:.2f}")

    parts.append(f"stage={safe_str(market_stage)}")
    parts.append(f"substage={safe_str(market_substage)}")

    if macro_warn:
        parts.append("macro_warn=YES")

    return f"{rec}: " + ", ".join(parts)


def _compute_sector_corr(symbol: str, df: pd.DataFrame) -> float:
    sector_corr = 0.0
    etf_symbol = sector_etfs.get(symbol_to_sector.get(symbol))
    if not etf_symbol or etf_symbol == symbol:
        return 0.0

    try:
        df_etf = fetch_data_cached(etf_symbol, "10 Y", "1 day", require_today=False)
        df_etf = compute_indicators(df_etf, symbol=etf_symbol)

        if "date" not in df.columns or "date" not in df_etf.columns:
            return 0.0

        s = df[["date", "close"]].rename(columns={"close": "close_sym"})
        e = df_etf[["date", "close"]].rename(columns={"close": "close_etf"})
        merged = s.merge(e, on="date", how="inner")

        if len(merged) < 21:
            return 0.0

        rolling_corr = merged["close_sym"].pct_change().rolling(20).corr(merged["close_etf"].pct_change())
        val = rolling_corr.iloc[-1]
        if pd.notna(val) and np.isfinite(val):
            sector_corr = float(val)
    except Exception:
        sector_corr = 0.0

    return sector_corr


def confidence_band(
        *,
        model_probability: float,
        entry_quality_score: float,
        signal_score: float,
        institutional_score: float,
) -> str:
    """
    Convert combined model + execution quality into a simple confidence grade.
    """
    p = float(np.clip(model_probability, 0.0, 1.0)) if np.isfinite(model_probability) else 0.0
    eq = float(np.clip(entry_quality_score, 0.0, 1.0)) if np.isfinite(entry_quality_score) else 0.0
    ss = float(np.clip(signal_score, 0.0, 1.0)) if np.isfinite(signal_score) else 0.0
    inst = float(np.clip(institutional_score, 0.0, 1.0)) if np.isfinite(institutional_score) else 0.0

    composite = 0.50 * p + 0.20 * eq + 0.15 * ss + 0.15 * inst

    if composite >= 0.80:
        return "A"
    if composite >= 0.65:
        return "B"
    if composite >= 0.50:
        return "C"
    return "D"


def analyze_symbol(symbol: str, df_raw: Optional[pd.DataFrame] = None) -> Optional[Dict[str, Any]]:
    try:
        df = df_raw.copy() if (df_raw is not None and not df_raw.empty) else fetch_data_cached(symbol, "10 Y", "1 day")
        df = compute_indicators(df, symbol=symbol)

        if "VIX Vol Regime" not in df.columns and "vix_vol_regime" not in df.columns:
            df = enrich_with_macro_features(df)

        if df is None or df.empty:
            raise ValueError("No data returned")

        last = df.iloc[-1]

        close = safe_float(last.get("close", np.nan), np.nan)
        ema21 = safe_float(last.get("EMA_21", np.nan), np.nan)
        ema20 = safe_float(last.get("EMA_20", ema21), np.nan)
        ema50 = safe_float(last.get("EMA_50", np.nan), np.nan)
        vwap_support = safe_float(last.get("vwap_support", np.nan), np.nan)
        adx = safe_float(last.get("ADX_14", np.nan), np.nan)
        volume_weight = safe_float(last.get("volume_weight", 0.0))
        institutional_score = safe_float(last.get("institutional_score", 0.0))
        confidence_score = safe_float(last.get("confidence_score", 0.0))

        trend = _trend_from_emas(close, ema20, ema50)
        trend_itf = safe_str(last.get("ITF_Trend", last.get("itf_trend", trend)), trend)
        trend_ltf = safe_str(last.get("LTF_Trend", last.get("ltf_trend", trend)), trend)

        sector_corr = _compute_sector_corr(symbol, df)
        df["sector_corr"] = sector_corr

        signal_score = _calc_signal_score(df)
        macro_ok, macro_warn = _macro_flags(df)
        mtf_score = _mtf_alignment_score(trend, trend_itf, trend_ltf)

        # Compute model probability FIRST
        model_proba = np.nan
        if MODEL is not None:
            latest_full = df.copy()

            # 🔥 Use SAME transformation as training
            latest_transformed = build_training_frame(latest_full)
            latest = latest_transformed.iloc[-1:].copy()
            expected_features = DEFAULT_FEATURES
            features = latest.reindex(columns=expected_features).fillna(0.0)

            # expected_features = [str(c).strip() for c in getattr(MODEL, "feature_names_in_", [])]

            if expected_features:
                for col in expected_features:
                    if col not in latest.columns:
                        latest[col] = 0.0

                features = latest.reindex(columns=expected_features)

                # ===============================
                # 🔥 DEBUG ML FEATURE PIPELINE
                # ===============================
                print(f"[{symbol}] MODEL loaded: {MODEL is not None}")
                print(f"[{symbol}] expected feature count: {len(expected_features)}")
                print(f"[{symbol}] latest column count: {len(latest.columns)}")

                missing_cols = [col for col in expected_features if col not in latest.columns]
                print(f"[{symbol}] missing feature count: {len(missing_cols)}")
                print(f"[{symbol}] first 20 missing features: {missing_cols[:20]}")

                # ===============================
                # 🔥 FEATURE COVERAGE SAFETY CHECK (ADD HERE)
                # ===============================
                missing_ratio = len(missing_cols) / len(expected_features) if expected_features else 0.0

                if missing_ratio > 0.30:
                    print(f"[{symbol}] ❌ TOO MANY MISSING FEATURES → ML DISABLED")
                    model_proba = np.nan

                for col in features.columns:
                    features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0.0)

                # Only run ML if features are OK
                if not np.isnan(model_proba):
                    try:
                        latest_row_dict = features.iloc[0].to_dict()  # ✅ FIX (use features, not raw latest)
                        model_proba = predict_ml_probability(MODEL, latest_row_dict)
                    except Exception as e:
                        print(f"[{symbol}] predict_proba failed: {e}")
                        model_proba = np.nan

        # provisional fallback before exact entry score is known
        fallback_model_proba = (
                0.45 * signal_score +
                0.55 * institutional_score
        )
        if not np.isfinite(model_proba):
            model_proba = fallback_model_proba

        model_proba = float(np.clip(model_proba, 0.0, 1.0))

        # ML-aligned entry prices
        entry_prices = compute_entry_prices(df, model_probability=model_proba, symbol=symbol)
        buy_price = safe_float(entry_prices.get("Refined Buy Price"), close)

        entries = candle_entries_multi(df)

        entry_prices["Candle Entry 2w"] = entries.get(2)
        entry_prices["Candle Entry 4w"] = entries.get(4)
        entry_prices["Candle Entry 6w"] = entries.get(6)
        entry_prices["Candle Entry 8w"] = entries.get(8)
        entry_prices["Candle Entry 12w"] = entries.get(12)
        entry_prices["Candle Entry 18w"] = entries.get(18)
        entry_prices["Candle Entry 30w"] = entries.get(30)

        # 🔒 HARD LOCK FLAG
        entry_prices["_entry_locked"] = True

        # ================================
        # 🔥 PREVENT FLAT LADDER (CRITICAL FIX)
        # ================================
        vals = [entry_prices.get(w) for w in [
            "Candle Entry 2w", "Candle Entry 4w", "Candle Entry 6w",
            "Candle Entry 8w", "Candle Entry 12w", "Candle Entry 18w", "Candle Entry 30w"
        ] if np.isfinite(entry_prices.get(w, np.nan))]

        if len(set([round(v, 2) for v in vals])) <= 1:
            for i, k in enumerate([
                "Candle Entry 2w", "Candle Entry 4w", "Candle Entry 6w",
                "Candle Entry 8w", "Candle Entry 12w", "Candle Entry 18w", "Candle Entry 30w"
            ]):
                if np.isfinite(entry_prices.get(k, np.nan)):
                    entry_prices[k] *= (1 - 0.002 * i)

        market_stage = safe_str(last.get("market_stage"), "Neutral/Transition")
        market_substage = safe_str(last.get("market_substage"), "NEUTRAL_RANGE")
        substage_confidence = safe_float(last.get("substage_confidence", 0.0), 0.0)
        regime_quality_score = safe_float(last.get("regime_quality_score", 0.0), 0.0)
        darvas_signal = int(safe_float(last.get("darvas_signal"), 0.0))
        price_vs_buy_pct = ((close / buy_price) - 1.0) if np.isfinite(close) and np.isfinite(
            buy_price) and buy_price > 0 else np.nan

        entry_quality_score = _entry_quality_score(
            current_price=close,
            refined_buy_price=buy_price,
            market_stage=market_stage,
            market_substage=market_substage,
            substage_confidence=substage_confidence,
            adx=adx,
            institutional_score=institutional_score,
            signal_score=signal_score,
        )

        recommendation = final_recommendation(
            model_prob=model_proba,
            signal_score=signal_score,
            institutional_score=institutional_score,
            entry_quality_score=entry_quality_score,
            market_stage=market_stage,
            market_substage=market_substage,
            substage_confidence=substage_confidence,
            mtf_alignment_score=mtf_score,
            regime_quality_score=regime_quality_score,
            macro_ok=macro_ok,
            macro_warn=macro_warn,
            thr=THRESHOLDS,
        )

        stage_key = _normalize_stage(market_stage)


        if recommendation in {"BUY", "STRONG_BUY"}:
            if stage_key not in {"markup", "accumulation"}:
                recommendation = "WATCH"

        old_rec = recommendation

        if recommendation == "STRONG_BUY" and substage_confidence < 0.60:
            recommendation = "BUY"
        if entry_quality_score < 0.55 and recommendation in {"BUY", "STRONG_BUY"}:
            recommendation = "WATCH"
        if entry_quality_score < 0.65 and recommendation == "STRONG_BUY":
            recommendation = "BUY"

        if recommendation != old_rec:
            print(f"[{symbol}] Entry filter adjusted:", f"{old_rec} → {recommendation}", f"| entry_quality={entry_quality_score:.2f}")
            print(f"[{symbol}] decision inputs | "f"prob={model_proba:.2f} "f"entry={entry_quality_score:.2f} "
                f"sub={substage_confidence:.2f} "f"mtf={mtf_score:.2f} "f"regime={regime_quality_score:.2f} "f"stage={market_stage} "
                f"substage={market_substage} "f"-> rec={recommendation}")

        stage_key = _normalize_stage(market_stage)

        avoid_new_entry = (
                macro_warn
                or (stage_key == "distribution")
                or (stage_key == "markdown" and model_proba < 0.55)
                or entry_quality_score < 0.45
                or substage_confidence < 0.40
                or mtf_score < 0.50
                or regime_quality_score < 0.45
                or model_proba < 0.50
        )

        if avoid_new_entry:
            print(f"[{symbol}] AVOID triggered:",
                  f"entry={entry_quality_score:.2f}",
                  f"substage={substage_confidence:.2f}",
                  f"mtf={mtf_score:.2f}",
                  f"regime={regime_quality_score:.2f}",
                  f"prob={model_proba:.2f}")

        confidence_grade = confidence_band(
            model_probability=model_proba,
            entry_quality_score=entry_quality_score,
            signal_score=signal_score,
            institutional_score=institutional_score,
        )

        holding_period = expected_holding_period(
            market_stage=market_stage,
            market_substage=market_substage,
            adx=adx,
            recommendation=recommendation,
        )

        execution_action = _execution_action(
            recommendation=recommendation,
            avoid_new_entry=avoid_new_entry,
            current_price=close,
            refined_buy_price=buy_price,
            entry_quality_score=entry_quality_score,
        )

        risk_state = _risk_state(
            market_stage=market_stage,
            macro_warn=macro_warn,
            adx=adx,
            institutional_score=institutional_score,
        )

        trade_levels = _compute_trade_levels(
            current_price=close,
            refined_buy_price=buy_price,
            candle_entry_4w=safe_float(entry_prices.get("Candle Entry 4w"), np.nan),
            candle_entry_8w=safe_float(entry_prices.get("Candle Entry 8w"), np.nan),
            atr=safe_float(last.get("ATR_14"), np.nan),
            market_stage=market_stage,
            recommendation=recommendation,
            avoid_new_entry=avoid_new_entry,
        )

        recommendation_rank = _recommendation_rank(
            recommendation=recommendation,
            confidence_grade=confidence_grade,
            model_probability=model_proba,
            entry_quality_score=entry_quality_score,
            signal_score=signal_score,
            institutional_score=institutional_score,
            avoid_new_entry=avoid_new_entry,
        )

        result: Dict[str, Any] = {
            "Symbol": symbol,
            "Current Price": round(close, 4) if np.isfinite(close) else np.nan,
            "VWAP Support": round(vwap_support, 4) if np.isfinite(vwap_support) else np.nan,
            "ADX": round(adx, 4) if np.isfinite(adx) else np.nan,
            "Institutional Score": round(institutional_score, 6),
            "Volume Weight": round(volume_weight, 6),
            "Confidence Score": round(safe_float(confidence_score, 0.0), 6),
            "Trend": trend,
            "ITF Trend": trend_itf,
            "LTF Trend": trend_ltf,
            "Rule Recommendation": safe_str(last.get("rule_recommendation"), "HOLD"),
            "Darvas Breakout %": round(safe_float(last.get("darvas_breakout_pct"), 0.0), 4),
            "Darvas Signal": "✅" if darvas_signal == 1 else "❌",
            "Market Stage": market_stage,
            "Market Sub-Stage": market_substage,
            "Substage Confidence": round(substage_confidence, 6),
            "Final_Action": execution_action or "WAIT",
            "MTF Alignment Score": round(mtf_score, 6),
            "Regime Quality Score": round(regime_quality_score, 6),
            "BUYPIPE V4 Composite": round(
                _v4_composite_score(
                    model_probability=model_proba,
                    entry_quality_score=entry_quality_score,
                    signal_score=signal_score,
                    institutional_score=institutional_score,
                    mtf_alignment_score=mtf_score,
                    substage_confidence=substage_confidence,
                    regime_quality_score=regime_quality_score,
                ),
                6,
            ),
            "Sector Corr": round(sector_corr, 6),
            "Signal Score": round(signal_score, 6),
            "Model Probability": round(model_proba, 6),
            "Entry Quality Score": round(entry_quality_score, 6),
            "Price vs Refined Buy %": round(price_vs_buy_pct, 4) if np.isfinite(price_vs_buy_pct) else np.nan,
            "Avoid New Entry": "YES" if avoid_new_entry else "NO",
            "Recommendation": recommendation,
            "Decision Reason": decision_reason(
                rec=recommendation,
                model_prob=model_proba,
                signal_score=signal_score,
                institutional_score=institutional_score,
                entry_quality_score=entry_quality_score,
                market_stage=market_stage,
                market_substage=market_substage,
                macro_warn=macro_warn,
            ),
            "Confidence Grade": confidence_grade,
            "Expected Holding Period": holding_period,
            "Primary Entry Price": (
                round(buy_price, 4)
                if (np.isfinite(buy_price) and not avoid_new_entry and recommendation in {"BUY", "STRONG_BUY"})
                else np.nan
            ),
            **entry_prices,
            "Execution Action": execution_action,
            "Risk State": risk_state,
            "Recommendation Rank": recommendation_rank,
            **trade_levels,
        }

        print(f"[{symbol}] ROW KEYS:", result.keys())
        print(f"[{symbol}] Market Stage:", result.get("Market Stage"))
        print(f"[{symbol}] Signal Score:", result.get("Signal Score"))

        return result

    except Exception as e:
        print(f"⚠️ Error analyzing {symbol}: {e}")
        return None





def compute_refined_buy_price(row: dict) -> float:
    price = float(row.get("Current Price") or np.nan)
    vwap = float(row.get("VWAP") or np.nan)
    atr = float(row.get("ATR14") or 0.0)

    darvas_top = float(row.get("Darvas Top") or np.nan)
    darvas_bottom = float(row.get("Darvas Bottom") or np.nan)
    smc_break = float(row.get("SMC Breakout Level") or np.nan)
    smc_invalid = float(row.get("SMC Invalidation Level") or np.nan)
    z = float(row.get("VWAP Deviation Z") or np.nan)

    market_mode = str(row.get("Market Mode") or "RANGE").upper()

    candidates = []

    # Mean reversion candidate
    if np.isfinite(vwap):
        if market_mode == "BEAR":
            candidates.append(vwap - 0.75 * atr)
        elif pd.notna(z) and z < -0.8:
            candidates.append(vwap)
        else:
            candidates.append(vwap - 0.25 * atr)

    # Darvas candidate
    if np.isfinite(darvas_top) and np.isfinite(darvas_bottom):
        if market_mode == "BULL":
            # buy near breakout retest, not at random deep discount
            candidates.append(darvas_top - 0.20 * atr)
        else:
            candidates.append(darvas_bottom + 0.10 * atr)

    # SMC candidate
    if np.isfinite(smc_break):
        candidates.append(smc_break - 0.15 * atr)
    if np.isfinite(smc_invalid):
        candidates.append(smc_invalid + 0.20 * atr)

    candidates = [c for c in candidates if np.isfinite(c) and c > 0]

    if not candidates:
        return float(price) if np.isfinite(price) else np.nan

    # Regime-aware selection
    if market_mode == "BULL":
        refined = max(candidates)   # do not set unrealistically low in bull trend
    elif market_mode == "BEAR":
        refined = min(candidates)   # demand deeper entry in bear regime
    else:
        refined = float(np.median(candidates))

    # Safety rails
    if np.isfinite(price):
        refined = min(refined, price * 1.02)
        refined = max(refined, price * 0.80)

    return float(refined)