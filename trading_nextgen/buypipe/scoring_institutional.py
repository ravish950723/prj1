from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from .deep_learning import predict_dl_probability
from .ml_models import load_ml_model, predict_ml_probability
from .utils import safe_float, logistic


# ============================================================
# RL threshold policy
# ============================================================

def _load_rl_policy(project_root: Path):
    p = project_root / "rl_threshold_policy.pkl"
    if p.exists():
        try:
            return joblib.load(p)
        except Exception:
            return None
    return None


def _clip01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0


def _safe_num(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        out = float(v)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _rl_thresholds(
        project_root: Path,
        row: Dict[str, Any],
        ml_prob: float,
        default_buy: float = 0.45,
        default_strong: float = 0.65,
) -> Tuple[float, float]:
    policy = _load_rl_policy(project_root)
    if policy is None or not hasattr(policy, "choose_thresholds"):
        return default_buy, default_strong

    state = {}
    for k, v in row.items():
        try:
            state[str(k)] = safe_float(v)
        except Exception:
            continue
    state["MODEL_PROBA"] = safe_float(ml_prob)

    try:
        buy_thr, strong_thr = policy.choose_thresholds(state)
        buy_thr = float(np.clip(buy_thr, 0.45, 0.75))
        strong_thr = float(np.clip(max(strong_thr, buy_thr + 0.08), 0.65, 0.90))
        return buy_thr, strong_thr
    except Exception:
        return default_buy, default_strong


# ============================================================
# General helpers
# ============================================================

def _confidence_band(p: float) -> str:
    if p >= 0.85:
        return "VERY_HIGH"
    if p >= 0.70:
        return "HIGH"
    if p >= 0.55:
        return "MEDIUM"
    if p >= 0.35:
        return "LOW"
    return "VERY_LOW"


def safe_str(x: Any, default: str = "") -> str:
    try:
        if x is None:
            return default
        s = str(x).strip()
        return s if s else default
    except Exception:
        return default


def _as_scalar(x: Any, default: float = np.nan) -> float:
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


def _normalize_stage(stage: str) -> str:
    s = safe_str(stage, "neutral").lower()
    s = s.replace("_", "").replace("-", "").replace("/", "").replace(" ", "")
    if "markup" in s or "uptrend" in s:
        return "markup"
    if "accum" in s:
        return "accumulation"
    if "distribution" in s:
        return "distribution"
    if "markdown" in s or "downtrend" in s:
        return "markdown"
    if "pullback" in s:
        return "pullback"
    return "neutral"


def _mtf_alignment_score(trend: str, trend_itf: str, trend_ltf: str) -> float:
    vals = [safe_str(trend).lower(), safe_str(trend_itf).lower(), safe_str(trend_ltf).lower()]

    def bucket(v: str) -> str:
        if any(k in v for k in ("bull", "up", "markup", "recovery")):
            return "bull"
        if any(k in v for k in ("bear", "down", "markdown")):
            return "bear"
        if any(k in v for k in ("sideways", "neutral", "base", "range")):
            return "neutral"
        return "neutral"

    buckets = [bucket(v) for v in vals]
    bull = buckets.count("bull")
    bear = buckets.count("bear")
    neutral = buckets.count("neutral")

    if bull == 3:
        return 1.00
    if bull == 2 and neutral == 1:
        return 0.82
    if bull == 2:
        return 0.75
    if bear == 3:
        return 0.12
    if bear == 2 and neutral == 1:
        return 0.26
    if bear >= 2:
        return 0.20
    if neutral >= 2:
        return 0.58
    return 0.50


# ============================================================
# Substage intelligence
# ============================================================

def _substage_profile(substage: str) -> Dict[str, Any]:
    s = safe_str(substage).upper()

    profile = {
        "phase": "neutral",
        "entry_bias": 0.0,
        "regime_bias": 0.0,
        "decision_bias": 0.0,
        "min_conf": 0.0,
        "reversal_candidate": False,
        "continuation_candidate": False,
        "avoid_new_long": False,
    }

    bullish_reversal = {
        "BASE_FORMATION",
        "LOW_VOLATILITY_COMPRESSION",
        "SECONDARY_TEST (ST_ACC)",
        "SPRING (SHAKEOUT)",
        "TEST_OF_SPRING",
        "ACCUMULATION_BREAKOUT",
        "EARLY_MARKUP_TRANSITION",
        "DEAD_CAT_BOUNCE",  # cautious reversal candidate if seen upstream
    }
    bullish_continuation = {
        "BREAKOUT_CONFIRMATION",
        "HIGH_VOLUME_BREAKOUT (CTA_BREAKOUT)",
        "PULLBACK_TO_VALUE",
        "SMART_MONEY_ENTRY",
        "TREND_CONTINUATION",
        "RE-ACCUMULATION",
    }
    bearish_continuation = {
        "CONTINUATION_DOWNTREND",
        "MOMENTUM_SELLOFF",
        "FAILED_BREAKOUT",
        "UPTHRUST",
        "UTAD",
        "LOWER_HIGH_FORMATION",
        "MARKDOWN_CONTINUATION",
        "DISTRIBUTION_BREAKDOWN",
    }
    bearish_exhaustion = {
        "SELLING_CLIMAX",
        "PANIC_LOW",
        "CAPITULATION",
        "OVERSOLD_BOUNCE_SETUP",
        "BEAR_TRAP",
    }

    if s in bullish_reversal:
        profile.update(
            {
                "phase": "bullish_reversal",
                "entry_bias": 0.12,
                "regime_bias": 0.10,
                "decision_bias": 0.08,
                "min_conf": 0.30,
                "reversal_candidate": True,
            }
        )
    elif s in bullish_continuation:
        profile.update(
            {
                "phase": "bullish_continuation",
                "entry_bias": 0.10,
                "regime_bias": 0.12,
                "decision_bias": 0.10,
                "min_conf": 0.35,
                "continuation_candidate": True,
            }
        )
    elif s in bearish_continuation:
        profile.update(
            {
                "phase": "bearish_continuation",
                "entry_bias": -0.16,
                "regime_bias": -0.18,
                "decision_bias": -0.16,
                "min_conf": 0.35,
                "avoid_new_long": True,
                "continuation_candidate": True,
            }
        )
    elif s in bearish_exhaustion:
        profile.update(
            {
                "phase": "bearish_exhaustion",
                "entry_bias": 0.04,
                "regime_bias": -0.04,
                "decision_bias": 0.02,
                "min_conf": 0.40,
                "reversal_candidate": True,
            }
        )
    else:
        if "BASE" in s or "SPRING" in s or "COMPRESSION" in s or "SECONDARY_TEST" in s:
            profile.update(
                {
                    "phase": "bullish_reversal",
                    "entry_bias": 0.10,
                    "regime_bias": 0.08,
                    "decision_bias": 0.06,
                    "min_conf": 0.30,
                    "reversal_candidate": True,
                }
            )
        elif "BREAKOUT" in s or "SMART_MONEY" in s:
            profile.update(
                {
                    "phase": "bullish_continuation",
                    "entry_bias": 0.08,
                    "regime_bias": 0.10,
                    "decision_bias": 0.08,
                    "min_conf": 0.35,
                    "continuation_candidate": True,
                }
            )
        elif "MOMENTUM_SELLOFF" in s or "CONTINUATION_DOWNTREND" in s or "FAILED_BREAKOUT" in s or "UPTHRUST" in s:
            profile.update(
                {
                    "phase": "bearish_continuation",
                    "entry_bias": -0.14,
                    "regime_bias": -0.16,
                    "decision_bias": -0.14,
                    "min_conf": 0.35,
                    "avoid_new_long": True,
                    "continuation_candidate": True,
                }
            )
        elif "OVERSOLD" in s or "CAPITULATION" in s or "CLIMAX" in s:
            profile.update(
                {
                    "phase": "bearish_exhaustion",
                    "entry_bias": 0.03,
                    "regime_bias": -0.03,
                    "decision_bias": 0.02,
                    "min_conf": 0.40,
                    "reversal_candidate": True,
                }
            )

    return profile


# ============================================================
# Signal quality
# ============================================================

def _calc_signal_score(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return 0.0

    last = df.iloc[-1]
    score = 0.0
    total = 0.0

    def add(cond: bool, weight: float) -> None:
        nonlocal score, total
        total += weight
        if cond:
            score += weight

    close_v = _as_scalar(last.get("close", last.get("Close", np.nan)))
    vwap_sup = _as_scalar(last.get("vwap_support", last.get("VWAP Support", np.nan)))
    macd_cross = _as_bool(last.get("MACD_Crossover", last.get("MACD_crossover", last.get("macd_cross", False))))
    rsi_val = _as_scalar(last.get("RSI_14", last.get("RSI", np.nan)))
    adx_val = _as_scalar(last.get("ADX_14", last.get("ADX", np.nan)))
    inst_score = _as_scalar(last.get("institutional_score", last.get("Institutional Score", 0.0)), 0.0)
    volume_weight = _as_scalar(last.get("volume_weight", last.get("Volume Weight", 0.0)), 0.0)
    ema_up = _as_bool(last.get("EMA_uptrend", last.get("EMA Uptrend", False)))
    near_support = _as_bool(last.get("near_support", last.get("Near Support", False)))
    darvas_signal = int(_as_scalar(last.get("darvas_signal", 0), 0.0))
    smc_breakout = _as_bool(last.get("SMC_Breakout", last.get("smc_breakout", False)))
    vol_surge = _as_scalar(last.get("VOL_SURGE_RATIO", last.get("volume_surge", last.get("Volume Surge", 0.0))), 0.0)

    add(np.isfinite(close_v) and np.isfinite(vwap_sup) and close_v >= vwap_sup, 0.14)
    add(ema_up, 0.14)
    add(macd_cross, 0.10)
    add(np.isfinite(rsi_val) and rsi_val >= 55, 0.10)
    add(near_support, 0.10)
    add(vol_surge >= 1.20, 0.08)
    add(darvas_signal == 1, 0.10)
    add(smc_breakout, 0.08)
    add(np.isfinite(adx_val) and adx_val >= 20, 0.08)
    add(inst_score >= 0.60, 0.05)
    add(volume_weight >= 1.05, 0.03)

    if total <= 0:
        return 0.0
    return float(np.clip(score / total, 0.0, 1.0))


def _macro_flags(df: pd.DataFrame) -> Tuple[bool, bool]:
    macro_warn = False
    try:
        vix_regime = str(df.get("VIX_regime", pd.Series([""])).iloc[-1]).upper()
        market_regime = str(df.get("Market_regime", pd.Series([""])).iloc[-1]).upper()
        if "HIGH" in vix_regime:
            macro_warn = True
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
    sub_profile = _substage_profile(market_substage)
    sub_u = safe_str(market_substage).upper()

    distance_pct = (current_price / refined_buy_price) - 1.0

    if distance_pct <= -0.03:
        price_fit = 0.85
    elif distance_pct <= 0.02:
        price_fit = 1.00
    elif distance_pct <= 0.05:
        price_fit = 0.82
    elif distance_pct <= 0.10:
        price_fit = 0.48
    else:
        price_fit = 0.18

    stage_bonus = 0.0
    if stage_key == "markup":
        stage_bonus = 0.12
    elif stage_key == "accumulation":
        stage_bonus = 0.16
    elif stage_key == "pullback":
        stage_bonus = 0.06
    elif stage_key == "distribution":
        stage_bonus = -0.10
    elif stage_key == "markdown":
        stage_bonus = -0.12

    if stage_key == "accumulation" and sub_profile["phase"] == "bullish_reversal":
        stage_bonus += 0.06
    if stage_key == "markdown" and sub_profile["phase"] == "bearish_continuation":
        stage_bonus -= 0.06
    if stage_key == "markdown" and sub_profile["phase"] == "bearish_exhaustion":
        stage_bonus += 0.05

    substage_bonus = float(sub_profile["entry_bias"])

    if "BREAKOUT" in sub_u:
        substage_bonus += 0.04
    if "PULLBACK" in sub_u or "SMART_MONEY_ENTRY" in sub_u:
        substage_bonus += 0.04

    adx_component = 0.0
    if np.isfinite(adx):
        if adx >= 30:
            adx_component = 0.08
        elif adx >= 20:
            adx_component = 0.05
        elif adx < 12:
            adx_component = -0.03

    inst_component = 0.12 * np.clip(institutional_score, 0.0, 1.0)
    signal_component = 0.20 * np.clip(signal_score, 0.0, 1.0)

    conf_weight = 0.08 if substage_confidence >= sub_profile["min_conf"] else 0.03
    conf_component = conf_weight * np.clip(substage_confidence, 0.0, 1.0)

    raw = price_fit + stage_bonus + substage_bonus + adx_component + inst_component + signal_component + conf_component
    return float(np.clip(raw, 0.0, 1.0))


def _regime_quality_score(
        *,
        market_stage: str,
        market_substage: str,
        substage_confidence: float,
        mtf_alignment_score: float,
        macro_ok: bool,
) -> float:
    stage_key = _normalize_stage(market_stage)
    sub_profile = _substage_profile(market_substage)

    score = 0.50

    if stage_key == "markup":
        score += 0.18
    elif stage_key == "accumulation":
        score += 0.14
    elif stage_key == "pullback":
        score += 0.05
    elif stage_key == "distribution":
        score -= 0.12
    elif stage_key == "markdown":
        score -= 0.16

    score += float(sub_profile["regime_bias"])

    if stage_key == "accumulation" and sub_profile["phase"] == "bullish_reversal":
        score += 0.06
    if stage_key == "markup" and sub_profile["phase"] == "bullish_continuation":
        score += 0.05
    if stage_key == "markdown" and sub_profile["phase"] == "bearish_continuation":
        score -= 0.06
    if stage_key == "markdown" and sub_profile["phase"] == "bearish_exhaustion":
        score += 0.04

    conf_weight = 0.10 if substage_confidence >= sub_profile["min_conf"] else 0.05
    score += conf_weight * np.clip(substage_confidence, 0.0, 1.0)
    score += 0.12 * np.clip(mtf_alignment_score, 0.0, 1.0)

    if not macro_ok:
        score -= 0.08

    return float(np.clip(score, 0.0, 1.0))


def _v4_composite_score(
    *,
    model_probability: float,
    entry_quality_score: float,
    signal_score: float,
    institutional_score: float,
    order_flow_score: float,              # ✅ ADD
    institutional_flow_score: float,      # ✅ ADD
    absorption_score: float,              # ✅ ADD
    mtf_alignment_score: float,
    substage_confidence: float,
    regime_quality_score: float,
    substage_decision_bias: float = 0.0,
) -> float:
    flow_component = (
            0.03 * np.clip(order_flow_score, 0.0, 1.0) +
            0.03 * np.clip(institutional_flow_score, 0.0, 1.0) +
            0.02 * np.clip(absorption_score, 0.0, 1.0)
    )

    score = (
            0.38 * model_probability +
            0.24 * entry_quality_score +
            0.10 * signal_score +
            0.18 * institutional_score +
            0.05 * mtf_alignment_score +
            0.04 * substage_confidence +
            0.05 * regime_quality_score +
            flow_component
    )
    score += 0.50 * float(substage_decision_bias)
    return float(np.clip(score, 0.0, 1.0))


# ============================================================
# Decision logic
# ============================================================

def final_recommendation(
        *,
        model_prob: float,
        signal_score: float,
        institutional_score: float,
        order_flow_score: float,  # ✅ ADD
        institutional_flow_score: float,  # ✅ ADD
        absorption_score: float,  # ✅ ADD
        entry_quality_score: float,
        market_stage: str,
        market_substage: str,
        substage_confidence: float,
        mtf_alignment_score: float,
        regime_quality_score: float,
        macro_ok: bool,
        macro_warn: bool,
        thr: Dict[str, float],
) -> Tuple[str, float, str]:
    stage_key = _normalize_stage(market_stage)
    sub_u = safe_str(market_substage).upper()
    sub_profile = _substage_profile(market_substage)



    composite = _v4_composite_score(
        model_probability=model_prob,
        entry_quality_score=entry_quality_score,
        signal_score=signal_score,
        institutional_score=institutional_score,
        order_flow_score=order_flow_score,  # ✅ ADD
        institutional_flow_score=institutional_flow_score,  # ✅ ADD
        absorption_score=absorption_score,  # ✅ ADD
        mtf_alignment_score=mtf_alignment_score,
        substage_confidence=substage_confidence,
        regime_quality_score=regime_quality_score,
        substage_decision_bias=float(sub_profile["decision_bias"]),
    )

    if institutional_score > 0.60:
        composite *= 1.08
    elif institutional_score < 0.25:
        composite *= 0.88

    if macro_warn:
        composite = min(composite, 0.78)

    if stage_key == "markdown":
        composite *= 0.94
    elif stage_key == "distribution":
        composite *= 0.93
    elif stage_key == "accumulation":
        composite *= 1.07
    elif stage_key == "markup":
        composite *= 1.10
    elif stage_key == "pullback":
        composite *= 1.05

    if stage_key == "accumulation" and sub_profile["phase"] == "bullish_reversal":
        composite *= 1.08
    if stage_key == "markup" and sub_profile["phase"] == "bullish_continuation":
        composite *= 1.06
    if stage_key == "markdown" and sub_profile["phase"] == "bearish_continuation":
        composite *= 0.88
    if stage_key == "distribution" and sub_profile["phase"] == "bearish_continuation":
        composite *= 0.92
    if stage_key == "markdown" and sub_profile["phase"] == "bearish_exhaustion":
        composite *= 1.05

    if any(x in sub_u for x in
           ["FAILED_BREAKOUT", "UTAD", "UPTHRUST", "PANIC", "MOMENTUM_SELLOFF", "CONTINUATION_DOWNTREND"]):
        if model_prob < 0.68:
            composite *= 0.88
        else:
            composite *= 0.96

    reversal_ok = (
            sub_profile["reversal_candidate"]
            and stage_key in {"markdown", "distribution", "accumulation"}
            and entry_quality_score >= 0.58
            and signal_score >= 0.18
            and substage_confidence >= max(0.25, sub_profile["min_conf"] - 0.05)
    )
    if reversal_ok:
        composite = max(composite, min(0.72, model_prob * 0.96))

    oversold_reversal = (
            stage_key in {"markdown", "distribution"}
            and sub_profile["reversal_candidate"]
            and entry_quality_score >= 0.52
            and signal_score >= 0.16
    )
    if oversold_reversal:
        composite *= 1.06

    if substage_confidence < 0.20:
        composite *= 0.92

    composite = float(np.clip(composite, 0.0, 1.0))

    # ==========================================
    # OVERSOLD / SMART-MONEY REVERSAL OVERRIDES
    # ==========================================

    flow_strength = (
            0.5 * order_flow_score +
            0.3 * institutional_flow_score +
            0.2 * absorption_score
    )

    # smart-money accumulation boost
    if flow_strength > 0.55 and stage_key in {"markdown", "accumulation"}:
        composite *= 1.12

    # oversold reversal boost
    oversold_reversal = (
            stage_key in {"markdown", "distribution"}
            and entry_quality_score >= 0.45
            and signal_score >= 0.35
    )

    if oversold_reversal:
        composite = max(composite, model_prob * 0.85)

    print(
        f"[DEBUG] {safe_str(market_stage)} | {safe_str(market_substage)} | "
        f"Model={model_prob:.3f} | Signal={signal_score:.3f} | Entry={entry_quality_score:.3f} | "
        f"Inst={institutional_score:.3f} | OF={order_flow_score:.3f} | IF={institutional_flow_score:.3f} | Abs={absorption_score:.3f} | "
        f"FlowStr={flow_strength:.3f} | Composite={composite:.3f} | Oversold={oversold_reversal}"
    )

    composite = float(np.clip(composite, 0.0, 1.0))

    buy_thr = float(thr["buy"])
    strong_thr = float(thr["strong"])

    if sub_profile["phase"] == "bullish_reversal" and substage_confidence >= sub_profile["min_conf"]:
        buy_thr -= 0.03
        strong_thr -= 0.02
    elif sub_profile["phase"] == "bullish_continuation" and substage_confidence >= sub_profile["min_conf"]:
        buy_thr -= 0.02
    elif sub_profile["phase"] == "bearish_continuation" and substage_confidence >= sub_profile["min_conf"]:
        buy_thr += 0.03
        strong_thr += 0.04

    buy_thr = float(np.clip(buy_thr, 0.40, 0.80))
    strong_thr = float(np.clip(max(strong_thr, buy_thr + 0.08), 0.55, 0.90))



    if composite >= strong_thr and entry_quality_score >= 0.58 and signal_score >= 0.18:
        rec = "STRONG_BUY"
        reason = "high_composite"
    elif composite >= buy_thr and entry_quality_score >= 0.42:
        rec = "BUY"
        reason = "qualified_buy"
    elif composite >= 0.40 or reversal_ok or oversold_reversal:
        rec = "WATCH"
        reason = "watch_setup"
    else:
        rec = "AVOID"
        reason = "weak_composite"

    if sub_profile["avoid_new_long"] and substage_confidence >= sub_profile["min_conf"]:
        if rec == "STRONG_BUY":
            rec = "BUY" if composite >= strong_thr + 0.06 and model_prob >= 0.72 else "WATCH"
            reason = "bearish_substage_guard"
        elif rec == "BUY" and composite < buy_thr + 0.06:
            rec = "WATCH"
            reason = "bearish_substage_guard"

    if model_prob < 0.25 and rec in {"BUY", "STRONG_BUY"}:
        rec = "WATCH"
        reason = "weak_model"

    return rec, composite, reason


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


def _execution_intelligence(
        *,
        recommendation: str,
        current_price: float,
        refined_buy_price: float,
        entry_quality_score: float,
        l2_status: str,
        l2_quality: str,
        spread_pct: float,
        imbalance: float,
        microprice: float,
        best_bid: float,
        best_ask: float,
) -> Dict[str, Any]:
    rec = safe_str(recommendation, "WATCH").upper()

    if rec not in {"BUY", "STRONG_BUY"}:
        return {
            "Execution Style": "NO_ENTRY",
            "Execution Order Type": "NONE",
            "Execution Aggression": "NONE",
            "Execution Note": "no_long_entry",
        }

    if safe_str(l2_status).upper() != "OK":
        return {
            "Execution Style": "PRICE_BASED_ONLY",
            "Execution Order Type": "LIMIT",
            "Execution Aggression": "MEDIUM",
            "Execution Note": "l2_unavailable_fallback",
        }

    spread_ok = np.isfinite(spread_pct) and spread_pct <= 0.08
    spread_wide = np.isfinite(spread_pct) and spread_pct > 0.20
    bid_pressure = np.isfinite(imbalance) and imbalance > 0.15
    ask_pressure = np.isfinite(imbalance) and imbalance < -0.15
    quality = safe_str(l2_quality).upper()

    if spread_wide or ask_pressure:
        style = "PASSIVE"
        order_type = "LIMIT"
        aggression = "LOW"
        note = "wide_spread_or_ask_pressure"
    elif spread_ok and bid_pressure and quality in {"GOOD", "DEEP"}:
        style = "AGGRESSIVE"
        order_type = "LIMIT_AT_ASK" if np.isfinite(best_ask) else "LIMIT"
        aggression = "HIGH"
        note = "tight_spread_bid_support"
    else:
        style = "BALANCED"
        order_type = "LIMIT"
        aggression = "MEDIUM"
        note = "normal_l2_conditions"

    chase_price = np.nan
    if np.isfinite(best_bid) and np.isfinite(best_ask):
        if style == "PASSIVE":
            chase_price = best_bid
        elif style == "AGGRESSIVE":
            chase_price = best_ask
        else:
            chase_price = (best_bid + best_ask) / 2.0

    return {
        "Execution Style": style,
        "Execution Order Type": order_type,
        "Execution Aggression": aggression,
        "Execution Note": note,
        "Execution Chase Price": chase_price,
        "Execution Microprice": microprice if np.isfinite(microprice) else np.nan,
    }


def _size_from_l2(
        *,
        recommendation: str,
        base_class: str,
        l2_quality: str,
        imbalance: float,
        spread_pct: float,
) -> str:
    if recommendation not in {"BUY", "STRONG_BUY"}:
        return "NONE"

    out = base_class
    q = safe_str(l2_quality).upper()

    if q in {"GOOD", "DEEP"} and np.isfinite(imbalance) and imbalance > 0.15 and np.isfinite(
            spread_pct) and spread_pct < 0.10:
        if base_class == "MEDIUM":
            out = "LARGE"
        elif base_class == "SMALL":
            out = "MEDIUM"

    if np.isfinite(spread_pct) and spread_pct > 0.20:
        if out == "LARGE":
            out = "MEDIUM"
        elif out == "MEDIUM":
            out = "SMALL"

    return out


def _risk_state(
        *,
        market_stage: str,
        macro_warn: bool,
        adx: float,
        institutional_score: float,
        market_substage: str = "",
        substage_confidence: float = 0.0,
) -> str:
    stage_key = _normalize_stage(market_stage)
    sub_profile = _substage_profile(market_substage)

    if macro_warn:
        return "HIGH_RISK"
    if stage_key == "markdown":
        return "HIGH_RISK"
    if stage_key == "distribution":
        return "MEDIUM_RISK"
    if sub_profile["phase"] == "bearish_continuation" and substage_confidence >= sub_profile["min_conf"]:
        return "HIGH_RISK"
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


def _derive_l2_fallback(row: Dict[str, Any], current_price: float) -> Dict[str, Any]:
    status = safe_str(row.get("L2 Status"), "")
    quality = safe_str(row.get("L2 Quality"), "")
    best_bid = safe_float(row.get("L2 Best Bid"), np.nan)
    best_ask = safe_float(row.get("L2 Best Ask"), np.nan)
    spread_pct = safe_float(row.get("BID_ASK_SPREAD_PCT"), np.nan)
    proxy_imbalance = safe_float(row.get("Order Flow Imbalance", row.get("L2 Imbalance")), np.nan)
    proxy_microprice = safe_float(row.get("L2 Microprice"), np.nan)

    if not status:
        status = "UNAVAILABLE"
    if not quality:
        quality = "THIN"

    if np.isfinite(best_bid) and np.isfinite(best_ask) and best_ask > 0 and best_ask >= best_bid:
        mid = (best_bid + best_ask) / 2.0
        if not np.isfinite(spread_pct):
            spread_pct = ((best_ask - best_bid) / mid) * 100.0 if mid > 0 else np.nan
        status = "OK" if safe_str(status).upper() not in {"PROXY", "NO_DEPTH"} else "PROXY"
        if np.isfinite(spread_pct):
            if spread_pct <= 0.05:
                quality = "DEEP"
            elif spread_pct <= 0.15:
                quality = "GOOD"
            else:
                quality = "THIN"
    elif np.isfinite(current_price) and current_price > 0:
        if not np.isfinite(spread_pct):
            spread_pct = safe_float(row.get("ATR14_PCT"), np.nan) * 100.0 * 0.20
        if not np.isfinite(best_bid) and np.isfinite(spread_pct):
            best_bid = current_price * (1.0 - max(spread_pct, 0.01) / 200.0)
        if not np.isfinite(best_ask) and np.isfinite(spread_pct):
            best_ask = current_price * (1.0 + max(spread_pct, 0.01) / 200.0)
        if not np.isfinite(proxy_microprice):
            proxy_microprice = current_price
        if not np.isfinite(proxy_imbalance):
            proxy_imbalance = 0.0
        status = "PROXY"

    return {
        "L2 Status": status,
        "L2 Quality": quality,
        "BID_ASK_SPREAD_PCT": spread_pct,
        "L2 Best Bid": best_bid,
        "L2 Best Ask": best_ask,
        "L2 Imbalance": proxy_imbalance,
        "L2 Microprice": proxy_microprice,
    }


def _recommendation_rank(
        *,
        recommendation: str,
        confidence_grade: str,
        model_probability: float,
        entry_quality_score: float,
        signal_score: float,
        institutional_score: float,
        order_flow_score: float,
        institutional_flow_score: float,
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

    grade_bonus = {"A": 10.0, "B": 5.0, "C": 0.0, "D": -5.0}.get(confidence_grade, 0.0)

    score = (
            base
            + grade_bonus
            + 10.0 * np.clip(model_probability, 0.0, 1.0)
            + 8.0 * np.clip(entry_quality_score, 0.0, 1.0)
            + 6.0 * np.clip(signal_score, 0.0, 1.0)
            + 6.0 * np.clip(institutional_score, 0.0, 1.0)
            + 6.0 * np.clip(order_flow_score, 0.0, 1.0)
            + 4.0 * np.clip(institutional_flow_score, 0.0, 1.0)
    )
    return round(float(score), 4)


# ============================================================
# Main scoring entry point
# ============================================================

def apply_scoring(
        project_root: Path,
        row: Dict[str, Any],
        df: pd.DataFrame,
        use_ml: bool = True,
        use_dl: bool = True,
        use_rl: bool = True,
) -> Dict[str, Any]:
    model = load_ml_model(project_root) if use_ml else None
    ml_prob, ml_status = predict_ml_probability(model, row) if use_ml else (np.nan, "disabled")
    row["ml_status"] = ml_status

    dl_prob = predict_dl_probability(project_root, df, row, use_dl=use_dl)
    quant_prob = safe_float(row.get("QUANT_COMPOSITE_SCORE", row.get("quant_composite_score")), 0.50)

    signal_score = safe_float(row.get("signal_score", row.get("Signal Score")), np.nan)
    if not np.isfinite(signal_score):
        signal_score = _calc_signal_score(df)

    inst_score = safe_float(row.get("institutional_score", row.get("Institutional Score")), 0.50)
    order_flow_score = safe_float(row.get("Order Flow Score", row.get("order_flow_score")), np.nan)
    if not np.isfinite(order_flow_score):
        order_flow_score = safe_float(row.get("L2 Imbalance"), 0.0)
        order_flow_score = float(np.clip((order_flow_score + 1.0) / 2.0, 0.0, 1.0))
    inst_flow_score = safe_float(row.get("Institutional Flow Score", row.get("institutional_flow_score")), np.nan)
    if not np.isfinite(inst_flow_score):
        inst_flow_score = inst_score
    inst_score = float(np.clip(0.65 * inst_score + 0.20 * inst_flow_score + 0.15 * order_flow_score, 0.0, 1.0))

    ml_component = 0.0 if not np.isfinite(ml_prob) else float(ml_prob)
    dl_component = 0.0 if not np.isfinite(dl_prob) else float(dl_prob)
    quant_component = np.clip(float(quant_prob), 0.0, 1.0)

    ensemble_raw = logistic(
        2.8 * ml_component
        + 1.0 * dl_component
        + 1.5 * quant_component
        + 0.7 * signal_score
        + 0.5 * inst_score
        + 0.45 * order_flow_score
        + 0.35 * inst_flow_score
        - 1.95
    )

    eps_q = safe_float(row.get("eps_quality_score"), 0.0)
    etf_q = safe_float(row.get("ETF_PROXY_GROWTH_SCORE"), 0.0)

    if safe_str(row.get("ASSET_TYPE"), "STOCK").upper() == "STOCK":
        fundamental_boost = 0.06 * np.clip(eps_q, 0.0, 1.0)
    else:
        fundamental_boost = 0.04 * np.clip(etf_q, 0.0, 1.0)

    ensemble_raw = float(np.clip(ensemble_raw + fundamental_boost, 0.0, 1.0))
    row["FUNDAMENTAL_BOOST"] = round(fundamental_boost, 6)

    order_flow_score = np.clip(_safe_num(row.get("Order Flow Score", 0.0)), 0.0, 1.0)
    institutional_flow_score = np.clip(_safe_num(row.get("Institutional Flow Score", 0.0)), 0.0, 1.0)
    absorption_score = np.clip(_safe_num(row.get("Absorption Score", 0.0)), 0.0, 1.0)
    signal_score = float(np.clip(0.78 * signal_score + 0.14 * order_flow_score + 0.08 * inst_flow_score, 0.0, 1.0))

    stage = row.get("market_stage", row.get("Market Stage", ""))
    substage = row.get("market_substage", row.get("Market Sub-Stage", ""))
    substage_conf = safe_float(row.get("substage_confidence", row.get("Substage Confidence")), 0.0)
    sub_profile = _substage_profile(substage)

    if safe_str(stage) == "":
        stage = "neutral"
    if safe_str(substage) == "":
        substage = "NEUTRAL_RANGE"

    trend_htf = safe_str(row.get("HTF_Trend", row.get("Trend", "")))
    trend_itf = safe_str(row.get("ITF_Trend", ""))
    trend_ltf = safe_str(row.get("LTF_Trend", ""))

    mtf_score = _mtf_alignment_score(trend_htf, trend_itf, trend_ltf)

    macro_ok, macro_warn = _macro_flags(df)

    current_price = safe_float(row.get("Current Price"), np.nan)
    refined_buy_price = safe_float(row.get("Refined Buy Price"), np.nan)
    adx = safe_float(row.get("ADX", row.get("ADX_14")), np.nan)

    entry_quality = _entry_quality_score(
        current_price=current_price,
        refined_buy_price=refined_buy_price,
        market_stage=safe_str(stage),
        market_substage=safe_str(substage),
        substage_confidence=substage_conf,
        adx=adx,
        institutional_score=inst_score,
        signal_score=signal_score,
    )

    regime_quality = _regime_quality_score(
        market_stage=safe_str(stage),
        market_substage=safe_str(substage),
        substage_confidence=substage_conf,
        mtf_alignment_score=mtf_score,
        macro_ok=macro_ok,
    )

    buy_thr, strong_thr = _rl_thresholds(project_root, row, ml_component) if use_rl else (0.45, 0.65)
    buy_thr = max(0.45, buy_thr)
    strong_thr = max(0.65, strong_thr)




    rec, final_score, decision_reason = final_recommendation(
        model_prob=ensemble_raw,
        signal_score=signal_score,
        institutional_score=inst_score,
        order_flow_score=order_flow_score,
        institutional_flow_score=institutional_flow_score,
        absorption_score=absorption_score,
        entry_quality_score=entry_quality,
        market_stage=safe_str(stage),
        market_substage=safe_str(substage),
        substage_confidence=substage_conf,
        mtf_alignment_score=mtf_score,
        regime_quality_score=regime_quality,
        macro_ok=macro_ok,
        macro_warn=macro_warn,
        thr={"buy": buy_thr, "strong": strong_thr},
    )

    upstream_rec = safe_str(row.get("Recommendation"), "").upper()
    if upstream_rec == "AVOID" and ensemble_raw < 0.55 and final_score < 0.52:
        rec = upstream_rec
        decision_reason = "upstream_avoid"
    elif upstream_rec == "WATCH" and ensemble_raw < 0.60 and final_score < 0.62:
        rec = upstream_rec
        decision_reason = "upstream_watch"

    avoid_new_entry = rec not in {"BUY", "STRONG_BUY"}
    final_action = _execution_action(
        recommendation=rec,
        avoid_new_entry=avoid_new_entry,
        current_price=current_price,
        refined_buy_price=refined_buy_price,
        entry_quality_score=entry_quality,
    )

    l2_fallback = _derive_l2_fallback(row, current_price)
    row.update({k: v for k, v in l2_fallback.items() if v is not None})

    l2_status = safe_str(row.get("L2 Status"), "")
    l2_quality = safe_str(row.get("L2 Quality"), "")
    spread_pct = safe_float(row.get("BID_ASK_SPREAD_PCT"), np.nan)
    imbalance = safe_float(row.get("L2 Imbalance"), np.nan)
    microprice = safe_float(row.get("L2 Microprice"), np.nan)
    best_bid = safe_float(row.get("L2 Best Bid"), np.nan)
    best_ask = safe_float(row.get("L2 Best Ask"), np.nan)

    exec_intel = _execution_intelligence(
        recommendation=rec,
        current_price=current_price,
        refined_buy_price=refined_buy_price,
        entry_quality_score=entry_quality,
        l2_status=l2_status,
        l2_quality=l2_quality,
        spread_pct=spread_pct,
        imbalance=imbalance,
        microprice=microprice,
        best_bid=best_bid,
        best_ask=best_ask,
    )

    signal_type = rec

    confidence_score = round(
        0.38 * final_score
        + 0.24 * ml_component
        + 0.08 * dl_component
        + 0.14 * quant_component
        + 0.12 * entry_quality
        + 0.06 * order_flow_score
        + 0.04 * inst_flow_score,
        6,
    )

    if confidence_score >= 0.84:
        confidence_grade = "A"
    elif confidence_score >= 0.70:
        confidence_grade = "B"
    elif confidence_score >= 0.54:
        confidence_grade = "C"
    else:
        confidence_grade = "D"

    row["Best_Risk_Reward"] = max(
        safe_float(row.get("LONG_RR_RATIO"), 0.0),
        safe_float(row.get("SHORT_RR_RATIO"), 0.0),
    )

    row["Buy_Window_Status"] = "ACTIVE" if rec in {"BUY", "STRONG_BUY"} else "WAIT"
    row["Position_Size_Class"] = (
        "LARGE" if rec == "STRONG_BUY"
        else "MEDIUM" if rec == "BUY"
        else "SMALL" if rec == "WATCH"
        else "NONE"
    )

    row["Position_Size_Class"] = _size_from_l2(
        recommendation=rec,
        base_class=row["Position_Size_Class"],
        l2_quality=l2_quality,
        imbalance=imbalance,
        spread_pct=spread_pct,
    )

    row["Primary_Entry_Price"] = (
            row.get("LONG_ENTRY_ZONE_LOW")
            or row.get("Refined Buy Price")
            or row.get("Current Price")
    )

    if row.get("LONG_ENTRY_ZONE_LOW"):
        row["Primary_Entry_Source"] = "LONG_ENTRY_ZONE_LOW"
    elif row.get("Refined Buy Price"):
        row["Primary_Entry_Source"] = "Refined Buy Price"
    else:
        row["Primary_Entry_Source"] = "Current Price"

    row["Add_On_Dip_Price"] = (
            row.get("LONG_ENTRY_ZONE_HIGH")
            or row.get("Candle Entry 2w")
            or row.get("Refined Buy Price")
            or row.get("Current Price")
    )

    atr = safe_float(row.get("ATR14", row.get("ATR_14")), np.nan)
    trade_levels = _compute_trade_levels(
        current_price=current_price,
        refined_buy_price=refined_buy_price,
        candle_entry_4w=safe_float(row.get("Candle Entry 4w"), np.nan),
        candle_entry_8w=safe_float(row.get("Candle Entry 8w"), np.nan),
        atr=atr,
        market_stage=safe_str(stage),
        recommendation=rec,
        avoid_new_entry=avoid_new_entry,
    )

    risk_state = _risk_state(
        market_stage=safe_str(stage),
        macro_warn=macro_warn,
        adx=adx,
        institutional_score=inst_score,
        market_substage=safe_str(substage),
        substage_confidence=substage_conf,
    )

    recommendation_rank = _recommendation_rank(
        recommendation=rec,
        confidence_grade=confidence_grade,
        model_probability=final_score,
        entry_quality_score=entry_quality,
        signal_score=signal_score,
        institutional_score=inst_score,
        order_flow_score=order_flow_score,
        institutional_flow_score=inst_flow_score,
        avoid_new_entry=avoid_new_entry,
    )

    row.update(
        {
            "ML Probability": round(ml_component, 6),
            "DL Probability": round(dl_component, 6),
            "QUANT_PROBABILITY_PROXY": round(quant_component, 6),
            "Model_Prob_Raw": round(ensemble_raw, 6),
            "Model Probability": round(final_score, 6),
            "RL_BUY_THRESHOLD": round(buy_thr, 6),
            "RL_STRONG_THRESHOLD": round(strong_thr, 6),
            "Confidence Band": _confidence_band(final_score),
            "Model-Driven Buy": "Y" if final_score >= buy_thr else "N",
            "Model-Driven Strong Buy": "Y" if final_score >= strong_thr else "N",
            "Signal_Type": signal_type,
            "Recommendation": rec,
            "Momentum Recommendation": rec,
            "Final_Action": final_action,
            "Decision Reason": decision_reason,
            "Confidence Score": confidence_score,
            "CONFIDENCE_SCORE": confidence_score,
            "Confidence Grade": confidence_grade,
            "signal_score": round(signal_score, 6),
            "Signal Score": round(signal_score, 6),
            "institutional_score": round(inst_score, 6),
            "Institutional Score": round(inst_score, 6),
            "order_flow_score": round(order_flow_score, 6),
            "Order Flow Score": round(order_flow_score, 6),
            "institutional_flow_score": round(inst_flow_score, 6),
            "Institutional Flow Score": round(inst_flow_score, 6),
            "entry_quality_score": round(entry_quality, 6),
            "Entry Quality Score": round(entry_quality, 6),
            "regime_quality_score": round(regime_quality, 6),
            "Regime Quality Score": round(regime_quality, 6),
            "mtf_alignment_score": round(mtf_score, 6),
            "MTF Alignment Score": round(mtf_score, 6),
            "Risk State": risk_state,
            "Recommendation Rank": recommendation_rank,
            "Macro OK": macro_ok,
            "Macro Warn": macro_warn,
            "Substage Phase": sub_profile["phase"],
            "Substage Entry Bias": round(float(sub_profile["entry_bias"]), 6),
            "Substage Regime Bias": round(float(sub_profile["regime_bias"]), 6),
            "Substage Decision Bias": round(float(sub_profile["decision_bias"]), 6),
            "Substage Min Confidence": round(float(sub_profile["min_conf"]), 6),
            "Substage Reversal Candidate": bool(sub_profile["reversal_candidate"]),
            "Substage Continuation Candidate": bool(sub_profile["continuation_candidate"]),
            "Substage Avoid New Long": bool(sub_profile["avoid_new_long"]),

            # Reporting aliases
            "Execution Action": final_action,
            "Signal": signal_type,
            "Rule-Based Buy": row.get("Rule-Based Buy",
                                      "Y" if safe_str(row.get("Rule Recommendation")).upper() == "BUY" else "N"),
            "Expected Holding Period": row.get("Expected Holding Period", "2-6 weeks"),
            "Momentum Decision Reason": decision_reason,
            "Momentum Confidence Grade": confidence_grade,
            "Momentum Expected Holding Period": row.get("Expected Holding Period", "2-6 weeks"),
            "Setup Quality": (
                "STRONG" if confidence_score >= 0.70 else
                "MODERATE" if confidence_score >= 0.50 else
                "WEAK"
            ),
            "Entry Timing": (
                "NOW" if final_action == "BUY_NOW" else
                "PULLBACK" if final_action == "BUY_ON_PULLBACK" else
                "WAIT"
            ),
            "Risk Reward Profile": (
                "HIGH" if safe_float(row.get("Risk/Reward T2"), np.nan) >= 3.0 else
                "MEDIUM" if safe_float(row.get("Risk/Reward T1"), np.nan) >= 1.5 else
                "LOW"
            ),

            # L2 fields
            "L2 Status": row.get("L2 Status"),
            "L2 Quality": row.get("L2 Quality"),
            "L2 Imbalance": row.get("L2 Imbalance"),
            "L2 Best Bid": row.get("L2 Best Bid"),
            "L2 Best Ask": row.get("L2 Best Ask"),
            "BID_ASK_SPREAD_PCT": row.get("BID_ASK_SPREAD_PCT"),
            "Execution Style": exec_intel.get("Execution Style"),
            "Execution Order Type": exec_intel.get("Execution Order Type"),
            "Execution Aggression": exec_intel.get("Execution Aggression"),
            "Execution Note": exec_intel.get("Execution Note"),
            "Execution Chase Price": exec_intel.get("Execution Chase Price"),
            "Execution Microprice": exec_intel.get("Execution Microprice"),
        }
    )

    row.update(trade_levels)
    return row
