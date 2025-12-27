# symbol_analysis_complete.py
# Decision-layer analysis: ML + rules + explainability
# NOTE: This file is designed to be called from buy.py via: from symbol_analysis import analyze_symbol

from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import joblib

from config import symbol_to_sector, sector_etfs
from fetching import fetch_data_cached
from compute import compute_indicators, candle_entries_multi
from exit_signals import compute_exit_signals
from macro_features import enrich_with_macro_features


# ----------------------------
# Config + model loading
# ----------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_THRESHOLDS: Dict[str, Any] = {
    "model_thresholds": {"watch": 0.30, "buy": 0.45, "strong_buy": 0.60},
    "boosters": {"min_signal_score_for_boost": 0.08, "min_institutional_for_boost": 0.70},
    "guards": {"cap_if_macro_warn": True},
    "training_reference": {},
}

def _load_thresholds() -> Dict[str, Any]:
    path = os.path.join(_THIS_DIR, "strong_buy_thresholds.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # shallow-merge with defaults
        out = dict(DEFAULT_THRESHOLDS)
        out.update(data or {})
        out["model_thresholds"] = {**DEFAULT_THRESHOLDS["model_thresholds"], **out.get("model_thresholds", {})}
        out["boosters"] = {**DEFAULT_THRESHOLDS["boosters"], **out.get("boosters", {})}
        out["guards"] = {**DEFAULT_THRESHOLDS["guards"], **out.get("guards", {})}
        return out
    except Exception:
        return dict(DEFAULT_THRESHOLDS)

THRESHOLDS = _load_thresholds()

def _load_model() -> Optional[Any]:
    # Prefer calibrated model if present
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


# ----------------------------
# Helpers: trend / signals
# ----------------------------

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if not np.isfinite(v):
            return default
        return v
    except Exception:
        return default

def _trend_from_emas(close: float, ema20: float, ema50: float) -> str:
    if close >= ema20 >= ema50:
        return "Bullish"
    if close <= ema20 <= ema50:
        return "Bearish"
    return "Neutral"

def _calc_signal_score(df: pd.DataFrame) -> float:
    """Aggregate rule-based signals into a [0..1] score."""
    score = 0.0
    total = 0.0

    def add(cond: bool, w: float) -> None:
        nonlocal score, total
        total += w
        if cond:
            score += w

    last = df.iloc[-1]

    add(bool(last.get("vwap_support", np.nan) and last["close"] >= last["vwap_support"]), 0.15)
    add(bool(last.get("ema_uptrend", False)), 0.15)
    add(bool(last.get("macd_cross", False)), 0.10)
    add(bool(last.get("rsi_state", "") in ("RISING", "BULLISH")), 0.10)
    add(bool(last.get("near_support", False)), 0.10)
    add(bool(last.get("volume_surge", 0.0) >= 1.5), 0.10)
    add(bool(last.get("darvas_signal", 0) == 1), 0.20)
    add(bool(last.get("smc_breakout", False)), 0.10)

    if total <= 0:
        return 0.0
    return float(np.clip(score / total, 0.0, 1.0))

def _macro_flags(df: pd.DataFrame) -> Tuple[bool, bool]:
    """
    Simple macro guard:
      - macro_warn True if VIX regime is HIGH or market regime indicates risk-off.
    Uses columns created by enrich_with_macro_features (always present but may be NaN).
    """
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
    macro_ok = not macro_warn
    return macro_ok, macro_warn


# ----------------------------
# Decision logic
# ----------------------------

def final_recommendation(
    *,
    model_prob: float,
    signal_score: float,
    institutional_score: float,
    macro_ok: bool,
    macro_warn: bool,
    thr: Dict[str, Any],
) -> str:
    mt = thr.get("model_thresholds", {})
    watch_thr = float(mt.get("watch", 0.30))
    buy_thr = float(mt.get("buy", 0.45))
    strong_thr = float(mt.get("strong_buy", 0.60))

    boosters = thr.get("boosters", {})
    min_signal = float(boosters.get("min_signal_score_for_boost", 0.08))
    min_insti = float(boosters.get("min_institutional_for_boost", 0.70))

    guards = thr.get("guards", {})
    cap_if_macro_warn = bool(guards.get("cap_if_macro_warn", True))

    # Base tier by model probability
    if model_prob >= strong_thr:
        rec = "STRONG_BUY"
    elif model_prob >= buy_thr:
        rec = "BUY"
    elif model_prob >= watch_thr:
        rec = "WATCH"
    else:
        rec = "WATCH"

    # Optional one-level promotion if confluence is strong
    if (signal_score >= min_signal) and (institutional_score >= min_insti):
        if rec == "WATCH" and model_prob >= watch_thr:
            rec = "BUY"
        elif rec == "BUY" and model_prob >= buy_thr:
            rec = "STRONG_BUY"

    # Macro guard caps risk
    if cap_if_macro_warn and macro_warn:
        if rec in ("BUY", "STRONG_BUY"):
            rec = "WATCH"

    return rec


def decision_reason(
    *,
    rec: str,
    model_prob: float,
    signal_score: float,
    institutional_score: float,
    macro_warn: bool,
) -> str:
    mt = THRESHOLDS["model_thresholds"]
    b = THRESHOLDS["boosters"]

    parts: List[str] = []

    if rec == "STRONG_BUY":
        parts.append(f"Model>{mt['strong_buy']}")
    elif rec == "BUY":
        parts.append(f"Model>{mt['buy']}")
    else:
        parts.append(f"Model>{mt['watch']}?")

    if signal_score >= float(b["min_signal_score_for_boost"]):
        parts.append("SMC/Rule Signal")

    if institutional_score >= float(b["min_institutional_for_boost"]):
        parts.append("Institutional Flow")

    parts.append("⚠ Macro Risk" if macro_warn else "No Macro Risk")
    return " + ".join(parts)


def compute_confidence_grade(
    *,
    recommendation: str,
    model_prob: float,
    signal_score: float,
    institutional_score: float,
    macro_warn: bool,
    adx: float,
    trend_htf: str,
    trend_itf: str,
    trend_ltf: str,
) -> str:
    """A/B/C confidence grade for Excel ranking."""
    if macro_warn:
        return "C"

    trend_bullish = (trend_htf == "Bullish") and (trend_itf == "Bullish" or trend_ltf == "Bullish")

    if recommendation == "STRONG_BUY":
        if model_prob >= 0.70 and signal_score >= 0.10 and institutional_score >= 0.80 and adx >= 25 and trend_bullish:
            return "A"
        return "B"

    if recommendation == "BUY":
        if model_prob >= 0.55 and (signal_score >= 0.08 or institutional_score >= 0.75) and adx >= 20 and trend_bullish:
            return "B"
        return "C"

    return "C"


def expected_holding_period(
    *,
    recommendation: str,
    market_stage: str,
    darvas_signal: int,
    mean_reversion: bool,
    adx: float,
    trend_htf: str,
) -> str:
    """Heuristic holding period guidance (swing/position)."""
    if recommendation not in ("BUY", "STRONG_BUY"):
        return "N/A"

    stage = (market_stage or "").upper()

    if darvas_signal == 1:
        return "1–3 months (breakout/retest)"

    if trend_htf == "Bullish" and adx >= 25 and ("MARKUP" in stage or "ACCUMULATION" in stage):
        return "3–6 months (trend ride)"

    if mean_reversion:
        return "2–6 weeks (mean reversion)"

    if "DISTRIBUTION" in stage:
        return "2–6 weeks (distribution risk)"

    return "4–8 weeks (swing)"


# ----------------------------
# Main entry: analyze_symbol
# ----------------------------

def analyze_symbol(symbol: str, df_raw: Optional[pd.DataFrame] = None) -> Optional[Dict[str, Any]]:
    try:
        # Fetch + indicators
        df = df_raw.copy() if (df_raw is not None and not df_raw.empty) else fetch_data_cached(symbol, "10 Y", "1 day")
        df = compute_indicators(df, symbol=symbol)
        # compute_indicators already attempts macro enrichment; only enrich here if needed.
        if "VIX Vol Regime" not in df.columns and "vix_vol_regime" not in df.columns:
            df = enrich_with_macro_features(df)

        if df is None or df.empty:
            raise ValueError("No data returned")

        last = df.iloc[-1]

        close = safe_float(last.get("close", np.nan))
        ema21 = safe_float(last.get("EMA_21", np.nan))
        ema20 = safe_float(last.get("EMA_20", ema21))
        ema50 = safe_float(last.get("EMA_50", np.nan))
        vwap_support = safe_float(last.get("vwap_support", np.nan))
        darvas_low = safe_float(last.get("darvas_low", np.nan))
        bb_lower = safe_float(last.get("BB_lower", np.nan))
        adx = safe_float(last.get("ADX_14", np.nan))
        volume_weight = safe_float(last.get("volume_weight", 0.0))
        institutional_score = safe_float(last.get("institutional_score", 0.0))
        confidence_score = safe_float(last.get("confidence_score", 0.0))

        # Trend (HTF/ITF/LTF - keep it simple, consistent strings)
        trend = _trend_from_emas(close, ema20, ema50)

        # Basic multi-timeframe using candle entries (if available)
        # If candle entries are not available for a timeframe, we keep neutral.
        trend_itf = str(last.get("itf_trend", trend))
        trend_ltf = str(last.get("ltf_trend", trend))

        # Sector correlation (optional)
        sector_corr = 0.0
        etf_symbol = sector_etfs.get(symbol_to_sector.get(symbol))
        if etf_symbol and etf_symbol != symbol:
            try:
                df_etf = fetch_data_cached(etf_symbol, "10 Y", "1 day", require_today=False)
                df_etf = compute_indicators(df_etf, symbol=etf_symbol)
                s = df[["date", "close"]].rename(columns={"close": "close_sym"})
                e = df_etf[["date", "close"]].rename(columns={"close": "close_etf"})
                merged = s.merge(e, on="date", how="inner")
                if len(merged) >= 21:
                    rolling_corr = (
                        merged["close_sym"].pct_change().rolling(20).corr(merged["close_etf"].pct_change())
                    )
                    val = rolling_corr.iloc[-1]
                    if pd.notna(val) and np.isfinite(val):
                        sector_corr = float(val)
            except Exception:
                sector_corr = 0.0
        df["sector_corr"] = sector_corr

        # Rule-based signal score
        signal_score = _calc_signal_score(df)

        # Macro flags
        macro_ok, macro_warn = _macro_flags(df)

        # ----------------------------
        # Buy / entry price logic (EMA/VWAP/Darvas/BB/Swing)
        # ----------------------------
        swing_low = float(df["low"].tail(5).min()) if "low" in df.columns else close

        price_candidates = [ema21, vwap_support, darvas_low, bb_lower, swing_low]
        price_candidates = [p for p in price_candidates if np.isfinite(p) and p > 0]

        if not price_candidates:
            buy_price = round(close, 2)
        else:
            # Outlier rejection
            q1, q3 = np.percentile(price_candidates, [25, 75])
            iqr = q3 - q1
            filtered = [p for p in price_candidates if (q1 - 1.5 * iqr) <= p <= (q3 + 1.5 * iqr)]
            valid = filtered if filtered else price_candidates

            # ADX-based weights: trend strong => EMA bias; else VWAP bias
            if adx >= 25:
                weights = [0.50, 0.20, 0.10, 0.10, 0.10]
            else:
                weights = [0.20, 0.40, 0.10, 0.10, 0.20]

            # If filtered list changed length, reset weights
            if len(valid) != len(weights):
                weights = [1.0] * len(valid)

            buy_price = round(float(np.average(valid, weights=weights)), 2)

        # Candle-based entry suggestions (optional enrichment)
        try:
            entries_dict = candle_entries_multi(df, weeks_list=[2, 4, 6, 8, 12, 18, 30])
        except Exception:
            entries_dict = {2: float("nan"), 4: float("nan"), 6: float("nan"), 8: float("nan"), 12: float("nan"), 18: float("nan"), 30: float("nan")}

        _valid_entries = [v for v in entries_dict.values() if isinstance(v, (int, float)) and np.isfinite(v)]
        if _valid_entries:
            buy_price = round((buy_price + float(np.nanmean(_valid_entries))) / 2.0, 2)

        # ----------------------------
        # Model probability
        # ----------------------------
        model_proba = 0.0
        if MODEL is not None:
            latest = df.iloc[-1:].copy()
            latest.columns = [str(c).strip() for c in latest.columns]

            expected_features = [str(c).strip() for c in getattr(MODEL, "feature_names_in_", [])]
            if expected_features:
                for col in expected_features:
                    if col not in latest.columns:
                        latest[col] = 0.0
                features = latest.reindex(columns=expected_features).astype(float)
                try:
                    model_proba = float(MODEL.predict_proba(features)[0][1])
                except Exception:
                    model_proba = 0.0

        model_proba = float(np.clip(model_proba, 0.0, 1.0))

        # ----------------------------
        # Final recommendation + explainability
        # ----------------------------
        recommendation = final_recommendation(
            model_prob=model_proba,
            signal_score=signal_score,
            institutional_score=institutional_score,
            macro_ok=macro_ok,
            macro_warn=macro_warn,
            thr=THRESHOLDS,
        )

        decision_reason_str = decision_reason(
            rec=recommendation,
            model_prob=model_proba,
            signal_score=signal_score,
            institutional_score=institutional_score,
            macro_warn=macro_warn,
        )

        decision_reason_parts: List[str] = [decision_reason_str] if decision_reason_str else []

        # Darvas signal raw
        darvas_signal_raw = int(safe_float(last.get("darvas_signal", 0), 0.0))
        mean_rev_val = bool(last.get("mean_reversion", False))
        market_stage_val = str(last.get("market_stage", ""))
        market_substage_val = str(last.get("market_substage", last.get("market_sub_stage", "")))

        # --- Market-stage gating (risk control) ---
        # Avoid BUY/STRONG_BUY in late-cycle / downtrend stages unless the setup is exceptionally strong.
        stage_upper = (market_stage_val or "").strip()
        if stage_upper in ("Mark-Down", "MARK-DOWN", "Mark-Down ", "Mark-Down/Transition"):
            if recommendation in ("BUY", "STRONG_BUY"):
                recommendation = "WATCH"
                decision_reason_parts.append("Stage=Mark-Down → gate to WATCH")
        elif stage_upper in ("Distribution", "DISTRIBUTION"):
            if recommendation == "STRONG_BUY" and macro_ok:
                decision_reason_parts.append("Stage=Distribution (allowed: STRONG_BUY only)")
            elif recommendation in ("BUY", "STRONG_BUY"):
                recommendation = "WATCH"
                decision_reason_parts.append("Stage=Distribution → gate to WATCH")

        # Recompute decision reason after gating adjustments
        decision_reason_str = decision_reason(
            rec=recommendation,
            model_prob=model_proba,
            signal_score=signal_score,
            institutional_score=institutional_score,
            macro_warn=macro_warn,
        )
        if decision_reason_str:
            decision_reason_parts = [decision_reason_str] + [p for p in decision_reason_parts[1:] if p]
        # Append any gating notes
        final_reason = ' + '.join([p for p in decision_reason_parts if p])
        decision_reason_str = final_reason

        confidence_grade = compute_confidence_grade(
            recommendation=recommendation,
            model_prob=model_proba,
            signal_score=signal_score,
            institutional_score=institutional_score,
            macro_warn=macro_warn,
            adx=adx,
            trend_htf=trend,
            trend_itf=trend_itf,
            trend_ltf=trend_ltf,
        )

        holding_period = expected_holding_period(
            recommendation=recommendation,
            market_stage=market_stage_val,
            darvas_signal=darvas_signal_raw,
            mean_reversion=mean_rev_val,
            adx=adx,
            trend_htf=trend,
        )

        # Exits
        exit_info = compute_exit_signals(df, entry_price=buy_price, atr_mult=2.0)

        # Output row
        result: Dict[str, Any] = {
            "Symbol": symbol,
            "Refined Buy Price": float(buy_price),
            "VWAP Support": round(float(vwap_support), 2) if np.isfinite(vwap_support) else np.nan,
            "ADX": round(float(adx), 2) if np.isfinite(adx) else np.nan,
            "Institutional Score": round(float(institutional_score), 2),
            "Volume Weight": round(float(volume_weight), 2),
            "Confidence Score": round(float(confidence_score), 2),
            "Sector Correlation": float(sector_corr),
            "Trend": trend,
            "HTF_Trend": trend,
            "ITF_Trend": trend_itf,
            "LTF_Trend": trend_ltf,
            "Recommendation": recommendation,
            "Decision Reason": decision_reason_str,
            "Confidence Grade": confidence_grade,
            "Expected Holding Period": holding_period,
            "Market Stage": market_stage_val,
            "Market Sub-Stage": market_substage_val,
            "Darvas Breakout %": round(safe_float(last.get("darvas_breakout_pct", 0.0)), 2),
            "Darvas Signal": "✅" if darvas_signal_raw == 1 else "❌",
            "Model Probability": round(model_proba, 4),
            "Signal Score": round(float(signal_score), 4),
            "Volume Surge": "✅" if bool(last.get("volume_surge", False)) else "❌",
            "Near Support": "✅" if bool(last.get("near_support", False)) else "❌",
            # Candle entries
            "Candle Entry 2w": round(float(entries_dict.get(2, np.nan)), 2) if np.isfinite(entries_dict.get(2, np.nan)) else np.nan,
            "Candle Entry 4w": round(float(entries_dict.get(4, np.nan)), 2) if np.isfinite(entries_dict.get(4, np.nan)) else np.nan,
            "Candle Entry 6w": round(float(entries_dict.get(6, np.nan)), 2) if np.isfinite(entries_dict.get(6, np.nan)) else np.nan,
            "Candle Entry 8w": round(float(entries_dict.get(8, np.nan)), 2) if np.isfinite(entries_dict.get(8, np.nan)) else np.nan,
            "Candle Entry 12w": round(float(entries_dict.get(12, np.nan)), 2) if np.isfinite(entries_dict.get(12, np.nan)) else np.nan,
            "Candle Entry 18w": round(float(entries_dict.get(18, np.nan)), 2) if np.isfinite(entries_dict.get(18, np.nan)) else np.nan,
            "Candle Entry 30w": round(float(entries_dict.get(30, np.nan)), 2) if np.isfinite(entries_dict.get(30, np.nan)) else np.nan,
            # Exits
            "Exit Now": exit_info.get("exit_now", False),
            "Atr Trailing Stop": exit_info.get("atr_trailing_stop", np.nan),
            "Exit Reasons": exit_info.get("reasons", ""),
        }

        # Backward-compat: some downstream code expects lowercase
        result["recommendation"] = recommendation

        return result

    except Exception as e:
        print(f"⚠️ Error analyzing {symbol}: {e}")
        return None
