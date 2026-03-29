from __future__ import annotations

import pandas as pd

from .config_loader import get_quant_config


def apply_quant_scores(df: pd.DataFrame) -> pd.DataFrame:
    cfg = get_quant_config()

    sig_w = cfg.get("signal_score_weights", {})
    conf_w = cfg.get("confidence_score_weights", {})
    regime_w = cfg.get("regime_quality_weights", {})
    thresholds = cfg.get("thresholds", {})

    signal_score = (
            df["EMA_uptrend"].astype(int) * float(sig_w.get("EMA_uptrend", 0.15))
            + df["MACD_crossover"].astype(int) * float(sig_w.get("MACD_crossover", 0.15))
            + (df["strong_trend"] > 0).astype(int) * float(sig_w.get("strong_trend", 0.15))
            + df["darvas_signal"].fillna(0).astype(int) * float(sig_w.get("darvas_signal", 0.10))
            + df["tight_range"].astype(int) * float(sig_w.get("tight_range", 0.10))
            + df["volume_surge"].astype(int) * float(sig_w.get("volume_surge", 0.10))
            + df["near_support"].astype(int) * float(sig_w.get("near_support", 0.05))
            + (df["green_candles"] / 3.0).clip(lower=0, upper=1) * float(sig_w.get("green_candles", 0.05))
            + df["above_EMA200"].astype(int) * float(sig_w.get("above_EMA200", 0.05))
            + (df["MACD_hist_slope"] > 0).astype(int) * float(sig_w.get("MACD_hist_slope_positive", 0.05))
    ).fillna(0.0)

    df["substage_weight"] = df["substage_confidence"].clip(0.3, 1.0)
    df["signal_score"] = signal_score * df["substage_weight"]
    stage_norm = (df["market_stage"].fillna("").astype(str).str.lower().str.replace("-", "", regex=False))

    bull_mask = stage_norm.str.contains("markup")
    bear_mask = stage_norm.str.contains("markdown")

    df.loc[bull_mask, "signal_score"] *= 1.10
    df.loc[bear_mask, "signal_score"] *= 0.70
    df.loc[bear_mask & (df["substage_confidence"] < 0.6), "signal_score"] *= 0.5

    df["refined_buy_signal"] = df["signal_score"] >= float(thresholds.get("refined_buy_signal_min", 0.75))

    df["confidence_score"] = (
        df["institutional_score"].fillna(0.0) * float(conf_w.get("institutional_score", 0.50))
        + df["volume_weight"].fillna(0.0) * float(conf_w.get("volume_weight", 0.30))
        + (df["ADX_14"].fillna(0.0) / 100.0) * float(conf_w.get("adx_component", 0.20))
    ).clip(0.0, 1.0)

    stage_strength = df.get("stage_strength_score", pd.Series(0.0, index=df.index)).fillna(0.0)
    substage_conf = df.get("substage_confidence", pd.Series(0.0, index=df.index)).fillna(0.0)

    confidence_score = df.get("confidence_score", pd.Series(0.0, index=df.index)).fillna(0.0)

    df["regime_quality_score"] = (
            stage_strength * float(regime_w.get("stage_strength_score", 0.45))
            + substage_conf * float(regime_w.get("substage_confidence", 0.25))
            + df["signal_score"] * float(regime_w.get("signal_score", 0.15))
            + confidence_score * float(regime_w.get("confidence_score", 0.15))
    ).clip(0.0, 1.0)

    return df