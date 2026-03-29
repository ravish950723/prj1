from __future__ import annotations

import numpy as np
import pandas as pd

from .config_loader import get_patterns_config


def apply_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    cfg = get_patterns_config()
    enabled = cfg.get("enabled_patterns", {})
    t = cfg.get("thresholds", {})

    hammer_lw_mult = float(t.get("hammer_lower_wick_multiple", 2.0))
    hammer_uw_body = float(t.get("hammer_upper_wick_max_body_multiple", 1.0))
    doji_ratio_max = float(t.get("doji_body_to_range_max", 0.10))
    breakout_vol_min = float(t.get("breakout_volume_ratio_min", 1.5))
    range_bound_max = float(t.get("range_bound_20d_max_pct", 0.08))
    tight_range_max = float(t.get("tight_range_20d_max_pct", 0.04))

    close = pd.to_numeric(df["close"], errors="coerce")
    open_ = pd.to_numeric(df["open"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")

    ema20 = pd.to_numeric(df.get("EMA_20"), errors="coerce")
    ema50 = pd.to_numeric(df.get("EMA_50"), errors="coerce")
    ema200 = pd.to_numeric(df.get("EMA_200"), errors="coerce")
    vwap = pd.to_numeric(df.get("vwap_support"), errors="coerce")

    volume_avg_20 = volume.rolling(20, min_periods=5).mean()
    vol_ratio = volume / volume_avg_20.replace(0, np.nan)

    rolling_20_high = high.rolling(20, min_periods=5).max()
    rolling_20_low = low.rolling(20, min_periods=5).min()

    body = (close - open_).abs()
    range_ = (high - low).replace(0, np.nan)
    lower_wick = (np.minimum(open_, close) - low)
    upper_wick = (high - np.maximum(open_, close))

    new_cols: dict[str, pd.Series] = {}

    if enabled.get("EMA_STACKED_BULLISH", True):
        new_cols["EMA_STACKED_BULLISH"] = ((ema20 > ema50) & (ema50 > ema200)).astype(int)

    if enabled.get("EMA_STACKED_BEARISH", True):
        new_cols["EMA_STACKED_BEARISH"] = ((ema20 < ema50) & (ema50 < ema200)).astype(int)

    if enabled.get("RANGE_BOUND", True):
        new_cols["RANGE_BOUND"] = (((rolling_20_high - rolling_20_low) / close) < range_bound_max).astype(int)

    if enabled.get("TIGHT_RANGE_CONSOLIDATION", True):
        new_cols["TIGHT_RANGE_CONSOLIDATION"] = (((rolling_20_high - rolling_20_low) / close) < tight_range_max).astype(int)

    if enabled.get("RESISTANCE_BREAKOUT", True):
        new_cols["RESISTANCE_BREAKOUT"] = (close > rolling_20_high.shift(1)).astype(int)

    if enabled.get("SUPPORT_BREAKDOWN", True):
        new_cols["SUPPORT_BREAKDOWN"] = (close < rolling_20_low.shift(1)).astype(int)

    if enabled.get("HIGH_VOLUME_BREAKOUT", True):
        new_cols["HIGH_VOLUME_BREAKOUT"] = (
            (close > rolling_20_high.shift(1)) & (vol_ratio > breakout_vol_min)
        ).astype(int)

    if enabled.get("VOLUME_SPIKE", True):
        new_cols["VOLUME_SPIKE"] = (vol_ratio > breakout_vol_min).astype(int)

    if enabled.get("VOLUME_DRY_UP", True):
        new_cols["VOLUME_DRY_UP"] = (vol_ratio < 0.7).astype(int)

    if enabled.get("HAMMER", True):
        new_cols["HAMMER"] = (
            (lower_wick > hammer_lw_mult * body) &
            (upper_wick < hammer_uw_body * body)
        ).astype(int)

    if enabled.get("DOJI", True):
        new_cols["DOJI"] = ((body / range_) < doji_ratio_max).fillna(False).astype(int)

    if enabled.get("VWAP_RECLAIM", True):
        new_cols["VWAP_RECLAIM"] = ((close > vwap) & (close.shift(1) <= vwap.shift(1))).astype(int)

    if enabled.get("VWAP_REJECTION", True):
        new_cols["VWAP_REJECTION"] = ((high > vwap) & (close < vwap)).astype(int)

    if enabled.get("GAP_UP", True):
        new_cols["GAP_UP"] = (low > high.shift(1)).astype(int)

    if enabled.get("GAP_DOWN", True):
        new_cols["GAP_DOWN"] = (high < low.shift(1)).astype(int)

    if not new_cols:
        return df

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)