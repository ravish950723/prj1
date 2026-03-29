from __future__ import annotations

from typing import Dict, Iterable
import numpy as np
import pandas as pd

WEEKS_DEFAULT = (2, 4, 6, 8, 12, 18, 30)


def _to_num(s, index=None) -> pd.Series:
    if isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce")
    if s is None:
        if index is None:
            return pd.Series(dtype="float64")
        return pd.Series(np.nan, index=index, dtype="float64")
    try:
        arr = pd.Series(s, index=index)
        return pd.to_numeric(arr, errors="coerce")
    except Exception:
        if index is None:
            return pd.Series(dtype="float64")
        return pd.Series(np.nan, index=index, dtype="float64")


def _to_bool(s, index=None) -> pd.Series:
    if isinstance(s, pd.Series):
        return s.fillna(False).astype(bool)
    if s is None:
        if index is None:
            return pd.Series(dtype="bool")
        return pd.Series(False, index=index, dtype="bool")
    try:
        arr = pd.Series(s, index=index)
        return arr.fillna(False).astype(bool)
    except Exception:
        if index is None:
            return pd.Series(dtype="bool")
        return pd.Series(False, index=index, dtype="bool")


def _safe_last(series: pd.Series, default=np.nan) -> float:
    try:
        if series is None or len(series) == 0:
            return float(default)
        v = float(series.iloc[-1])
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _safe_tail_min(series: pd.Series, bars: int, default=np.nan) -> float:
    try:
        if series is None or len(series) == 0:
            return float(default)
        v = float(series.tail(max(1, int(bars))).min())
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _safe_tail_max(series: pd.Series, bars: int, default=np.nan) -> float:
    try:
        if series is None or len(series) == 0:
            return float(default)
        v = float(series.tail(max(1, int(bars))).max())
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _clamp(value: float, lo: float, hi: float) -> float:
    if not np.isfinite(value):
        return np.nan
    if np.isfinite(lo):
        value = max(value, lo)
    if np.isfinite(hi):
        value = min(value, hi)
    return float(value)


def _mean_valid(values) -> float:
    vals = []
    for v in values:
        try:
            x = float(v)
            if np.isfinite(x) and x > 0:
                vals.append(x)
        except Exception:
            pass
    return float(np.mean(vals)) if vals else np.nan


def _median_valid(values) -> float:
    vals = []
    for v in values:
        try:
            x = float(v)
            if np.isfinite(x) and x > 0:
                vals.append(x)
        except Exception:
            pass
    return float(np.median(vals)) if vals else np.nan


def _robust_filter(values) -> list[float]:
    arr = np.array(
        [float(v) for v in values if v is not None and np.isfinite(v) and float(v) > 0],
        dtype=float,
    )
    if arr.size == 0:
        return []
    if arr.size < 4:
        return arr.tolist()

    med = np.nanmedian(arr)
    mad = np.nanmedian(np.abs(arr - med))
    if not np.isfinite(mad) or mad <= 0:
        return arr.tolist()

    keep = np.abs(arr - med) <= 2.5 * mad
    filtered = arr[keep]
    if filtered.size == 0:
        return arr.tolist()
    return filtered.tolist()


def _stage_name(df: pd.DataFrame) -> str:
    try:
        s = str(df.get("market_stage", pd.Series(["Neutral/Transition"])).iloc[-1]).strip().lower()
    except Exception:
        s = "neutral/transition"

    if "mark-up" in s or "markup" in s:
        return "markup"
    if "accum" in s:
        return "accumulation"
    if "distribution" in s:
        return "distribution"
    if "mark-down" in s or "markdown" in s:
        return "markdown"
    return "neutral"


def _substage_name(df: pd.DataFrame) -> str:
    try:
        return str(df.get("market_substage", pd.Series(["NEUTRAL_RANGE"])).iloc[-1]).strip().upper()
    except Exception:
        return "NEUTRAL_RANGE"


def _trend_strength(df: pd.DataFrame) -> float:
    adx = _safe_last(_to_num(df.get("ADX_14"), index=df.index), default=np.nan)
    if not np.isfinite(adx):
        return 0.0
    if adx >= 35:
        return 1.0
    if adx >= 25:
        return 0.75
    if adx >= 18:
        return 0.45
    return 0.2


def _institutional_bias(df: pd.DataFrame) -> float:
    inst = _safe_last(_to_num(df.get("institutional_score"), index=df.index), default=np.nan)
    conf = _safe_last(_to_num(df.get("confidence_score"), index=df.index), default=np.nan)

    score = 0.0
    if np.isfinite(inst):
        score += 0.7 * np.clip(inst, 0.0, 1.0)
    if np.isfinite(conf):
        score += 0.3 * np.clip(conf, 0.0, 1.0)
    return float(np.clip(score, 0.0, 1.0))


def _fallback_price(df: pd.DataFrame) -> float:
    close = _safe_last(_to_num(df.get("close"), index=df.index), default=np.nan)
    ema21 = _safe_last(_to_num(df.get("EMA_21"), index=df.index), default=np.nan)
    ema50 = _safe_last(_to_num(df.get("EMA_50"), index=df.index), default=np.nan)
    low = _to_num(df.get("low"), index=df.index)

    vals = _robust_filter([
        close * 0.97 if np.isfinite(close) else np.nan,
        ema21,
        ema50,
        _safe_tail_min(low, 20, default=np.nan),
    ])
    return _median_valid(vals)


def _level_or_nan(v: float) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) and x > 0 else np.nan
    except Exception:
        return np.nan


def _valid_levels_desc(values) -> list[float]:
    vals = _robust_filter(values)
    return sorted(vals, reverse=True)


def _valid_levels_asc(values) -> list[float]:
    vals = _robust_filter(values)
    return sorted(vals)


def _pick_shallow(values) -> float:
    vals = _valid_levels_desc(values)
    return vals[0] if vals else np.nan


def _pick_medium(values) -> float:
    vals = _valid_levels_asc(values)
    if not vals:
        return np.nan
    return vals[len(vals) // 2]


def _pick_deep(values) -> float:
    vals = _valid_levels_asc(values)
    return vals[0] if vals else np.nan


def _is_breakout_context(substage: str, darvas_signal: int, macd_cross: bool, vol_surge_ratio: float) -> bool:
    sub = str(substage or "").upper()
    return bool(
        darvas_signal == 1
        or macd_cross
        or vol_surge_ratio >= 1.6
        or "BREAKOUT" in sub
        or "VWAP_RECLAIM" in sub
        or "RECLAIM" in sub
        or "ACCELERATION" in sub
        or "CTA_BREAKOUT" in sub
    )


def _is_pullback_context(substage: str, near_support: bool, close: float, ema21: float, vwap: float) -> bool:
    sub = str(substage or "").upper()
    if (
            "PULLBACK" in sub
            or "SMART_MONEY_ENTRY" in sub
            or "SPRING" in sub
            or "BASE" in sub
            or "SECONDARY_TEST" in sub
    ):
        return True

    checks = []
    if np.isfinite(close) and np.isfinite(ema21) and close > 0:
        checks.append(abs(close - ema21) / close <= 0.035)
    if np.isfinite(close) and np.isfinite(vwap) and close > 0:
        checks.append(abs(close - vwap) / close <= 0.04)
    return bool(near_support or any(checks))


def _cap_not_chasing(entry: float, close: float, pct_cap: float) -> float:
    if not (np.isfinite(entry) and np.isfinite(close) and close > 0):
        return entry
    return min(entry, close * pct_cap)


def _pick_markup_level(
        *,
        close: float,
        ema21: float,
        ema50: float,
        vwap: float,
        bb_lower: float,
        swing_low: float,
        atr: float,
        breakout_context: bool,
        pullback_context: bool,
        trend_strength: float,
        vol_surge_ratio: float,
        institutional_bias: float,
        week: int,
) -> float:
    shallow_cluster = _robust_filter([ema21, vwap, ema50])
    medium_cluster = _robust_filter([ema50, bb_lower, swing_low])
    deep_cluster = _robust_filter([bb_lower, swing_low])

    if breakout_context:
        if week <= 4:
            entry = _pick_shallow([ema21, vwap, close - 0.25 * atr if np.isfinite(atr) else np.nan])
            entry = _cap_not_chasing(entry, close, 0.995)
        elif week <= 8:
            entry = _pick_medium([ema21, vwap, ema50, bb_lower])
        else:
            entry = _pick_deep([ema50, bb_lower, swing_low])

    elif pullback_context:
        if week <= 4:
            entry = _pick_medium([ema21, vwap, ema50])
            entry = _cap_not_chasing(entry, close, 0.9925)
        elif week <= 8:
            entry = _pick_medium([ema21, ema50, bb_lower, swing_low])
        else:
            entry = _pick_deep([ema50, bb_lower, swing_low])

    else:
        if week <= 4:
            entry = _pick_medium([ema21, vwap, ema50])
            entry = _cap_not_chasing(entry, close, 0.99)
        elif week <= 8:
            entry = _pick_medium([ema50, bb_lower, swing_low, vwap])
        else:
            entry = _pick_deep([bb_lower, swing_low, ema50])

    if np.isfinite(entry) and np.isfinite(atr):
        if trend_strength >= 0.75 or vol_surge_ratio >= 1.8 or institutional_bias >= 0.75:
            if week <= 4:
                entry += 0.05 * atr
            elif week <= 8:
                entry += 0.02 * atr
        elif trend_strength <= 0.25:
            if week >= 8:
                entry -= 0.05 * atr

    if week <= 4:
        entry = _cap_not_chasing(entry, close, 0.9975)

    return entry


def _pick_accumulation_level(
        *,
        close: float,
        ema21: float,
        ema50: float,
        vwap: float,
        bb_lower: float,
        swing_low: float,
        atr: float,
        breakout_context: bool,
        pullback_context: bool,
        trend_strength: float,
        vol_surge_ratio: float,
        institutional_bias: float,
        week: int,
) -> float:
    base_low = _pick_deep([swing_low, bb_lower, ema50])
    tactical = _pick_medium([vwap, ema21, ema50])
    structural = _pick_deep([swing_low, bb_lower, ema50])

    if breakout_context:
        if week <= 4:
            entry = _pick_medium([ema21, vwap, ema50])
            if np.isfinite(atr):
                entry -= 0.03 * atr
        elif week <= 8:
            entry = _pick_medium([ema50, bb_lower, swing_low])
        else:
            entry = structural

    elif pullback_context:
        if week <= 4:
            entry = _pick_medium([ema50, bb_lower, vwap])
        elif week <= 8:
            entry = _pick_medium([bb_lower, swing_low, ema50])
        else:
            entry = structural

    else:
        # Explicit spread to avoid flat ladder in accumulation
        if np.isfinite(base_low):
            if week <= 4:
                entry = base_low * 1.02
            elif week <= 8:
                entry = base_low * 1.00
            elif week <= 12:
                entry = base_low * 0.99
            else:
                entry = base_low * 0.98
        else:
            if week <= 4:
                entry = tactical
            elif week <= 8:
                entry = _pick_medium([ema50, bb_lower, swing_low])
            else:
                entry = structural

    if np.isfinite(entry) and np.isfinite(atr):
        if institutional_bias >= 0.75 and vol_surge_ratio >= 1.5 and week <= 6:
            entry += 0.03 * atr
        elif trend_strength <= 0.25 and week >= 8:
            entry -= 0.03 * atr

    return entry


def _pick_defensive_level(
        *,
        close: float,
        ema50: float,
        bb_lower: float,
        swing_low: float,
        atr: float,
        week: int,
) -> float:
    if week <= 4:
        entry = _pick_medium([ema50, bb_lower, swing_low])
    elif week <= 8:
        entry = _pick_deep([ema50, bb_lower, swing_low])
    else:
        entry = _pick_deep([bb_lower, swing_low])

    if np.isfinite(entry) and np.isfinite(atr):
        entry -= 0.05 * atr
    return entry


def _pick_neutral_level(
        *,
        close: float,
        ema21: float,
        ema50: float,
        vwap: float,
        bb_lower: float,
        swing_low: float,
        atr: float,
        week: int,
) -> float:
    # Neutral must be more defensive than markup
    if week <= 4:
        entry = _pick_deep([ema50, bb_lower, vwap])
    elif week <= 8:
        entry = _pick_deep([bb_lower, swing_low, ema50])
    else:
        entry = _pick_deep([bb_lower, swing_low])

    if np.isfinite(entry) and np.isfinite(atr) and week >= 8:
        entry -= 0.02 * atr
    return entry


def candle_entries_multi(
        df: pd.DataFrame,
        weeks_list: Iterable[int] = WEEKS_DEFAULT,
) -> Dict[int, float]:
    out: Dict[int, float] = {}

    if df is None or df.empty:
        return {int(w): np.nan for w in weeks_list}

    low = _to_num(df.get("low"), index=df.index)
    close = _to_num(df.get("close"), index=df.index)
    ema21 = _to_num(df.get("EMA_21"), index=df.index)
    ema50 = _to_num(df.get("EMA_50"), index=df.index)
    bb_lower = _to_num(df.get("BB_lower"), index=df.index)
    vwap_support = _to_num(df.get("vwap_support"), index=df.index)
    atr = _to_num(df.get("ATR_14"), index=df.index)
    volume_weight = _to_num(df.get("volume_weight"), index=df.index)
    vol_surge_ratio = _to_num(
        df.get("VOL_SURGE_RATIO", df.get("volume_surge_ratio", df.get("volume_weight"))),
        index=df.index,
    )
    darvas_signal = _to_num(df.get("darvas_signal"), index=df.index)
    macd_cross = _to_bool(
        df.get("MACD_Crossover", df.get("MACD_crossover", df.get("MACD_CROSSOVER"))),
        index=df.index,
    )
    near_support = _to_bool(df.get("near_support"), index=df.index)

    last_close = _safe_last(close, default=np.nan)
    last_ema21 = _safe_last(ema21, default=np.nan)
    last_ema50 = _safe_last(ema50, default=np.nan)
    last_bb_lower = _safe_last(bb_lower, default=np.nan)
    last_vwap = _safe_last(vwap_support, default=np.nan)
    last_atr = _safe_last(atr, default=np.nan)
    last_vol_surge = _safe_last(vol_surge_ratio, default=np.nan)
    last_volume_weight = _safe_last(volume_weight, default=np.nan)
    last_darvas_signal = int(_safe_last(darvas_signal, default=0))
    last_macd_cross = bool(macd_cross.iloc[-1]) if len(macd_cross) else False
    last_near_support = bool(near_support.iloc[-1]) if len(near_support) else False

    stage = _stage_name(df)
    substage = _substage_name(df)
    substage_conf = _safe_last(_to_num(df.get("substage_confidence"), index=df.index), default=0.0)
    trend_strength = _trend_strength(df)
    inst_bias = _institutional_bias(df)

    if not np.isfinite(last_vol_surge) or last_vol_surge <= 0:
        last_vol_surge = last_volume_weight if np.isfinite(last_volume_weight) else 1.0

    breakout_context = _is_breakout_context(
        substage=substage,
        darvas_signal=last_darvas_signal,
        macd_cross=last_macd_cross,
        vol_surge_ratio=last_vol_surge if np.isfinite(last_vol_surge) else 1.0,
    )
    pullback_context = _is_pullback_context(
        substage=substage,
        near_support=last_near_support,
        close=last_close,
        ema21=last_ema21,
        vwap=last_vwap,
    )

    if not np.isfinite(last_atr) or last_atr <= 0:
        last_atr = last_close * 0.03 if np.isfinite(last_close) else np.nan

    for w in weeks_list:
        bars = max(5, int(w) * 5)
        swing_low = _safe_tail_min(low, bars, default=np.nan)

        if stage == "markup":
            entry = _pick_markup_level(
                close=last_close,
                ema21=last_ema21,
                ema50=last_ema50,
                vwap=last_vwap,
                bb_lower=last_bb_lower,
                swing_low=swing_low,
                atr=last_atr,
                breakout_context=breakout_context,
                pullback_context=pullback_context,
                trend_strength=trend_strength,
                vol_surge_ratio=last_vol_surge if np.isfinite(last_vol_surge) else 1.0,
                institutional_bias=inst_bias,
                week=int(w),
            )
        elif stage == "accumulation":
            entry = _pick_accumulation_level(
                close=last_close,
                ema21=last_ema21,
                ema50=last_ema50,
                vwap=last_vwap,
                bb_lower=last_bb_lower,
                swing_low=swing_low,
                atr=last_atr,
                breakout_context=breakout_context,
                pullback_context=pullback_context,
                trend_strength=trend_strength,
                vol_surge_ratio=last_vol_surge if np.isfinite(last_vol_surge) else 1.0,
                institutional_bias=inst_bias,
                week=int(w),
            )
        elif stage in {"distribution", "markdown"}:
            entry = _pick_defensive_level(
                close=last_close,
                ema50=last_ema50,
                bb_lower=last_bb_lower,
                swing_low=swing_low,
                atr=last_atr,
                week=int(w),
            )
        else:
            entry = _pick_neutral_level(
                close=last_close,
                ema21=last_ema21,
                ema50=last_ema50,
                vwap=last_vwap,
                bb_lower=last_bb_lower,
                swing_low=swing_low,
                atr=last_atr,
                week=int(w),
            )

        floor_candidates = _robust_filter([
            swing_low,
            last_bb_lower,
            last_vwap * 0.97 if np.isfinite(last_vwap) else np.nan,
        ])
        floor_val = min(floor_candidates) if floor_candidates else np.nan
        cap_val = last_close if np.isfinite(last_close) else np.nan

        # horizon-aware floor/cap so longer horizons can sit deeper
        if np.isfinite(floor_val):
            if stage == "accumulation":
                floor_mult = {
                    2: 1.020, 4: 1.010, 6: 1.000, 8: 0.995, 12: 0.985, 18: 0.975, 30: 0.965
                }.get(int(w), 1.0)
            elif stage in {"distribution", "markdown"}:
                floor_mult = {
                    2: 1.005, 4: 1.000, 6: 0.992, 8: 0.985, 12: 0.975, 18: 0.965, 30: 0.955
                }.get(int(w), 1.0)
            else:
                floor_mult = {
                    2: 1.010, 4: 1.000, 6: 0.995, 8: 0.990, 12: 0.980, 18: 0.970, 30: 0.960
                }.get(int(w), 1.0)
            floor_val = floor_val * floor_mult

        if np.isfinite(cap_val):
            cap_mult = {
                2: 0.9975, 4: 0.9970, 6: 0.9960, 8: 0.9950, 12: 0.9930, 18: 0.9910, 30: 0.9890
            }.get(int(w), 0.9950)
            cap_val = cap_val * cap_mult

        entry = _clamp(entry, floor_val, cap_val)
        out[int(w)] = round(entry, 4) if np.isfinite(entry) else np.nan

    # enforce a realistic ladder: shorter horizons shall be shallower than longer horizons
    ordered = sorted(out.keys())
    prev_val = np.nan
    for i, wk in enumerate(ordered):
        v = out[wk]
        if not np.isfinite(v):
            continue
        if np.isfinite(prev_val):
            min_step = (0.0008 + 0.0002 * i) * (last_atr if np.isfinite(last_atr) else max(last_close * 0.002, 0.01))
            if stage == "accumulation":
                v = min(v, prev_val - min_step)
            elif stage in {"distribution", "markdown"}:
                v = min(v, prev_val - min_step * 0.8)
            else:
                v = min(v, prev_val - min_step * 0.9)
        out[wk] = round(v, 4) if np.isfinite(v) else np.nan
        prev_val = out[wk]

    # if everything still collapses to one level, inject a deterministic week ladder
    vals = [round(v, 4) for v in out.values() if np.isfinite(v)]
    if len(set(vals)) <= 1 and vals:
        base = vals[0]
        atr_unit = last_atr if np.isfinite(last_atr) and last_atr > 0 else max(last_close * 0.01, 0.05)
        for i, wk in enumerate(ordered):
            # shorter windows shallower, longer windows deeper
            bump = (0.04 + 0.025 * i) * atr_unit
            out[wk] = round(max(base - bump, base * 0.85), 4)

    return out


def _safe_prob(x, default=np.nan) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return float(np.clip(v, 0.0, 1.0))
        return default
    except Exception:
        return default


def _ml_entry_bias(
        *,
        model_probability: float,
        stage: str,
        breakout_context: bool,
        pullback_context: bool,
) -> Dict[str, object]:
    """
    Decide how aggressive the entry should be.

    Returns:
      {
        "target_key": one of 2/4/6/8/12,
        "mode": "aggressive" | "balanced" | "patient" | "defensive",
        "atr_adjust": float
      }
    """
    p = _safe_prob(model_probability, np.nan)
    stage = str(stage or "").lower()

    if not np.isfinite(p):
        # fallback: keep existing non-ML behavior centered around 4w/6w
        if stage in {"distribution", "markdown"}:
            return {"target_key": 8, "mode": "defensive", "atr_adjust": -0.03}
        if breakout_context:
            return {"target_key": 4, "mode": "balanced", "atr_adjust": 0.01}
        if pullback_context:
            return {"target_key": 6, "mode": "balanced", "atr_adjust": 0.00}
        return {"target_key": 6, "mode": "balanced", "atr_adjust": 0.00}

    # Defensive stages always deeper unless ML is extremely strong
    if stage in {"distribution", "markdown"}:
        if p >= 0.85 and breakout_context:
            return {"target_key": 6, "mode": "balanced", "atr_adjust": -0.01}
        return {"target_key": 8, "mode": "defensive", "atr_adjust": -0.03}

    # Strong conviction
    if p >= 0.75:
        if breakout_context:
            return {"target_key": 2, "mode": "aggressive", "atr_adjust": 0.03}
        if pullback_context:
            return {"target_key": 4, "mode": "aggressive", "atr_adjust": 0.02}
        return {"target_key": 4, "mode": "aggressive", "atr_adjust": 0.01}

    # Good conviction
    if p >= 0.60:
        if breakout_context:
            return {"target_key": 4, "mode": "balanced", "atr_adjust": 0.02}
        if pullback_context:
            return {"target_key": 4, "mode": "balanced", "atr_adjust": 0.01}
        return {"target_key": 6, "mode": "balanced", "atr_adjust": 0.00}

    # Marginal conviction
    if p >= 0.45:
        if stage == "accumulation":
            return {"target_key": 6, "mode": "patient", "atr_adjust": -0.01}
        return {"target_key": 8, "mode": "patient", "atr_adjust": -0.02}

    # Weak conviction
    return {"target_key": 12, "mode": "defensive", "atr_adjust": -0.03}


def _entry_from_key(entries: Dict[int, float], key: int) -> float:
    v = entries.get(int(key), np.nan)
    try:
        v = float(v)
        return v if np.isfinite(v) and v > 0 else np.nan
    except Exception:
        return np.nan



def compute_entry_prices(
    df: pd.DataFrame,
    model_probability: float | None = None,
    symbol: str = "",
) -> Dict[str, float]:
    out: Dict[str, float] = {}

    if df is None or df.empty:
        out["Refined Buy Price"] = np.nan
        for w in WEEKS_DEFAULT:
            out[f"Candle Entry {w}w"] = np.nan
        out["ML Entry Target"] = ""
        out["ML Entry Mode"] = ""
        out["ML Entry Bias ATR"] = np.nan
        return out

    close = _to_num(df.get("close"), index=df.index)
    low = _to_num(df.get("low"), index=df.index)
    ema21 = _to_num(df.get("EMA_21"), index=df.index)
    ema50 = _to_num(df.get("EMA_50"), index=df.index)
    ema200 = _to_num(df.get("EMA_200"), index=df.index)
    vwap_support = _to_num(df.get("vwap_support"), index=df.index)
    darvas_low = _to_num(df.get("darvas_low"), index=df.index)
    bb_lower = _to_num(df.get("BB_lower"), index=df.index)
    atr = _to_num(df.get("ATR_14"), index=df.index)
    volume_weight = _to_num(df.get("volume_weight"), index=df.index)
    vol_surge_ratio = _to_num(
        df.get("VOL_SURGE_RATIO", df.get("volume_surge_ratio", df.get("volume_weight"))),
        index=df.index,
    )
    darvas_signal = _to_num(df.get("darvas_signal"), index=df.index)
    macd_cross = _to_bool(
        df.get("MACD_Crossover", df.get("MACD_crossover", df.get("MACD_CROSSOVER"))),
        index=df.index,
    )
    near_support = _to_bool(df.get("near_support"), index=df.index)

    last_close = _safe_last(close, default=np.nan)
    last_ema21 = _safe_last(ema21, default=np.nan)
    last_ema50 = _safe_last(ema50, default=np.nan)
    last_ema200 = _safe_last(ema200, default=np.nan)
    last_vwap = _safe_last(vwap_support, default=np.nan)
    last_darvas_low = _safe_last(darvas_low, default=np.nan)
    last_bb_lower = _safe_last(bb_lower, default=np.nan)
    last_atr = _safe_last(atr, default=np.nan)
    last_volume_weight = _safe_last(volume_weight, default=np.nan)
    last_vol_surge = _safe_last(vol_surge_ratio, default=np.nan)
    last_darvas_signal = int(_safe_last(darvas_signal, default=0))
    last_macd_cross = bool(macd_cross.iloc[-1]) if len(macd_cross) else False
    last_near_support = bool(near_support.iloc[-1]) if len(near_support) else False

    if not np.isfinite(last_vol_surge) or last_vol_surge <= 0:
        last_vol_surge = last_volume_weight if np.isfinite(last_volume_weight) else 1.0

    swing_low_20 = _safe_tail_min(low, 20, default=np.nan)
    swing_low_40 = _safe_tail_min(low, 40, default=np.nan)

    stage = _stage_name(df)
    substage = _substage_name(df)
    trend_strength = _trend_strength(df)
    inst_bias = _institutional_bias(df)
    substage_conf = _safe_last(
        _to_num(df.get("substage_confidence"), index=df.index),
        default=0.0,
    )
    breakout_context = _is_breakout_context(
        substage=substage,
        darvas_signal=last_darvas_signal,
        macd_cross=last_macd_cross,
        vol_surge_ratio=last_vol_surge if np.isfinite(last_vol_surge) else 1.0,
    )
    pullback_context = _is_pullback_context(
        substage=substage,
        near_support=last_near_support,
        close=last_close,
        ema21=last_ema21,
        vwap=last_vwap,
    )

    if not np.isfinite(last_atr) or last_atr <= 0:
        last_atr = last_close * 0.03 if np.isfinite(last_close) else np.nan

    entries = candle_entries_multi(df, weeks_list=WEEKS_DEFAULT)

    for w in WEEKS_DEFAULT:
        out[f"Candle Entry {w}w"] = entries.get(int(w), np.nan)

    e2 = entries.get(2, np.nan)
    e4 = entries.get(4, np.nan)
    e6 = entries.get(6, np.nan)
    e8 = entries.get(8, np.nan)
    e12 = entries.get(12, np.nan)
    e18 = entries.get(18, np.nan)
    e30 = entries.get(30, np.nan)

    # ML-aligned execution selection
    if np.isfinite(substage_conf) and substage_conf < 0.35 and np.isfinite(model_probability):
        model_probability = min(float(model_probability), 0.59)

    p = float(model_probability) if model_probability is not None else np.nan

    ml_mode = "balanced"
    ml_target = "4w"
    ml_bias = 0.0
    refined_buy = e6 if np.isfinite(e6) else (e8 if np.isfinite(e8) else _fallback_price(df))

    # -----------------------------
    # STEP 1: ML decides entry level
    # -----------------------------
    if np.isfinite(p):

        if p >= 0.75:
            refined_buy = e2 if np.isfinite(e2) else e4
            ml_mode = "aggressive"
            ml_target = "2w"
            ml_bias = 0.03

        elif p >= 0.60:
            refined_buy = e4 if np.isfinite(e4) else e6
            ml_mode = "balanced"
            ml_target = "4w"
            ml_bias = 0.01

        elif p >= 0.45:
            refined_buy = e6 if np.isfinite(e6) else e8
            ml_mode = "patient"
            ml_target = "6w"
            ml_bias = -0.01

        elif p >= 0.25:
            refined_buy = e8 if np.isfinite(e8) else e12
            ml_mode = "defensive"
            ml_target = "8w"
            ml_bias = -0.03

        else:
            # 🔥 ULTRA LOW CONFIDENCE (NOW CORRECT)
            refined_buy = e12 if np.isfinite(e12) else e18
            ml_mode = "extreme_value"
            ml_target = "12w+"
            ml_bias = -0.05

    # -----------------------------
    # STEP 2: Stage safety override
    # -----------------------------
    if stage in {"distribution", "markdown"}:
        refined_buy = e8 if np.isfinite(e8) else refined_buy
        ml_mode = "defensive"
        ml_target = "8w"
        ml_bias -= 0.02

    # -----------------------------
    # STEP 3: ATR fine tuning
    # -----------------------------
    if np.isfinite(refined_buy) and np.isfinite(last_atr):
        refined_buy += ml_bias * last_atr


    # -----------------------------
    # STEP 4: Ladder alignment (CRITICAL FIX)
    # -----------------------------
    if np.isfinite(refined_buy):

        if ml_target == "2w":
            refined_buy = _median_valid([refined_buy, e2, e4])

        elif ml_target == "4w":
            refined_buy = _median_valid([refined_buy, e4, e6])

        elif ml_target == "6w":
            refined_buy = _median_valid([refined_buy, e6, e8])

        elif ml_target == "8w":
            refined_buy = _median_valid([refined_buy, e8, e12])

    # -----------------------------
    # STEP 4.6: Structural floor protection
    # -----------------------------
    if np.isfinite(e30) and np.isfinite(refined_buy):
        floor_price = e30 * 0.95
        if refined_buy < floor_price:
            print(
                f"[{symbol}] Structural floor applied | "
                f"30w={e30:.2f} | old={refined_buy:.2f} | new={floor_price:.2f} | "
                f"Never buy below ~95% of macro support"
            )
            refined_buy = floor_price

    # -----------------------------
    # STEP 4.7: Substage-aware micro adjustment
    # -----------------------------
    sub_u = str(substage or "").upper()
    if np.isfinite(refined_buy):
        if (
            ("BREAKOUT" in sub_u or "VWAP_RECLAIM" in sub_u or "CTA_BREAKOUT" in sub_u)
            and stage in {"markup", "accumulation"}
            and substage_conf >= 0.55
        ):
            refined_buy *= 1.003
        elif (
            "PULLBACK" in sub_u
            or "SMART_MONEY_ENTRY" in sub_u
            or "SPRING" in sub_u
            or "BASE" in sub_u
        ):
            refined_buy *= 0.995
        elif (
            "FAILED_BREAKOUT" in sub_u
            or "UTAD" in sub_u
            or "UPTHRUST" in sub_u
            or "PANIC" in sub_u
            or "MOMENTUM_SELLOFF" in sub_u
        ):
            refined_buy *= 0.975

    # -----------------------------
    # STEP 5: Final sanity clamp + fallback
    # -----------------------------
    fallback = _fallback_price(df)
    if np.isfinite(refined_buy) and np.isfinite(last_close) and last_close > 0:
        refined_buy = min(refined_buy, last_close * 0.9975)
        refined_buy = max(refined_buy, last_close * 0.82)

    if not np.isfinite(refined_buy) or refined_buy <= 0:
        refined_buy = fallback

    # -----------------------------
    # STEP 6: Save outputs
    # -----------------------------
    out["Refined Buy Price"] = round(refined_buy, 4) if np.isfinite(refined_buy) else np.nan
    out["ML Entry Target"] = ml_target
    out["ML Entry Mode"] = ml_mode
    out["ML Entry Bias ATR"] = round(ml_bias, 4)

    return out