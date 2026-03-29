
import pandas_ta as ta
import numpy as np
import pandas as pd

from pathlib import Path
import os
import re

from .entry_prices import compute_entry_prices
from .fetching import fetch_data_cached
from .institutional_investor import score_institutional_investor
from .darvas import darvas_box_signal
from .macro_features import enrich_with_macro_features
from .pattern_engine import apply_pattern_features
from .quant_engine import apply_quant_scores
from .compute_orderflow_institutional import _add_orderflow_institutional_layer as _oflow_impl

# ================================
# Scalar/boolean safety helpers
# Avoid: "The truth value of a Series is ambiguous"
# ================================
def _as_scalar(x, default=None):
    """Convert Series/array/scalar to a float scalar (last element for Series)."""
    try:
        import numpy as _np
        if default is None:
            default = _np.nan
        import pandas as _pd
        if isinstance(x, _pd.Series):
            if x.empty:
                return default
            x = x.iloc[-1]
        elif isinstance(x, _np.ndarray):
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
        if _np.isfinite(v):
            return v
        return default
    except Exception:
        return default

def _as_bool(x, default=False):
    """Convert Series/array/scalar to bool (last element for Series)."""
    try:
        import numpy as _np
        import pandas as _pd
        if isinstance(x, _pd.Series):
            if x.empty:
                return default
            x = x.iloc[-1]
        elif isinstance(x, _np.ndarray):
            if x.size == 0:
                return default
            x = x.reshape(-1)[-1]
        elif isinstance(x, (list, tuple)):
            if len(x) == 0:
                return default
            x = x[-1]
        if x is None:
            return default
        if isinstance(x, (float, _np.floating)) and _np.isnan(x):
            return default
        return bool(x)
    except Exception:
        return default



DEBUG = os.getenv("BUYPIPE_DEBUG", "0") == "1"

SUBSTAGE_FILE = Path(__file__).resolve().parent.parent / "configs" / "substages.yml"

def _load_substage_catalog(path: Path = SUBSTAGE_FILE) -> dict[str, list[str]]:
    """
    Parse the attached substages.yml, which is a simple indented taxonomy,
    not strict YAML. Returns:
      {
        "ACCUMULATION": [...],
        "MARKUP": [...],
        "DISTRIBUTION": [...],
        "MARKDOWN": [...],
      }
    """
    if DEBUG:
        print("SUBSTAGE_FILE:", path)
        print("SUBSTAGE_FILE exists:", path.exists())

    catalog = {
        "ACCUMULATION": [],
        "MARKUP": [],
        "DISTRIBUTION": [],
        "MARKDOWN": [],
    }
    if not path.exists():
        return catalog

    current = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue

        head = line.strip().upper()
        if head.startswith("ACCUMULATION"):
            current = "ACCUMULATION"
            continue
        if head.startswith("MARKUP"):
            current = "MARKUP"
            continue
        if head.startswith("DISTRIBUTION"):
            current = "DISTRIBUTION"
            continue
        if head.startswith("MARKDOWN"):
            current = "MARKDOWN"
            continue

        if current is not None:
            catalog[current].append(line.strip())

    return catalog

SUBSTAGE_CATALOG = _load_substage_catalog()


def _valid_substage_labels() -> set[str]:
    return {item for items in SUBSTAGE_CATALOG.values() for item in items}

VALID_SUBSTAGES = _valid_substage_labels()

def _safe_substage_label(label: str, fallback: str = "NEUTRAL_RANGE") -> str:
    if label in VALID_SUBSTAGES:
        return label
    return fallback


def compute_mean_reversion(df):
    price = df["close"].iloc[-1]

    vwap = df.get("VWAP", df.get("vwap_support", df["close"])).iloc[-1]
    rsi = df.get("RSI", df.get("RSI_14", 50)).iloc[-1]

    if price < 0.95 * vwap and rsi < 35:
        return 1
    elif price < vwap:
        return 0.5
    return 0

def compute_risk_reward(entry, target, stop):
    if entry and stop and target:
        risk = entry - stop
        reward = target - entry
        if risk > 0:
            return reward / risk
    return np.nan


def generate_exit_reason(row):
    reasons = []

    if row.get("RSI", 50) > 70:
        reasons.append("RSI_OVERBOUGHT")

    if row.get("MACD Cross") == "BEARISH":
        reasons.append("MACD_BEARISH")

    if row.get("Current Price") > row.get("VWAP"):
        reasons.append("EXTENDED_ABOVE_VWAP")

    return ",".join(reasons) if reasons else "HOLD"

def _classify_market_stage(df: pd.DataFrame) -> pd.Series:
    """
    Classify each row into a market stage using already-computed columns.
    Stages: Accumulation, Mark-Up, Distribution, Mark-Down, Neutral/Transition.
    Priority: Mark-Down > Distribution > Mark-Up > Accumulation > Neutral/Transition
    """
    n = len(df)
    stage = pd.Series(["Accumulation"] * n, index=df.index, dtype="object")

    def getcol(name, default, dtype=None):
        s = df[name] if name in df.columns else pd.Series([default] * n, index=df.index)
        if dtype == "bool":
            return s.fillna(False).astype(bool)
        if dtype == "float":
            return pd.to_numeric(s, errors="coerce").fillna(float(default)).astype(float)
        if dtype == "int":
            return pd.to_numeric(s, errors="coerce").fillna(int(default)).astype(int)
        if dtype == "str":
            return s.fillna(str(default)).astype(str)
        return s

    tight_range = getcol("tight_range", False, "bool")
    ema_up = getcol("EMA_uptrend", False, "bool")
    htf = getcol("HTF_Trend", "NEUTRAL", "str").str.upper()
    strong_trend = getcol("strong_trend", 0, "int")
    adx = getcol("ADX_14", 0.0, "float")
    macd_hist_slope = getcol("MACD_hist_slope", 0.0, "float")
    macd_hist = getcol("MACD_hist", 0.0, "float")

    mark_down = (htf == "DOWN") & (~ema_up) & (macd_hist < 0)
    mark_up = (htf == "UP") & ema_up & ((strong_trend == 1) | (adx >= 22)) & (macd_hist >= 0)
    distribution = tight_range & (htf == "UP") & (macd_hist_slope <= 0) & (macd_hist <= 0.05)
    accumulation = tight_range & (~ema_up) & (macd_hist >= -0.05)

    stage.loc[accumulation] = "Accumulation"
    stage.loc[mark_up] = "Mark-Up"
    stage.loc[distribution] = "Distribution"
    stage.loc[mark_down] = "Mark-Down"

    if DEBUG:
        print("[stage] counts:", stage.value_counts(dropna=False).to_dict())
        print(
            "[stage] masks:",
            {
                "accum": int(accumulation.sum()),
                "mark_up": int(mark_up.sum()),
                "dist": int(distribution.sum()),
                "mark_down": int(mark_down.sum()),
            },
        )

    return stage


def _safe_series(df: pd.DataFrame, col: str, default=np.nan, numeric: bool = True) -> pd.Series:
    s = df[col] if col in df.columns else pd.Series(default, index=df.index)
    if numeric:
        return pd.to_numeric(s, errors="coerce")
    return s.astype(str)

def _stage_key(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.upper().str.replace("-", "", regex=False).str.replace("/", "", regex=False).str.replace(" ", "", regex=False)




def compute_indicators(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    # === Core Trend & Momentum Indicators ===
    df["EMA_20"] = ta.ema(df["close"], length=20)
    df["EMA_21"] = ta.ema(df["close"], length=21)
    df["EMA_50"] = ta.ema(df["close"], length=50)
    df["EMA_200"] = ta.ema(df["close"], length=200)

    # ADX (safe)
    adx = ta.adx(df["high"], df["low"], df["close"], length=14)
    if adx is None or "ADX_14" not in getattr(adx, "columns", []):
        df["ADX_14"] = pd.Series([np.nan] * len(df), index=df.index)
    else:
        df["ADX_14"] = pd.to_numeric(adx["ADX_14"], errors="coerce")

    # MACD (safe)
    macd = ta.macd(df["close"])
    if macd is None:
        df["MACD"] = df["MACD_signal"] = df["MACD_hist"] = pd.Series([np.nan] * len(df), index=df.index)
    else:
        df["MACD"] = pd.to_numeric(macd.get("MACD_12_26_9"), errors="coerce")
        df["MACD_signal"] = pd.to_numeric(macd.get("MACDs_12_26_9"), errors="coerce")
        df["MACD_hist"] = pd.to_numeric(macd.get("MACDh_12_26_9"), errors="coerce")

    df["MACD_hist_slope"] = df["MACD_hist"].diff().rolling(3).mean()

    # MACD crossover (histogram crosses from negative -> positive)
    # Keep multiple column-name variants for backward compatibility across the pipeline.
    macd_x = (df["MACD_hist"] > 0) & (df["MACD_hist"].shift(1) < 0)
    df["MACD_crossover"] = macd_x
    df["MACD_Crossover"] = macd_x
    df["MACD_CROSSOVER"] = macd_x

    # RSI/OBV (safe)
    rsi = ta.rsi(df["close"], length=14)
    df["RSI_14"] = pd.to_numeric(rsi, errors="coerce")
    df["RSI_slope"] = df["RSI_14"].diff().rolling(3).mean()
    obv = ta.obv(df["close"], df["volume"])
    df["OBV"] = pd.to_numeric(obv, errors="coerce")

    # Bollinger Bands (safe)
    # === SAFE Bollinger Bands (version-proof) ===
    bb = ta.bbands(df["close"], length=20)

    if bb is None or getattr(bb, "empty", True):
        df["BB_upper"] = np.nan
        df["BB_middle"] = np.nan
        df["BB_lower"] = np.nan
    else:
        cols = list(bb.columns)

        def find_col(prefix):
            for c in cols:
                if str(c).startswith(prefix):
                    return c
            return None

        u = find_col("BBU_")
        m = find_col("BBM_")
        l = find_col("BBL_")

        if u and m and l:
            df["BB_upper"] = bb[u]
            df["BB_middle"] = bb[m]
            df["BB_lower"] = bb[l]
        else:
            # fallback manual
            mid = df["close"].rolling(20).mean()
            std = df["close"].rolling(20).std()
            df["BB_upper"] = mid + 2 * std
            df["BB_middle"] = mid
            df["BB_lower"] = mid - 2 * std

    # ATR (safe)
    atr = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["ATR_14"] = pd.to_numeric(atr, errors="coerce")

    # === Trend Confirmation ===
    df["EMA_21_slope"] = df["EMA_21"].diff().rolling(5).mean()
    df["EMA_uptrend"]  = (df["EMA_21"] > df["EMA_50"]) & (df["EMA_21_slope"] > 0)
    df["strong_trend"] = (df["ADX_14"] > 25).astype(int)

    df["above_EMA21"]  = (df["close"] > df["EMA_21"]).astype(int)
    df["above_EMA50"]  = (df["close"] > df["EMA_50"]).astype(int)
    df["above_EMA200"] = (df["close"] > df["EMA_200"]).astype(int)

    # === Behavioral Patterns ===
    df["green_candles"] = (df["close"] > df["open"]).astype(int).rolling(3).sum()
    df["red_candles"]   = (df["close"] < df["open"]).astype(int).rolling(3).sum()
    df["volume_trend"]  = df["volume"].diff().rolling(3).mean()

    # === Tight Range Detection ===
    # Compare recent intraday range variability vs a 50-day baseline
    intraday_range = (df["high"] - df["low"])
    df["range_std"]  = intraday_range.rolling(10, min_periods=10).std()
    base_std         = df["range_std"].rolling(50, min_periods=20).mean()
    df["tight_range"] = (df["range_std"] < base_std * 0.8).fillna(False)

    # === Darvas Box Breakout ===
    df = darvas_box_signal(df)

    # === Volume Surge Confirmation ===
    df["volume_avg_20"] = df["volume"].rolling(20, min_periods=1).mean()
    df["volume_surge"]  = (df["volume"] > 1.5 * df["volume_avg_20"]).fillna(False)
    df["VOL_SURGE_RATIO"] = (df["volume"] / df["volume_avg_20"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["VOL_TODAY"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    df["AVG_VOL_20D"] = pd.to_numeric(df["volume_avg_20"], errors="coerce").fillna(0.0)
    df["VOL_TREND_5D"] = pd.to_numeric(df["volume"].pct_change(5), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # === VWAP Support (proxy) ===
    # If you add true session VWAP later, wire it here; for now a 5-bar swing low proxy is used.
    df["vwap_support"] = df["low"].rolling(5, min_periods=1).min()

    # === Price Near Support Check (VWAP or BB Lower) ===
    df["near_support"] = (
        (df["close"] <= df["vwap_support"] * 1.02) |
        (df["close"] <= df["BB_lower"] * 1.02)
    ).fillna(False)

    # === Sector Rotation Strength (placeholder for future enhancement) ===
    df["sector_outperformance"] = 0.0
    # === Multi-timeframe Volume Profiles ===
    for win in (5, 20, 60):
        roll = df["volume"].rolling(win, min_periods=1)
        df[f"vol_mean_{win}"] = roll.mean()
        df[f"vol_z_{win}"] = (df["volume"] - df[f"vol_mean_{win}"]) / (roll.std() + 1e-9)

    df["vol_ratio_5_20"] = df["vol_mean_5"] / (df["vol_mean_20"] + 1e-9)
    df["vol_ratio_20_60"] = df["vol_mean_20"] / (df["vol_mean_60"] + 1e-9)

    # Symbol-only volatility regime (even if macro fetch fails)
    sym_ret = df["close"].pct_change()
    sym_vol_20 = sym_ret.rolling(20).std()
    sym_vol_250 = sym_ret.rolling(250).std()
    df["sym_vol_ratio"] = sym_vol_20 / (sym_vol_250 + 1e-9)

    # Fractal swing points & liquidity zones
    df = _compute_fractals(df)
    liq_sup, liq_res = _compute_liquidity_zones(df, lookback=60, bins=24)
    df["liq_support"] = liq_sup
    df["liq_resistance"] = liq_res


    def _sym_regime(x):
        if not np.isfinite(x):
            return 0
        if x >= 1.5:
            return 1
        if x <= 0.7:
            return -1
        return 0

    df["sym_vol_regime"] = df["sym_vol_ratio"].apply(_sym_regime)

    # === Macro overlay (VIX, QQQ, TLT, macro vol regimes) ===
    try:
        df = enrich_with_macro_features(df)
    except Exception as e:
        if DEBUG:
            print(f"[compute] macro feature enrichment failed: {e}")

    # 1) patterns
    df = apply_pattern_features(df)

    # 2) institutional inputs
    df["institutional_score"] = score_institutional_investor(df)
    df["volume_weight"] = np.minimum(
        df["volume"] / df["volume_avg_20"].replace(0, np.nan), 2.0
    ).fillna(0.0)

    # 3) trend labels FIRST
    df["HTF_Trend"] = np.where((df["EMA_21"] > df["EMA_50"]), "UP", "DOWN")
    df["ITF_Trend"] = np.where(ta.ema(df["close"], length=8) > ta.ema(df["close"], length=21), "UP", "DOWN")
    df["LTF_Trend"] = np.where(ta.ema(df["close"], length=5) > ta.ema(df["close"], length=13), "UP", "DOWN")

    # 4) market stage
    df["market_stage"] = _classify_market_stage(df)

    # 5) substage
    market_substage, substage_confidence = _classify_market_substage(df, df["market_stage"])
    df["market_substage"] = market_substage
    df["substage_confidence"] = pd.to_numeric(substage_confidence, errors="coerce").fillna(0.0)

    # Confidence calibration to avoid overconfident generic labels
    adx_norm = (pd.to_numeric(df.get("ADX_14"), errors="coerce").fillna(0.0) / 40.0).clip(0.0, 1.0)
    df["substage_confidence"] = np.clip(
        0.70 * df["substage_confidence"] + 0.30 * adx_norm,
        0.0,
        1.0,
    )
    generic_mask = df["market_substage"].isin([
        "BASE_FORMATION",
        "LOW_VOLATILITY_COMPRESSION",
        "CONTINUATION_DOWNTREND",
    ])
    df.loc[generic_mask, "substage_confidence"] = (
        pd.to_numeric(df.loc[generic_mask, "substage_confidence"], errors="coerce") * 0.85
    ).clip(0.0, 1.0)

    # 6) stage strength
    # 6) stage strength
    stage_key = (df["market_stage"].fillna("").str.lower().str.replace("-", "", regex=False))

    df["stage_strength_score"] = 0.0
    df.loc[stage_key.str.contains("markup"), "stage_strength_score"] = 0.80
    df.loc[stage_key.str.contains("accum"), "stage_strength_score"] = 0.70
    df.loc[stage_key.str.contains("distribution"), "stage_strength_score"] = 0.30
    df.loc[stage_key.str.contains("markdown"), "stage_strength_score"] = 0.15


    # 7) quant
    df = apply_quant_scores(df)

    # 8) nextgen features
    df = _add_nextgen_features(df)

    # 9) recommendation
    df["rule_recommendation"] = np.where(df["refined_buy_signal"], "BUY", "HOLD")


    if DEBUG:
        print("[compute] market_stage counts:", df["market_stage"].value_counts(dropna=False).to_dict())
        print("substage raw counts:", market_substage.value_counts(dropna=False).to_dict())
        print("substage confidence summary:", df["substage_confidence"].describe())
        print("final substage counts:", df["market_substage"].value_counts(dropna=False).to_dict())
        print("VALID_SUBSTAGES size:", len(VALID_SUBSTAGES))
        print("SUBSTAGE_FILE:", SUBSTAGE_FILE)
        print("SUBSTAGE_FILE exists:", SUBSTAGE_FILE.exists())
        print("SUBSTAGE_FILE:", SUBSTAGE_FILE)
        print("EXISTS:", SUBSTAGE_FILE.exists())
        print("VALID_SUBSTAGES:", len(VALID_SUBSTAGES))


        df = df.copy()
    # ===============================
    # DEBUG: Final stage/substage (LATEST ROW ONLY)
    # ===============================
    if DEBUG:
        try:
            print(f"[{symbol}] FINAL ROW:")
            print(df[["market_stage", "market_substage", "substage_confidence","EMA_21", "EMA_50","ADX_14","MACD_hist", "volume_surge" ]].tail(1))
        except Exception as e:
            print(f"[{symbol}] Debug print failed:", e)


    # 10) schema-friendly aliases
    df["ADX"] = pd.to_numeric(df.get("ADX_14"), errors="coerce")
    df["ATR14"] = pd.to_numeric(df.get("ATR_14"), errors="coerce")
    df["ATR14_PCT"] = pd.to_numeric(df.get("ATR14_PCT", df.get("ATR_14") / df.get("close", pd.Series(np.nan, index=df.index)).replace(0, np.nan)), errors="coerce")
    df["EMA Uptrend"] = df.get("EMA_uptrend", False)
    df["EMA21 Slope"] = pd.to_numeric(df.get("EMA_21_slope"), errors="coerce")
    df["MACD Cross"] = df.get("MACD_Crossover", df.get("MACD_crossover", False))
    df["MACD_SIGNAL"] = pd.to_numeric(df.get("MACD_signal"), errors="coerce")
    df["MACD_HIST"] = pd.to_numeric(df.get("MACD_hist"), errors="coerce")
    df["RSI"] = pd.to_numeric(df.get("RSI_14"), errors="coerce")
    df["VWAP Support"] = pd.to_numeric(df.get("vwap_support"), errors="coerce")
    df["Near Support"] = df.get("near_support", False)
    df["Volume Surge"] = df.get("volume_surge", False)

    return df




def _add_nextgen_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add stronger non-leaky features for model separation."""
    close = pd.to_numeric(df.get("close"), errors="coerce")
    high = pd.to_numeric(df.get("high"), errors="coerce")
    low = pd.to_numeric(df.get("low"), errors="coerce")
    volume = pd.to_numeric(df.get("volume"), errors="coerce")
    atr14 = pd.to_numeric(df.get("ATR_14", np.nan), errors="coerce")

    roll_20_high = high.rolling(20, min_periods=5).max()
    roll_20_low = low.rolling(20, min_periods=5).min()
    roll_60_high = high.rolling(60, min_periods=20).max()
    roll_252_high = high.rolling(252, min_periods=60).max()
    roll_252_low = low.rolling(252, min_periods=60).min()

    df["ATR14_PCT"] = atr14 / close.replace(0, np.nan)
    df["rolling_20_high"] = roll_20_high
    df["rolling_20_low"] = roll_20_low
    df["rolling_60_high"] = roll_60_high
    df["rolling_252_high"] = roll_252_high
    df["rolling_252_low"] = roll_252_low
    df["dist_from_20d_high"] = close / roll_20_high.replace(0, np.nan) - 1.0
    df["dist_from_52w_high"] = close / roll_252_high.replace(0, np.nan) - 1.0
    df["dist_from_52w_low"] = close / roll_252_low.replace(0, np.nan) - 1.0
    df["range_position_20d"] = (close - roll_20_low) / (roll_20_high - roll_20_low + 1e-9)
    df["range_position_60d"] = (close - low.rolling(60, min_periods=20).min()) / (roll_60_high - low.rolling(60, min_periods=20).min() + 1e-9)

    df["EMA20_SLOPE_PCT_5"] = df["EMA_20"] / df["EMA_20"].shift(5) - 1.0
    df["EMA50_SLOPE_PCT_10"] = df["EMA_50"] / df["EMA_50"].shift(10) - 1.0
    df["EMA21_SLOPE_PCT_5"] = df["EMA_21"] / df["EMA_21"].shift(5) - 1.0
    df["trend_persistence_10"] = (close > df["EMA_20"]).rolling(10, min_periods=5).mean()
    df["trend_persistence_20"] = (close > df["EMA_50"]).rolling(20, min_periods=10).mean()

    df["OBV_slope_5"] = df["OBV"] / df["OBV"].shift(5).replace(0, np.nan) - 1.0
    df["OBV_slope_20"] = df["OBV"] / df["OBV"].shift(20).replace(0, np.nan) - 1.0
    df["volume_pressure"] = ((close - low) - (high - close)) / (high - low + 1e-9)
    df["VOL_SURGE_RATIO"] = volume / volume.rolling(20, min_periods=5).mean().replace(0, np.nan)
    df["VOL_TREND_20"] = volume.rolling(5, min_periods=3).mean() / volume.rolling(20, min_periods=5).mean().replace(0, np.nan)

    df["VWAP_DISTANCE_PCT"] = close / df["vwap_support"].replace(0, np.nan) - 1.0
    df["breakout_strength"] = (close - roll_20_high.shift(1)) / (atr14 + 1e-9)
    df["pullback_depth_atr"] = (df["EMA_21"] - close) / (atr14 + 1e-9)
    df["compression_score"] = (high - low).rolling(10, min_periods=5).mean() / ((high - low).rolling(50, min_periods=20).mean() + 1e-9)
    df["close_to_liq_support_atr"] = (close - df["liq_support"]) / (atr14 + 1e-9)
    df["close_to_liq_resistance_atr"] = (df["liq_resistance"] - close) / (atr14 + 1e-9)

    ret1 = close.pct_change()
    df["TREND_EFFICIENCY_20"] = (close - close.shift(20)).abs() / (close.diff().abs().rolling(20, min_periods=10).sum() + 1e-9)
    df["up_day_ratio_20"] = (ret1 > 0).rolling(20, min_periods=10).mean()
    df["down_day_ratio_20"] = (ret1 < 0).rolling(20, min_periods=10).mean()
    if "QQQ_ret_20" in df.columns:
        df["REL_STRENGTH_20D_VS_QQQ"] = close.pct_change(20) - pd.to_numeric(df["QQQ_ret_20"], errors="coerce")
    else:
        df["REL_STRENGTH_20D_VS_QQQ"] = close.pct_change(20)

    df["PCT_FROM_DMA50"] = close / df["EMA_50"].replace(0, np.nan) - 1.0
    df["PCT_FROM_DMA200"] = close / df["EMA_200"].replace(0, np.nan) - 1.0
    df["QUANT_COMPOSITE_SCORE"] = (
        0.18 * (pd.to_numeric(df["RSI_14"], errors="coerce") / 100.0)
        + 0.18 * pd.to_numeric(df["trend_persistence_20"], errors="coerce").fillna(0.0)
        + 0.14 * (pd.to_numeric(df["VOL_SURGE_RATIO"], errors="coerce").clip(0, 3).fillna(0.0) / 3.0)
        + 0.16 * pd.to_numeric(df["TREND_EFFICIENCY_20"], errors="coerce").clip(0, 1).fillna(0.0)
        + 0.16 * ((pd.to_numeric(df["breakout_strength"], errors="coerce").clip(-2, 4).fillna(0.0) + 2.0) / 6.0)
        + 0.18 * pd.to_numeric(df["confidence_score"], errors="coerce").fillna(0.0)
    )

    fundamental_boost = pd.to_numeric(
        df["FUNDAMENTAL_BOOST"] if "FUNDAMENTAL_BOOST" in df.columns else pd.Series(0.0, index=df.index),
        errors="coerce"
    ).fillna(0.0)

    df["LONG_SCORE"] = df["QUANT_COMPOSITE_SCORE"] + 0.10 * fundamental_boost
    df["SHORT_SCORE"] = 1.0 - df["QUANT_COMPOSITE_SCORE"]

    num_cols = [
        "ATR14_PCT", "dist_from_20d_high", "dist_from_52w_high", "dist_from_52w_low", "range_position_20d",
        "range_position_60d", "EMA20_SLOPE_PCT_5", "EMA50_SLOPE_PCT_10", "EMA21_SLOPE_PCT_5", "trend_persistence_10",
        "trend_persistence_20", "OBV_slope_5", "OBV_slope_20", "volume_pressure", "VOL_SURGE_RATIO", "VOL_TREND_20",
        "VWAP_DISTANCE_PCT", "breakout_strength", "pullback_depth_atr", "compression_score",
        "close_to_liq_support_atr", "close_to_liq_resistance_atr", "TREND_EFFICIENCY_20", "up_day_ratio_20",
        "down_day_ratio_20", "REL_STRENGTH_20D_VS_QQQ", "PCT_FROM_DMA50", "PCT_FROM_DMA200",
        "QUANT_COMPOSITE_SCORE", "LONG_SCORE", "SHORT_SCORE",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return df


def analyze_symbol_all(symbol: str):
    try:
        df = fetch_data_cached(symbol, "10 Y", "1 day", force_refresh=True)
        df = compute_indicators(df, symbol=symbol)

        # Compute entry prices ONCE
        entry_prices = compute_entry_prices(df)

        vwap_support = float(df["vwap_support"].iloc[-1])

        result = {
            "Symbol": symbol,
            "VWAP Support": round(vwap_support, 2),
            "ADX": round(float(df["ADX_14"].iloc[-1]), 2),
            "Institutional Score": round(float(df["institutional_score"].iloc[-1]), 2),
            "Volume Weight": round(float(df["volume_weight"].iloc[-1]), 2),
            "Confidence Score": round(float(df["confidence_score"].iloc[-1]), 2),
            "Trend": str(df["HTF_Trend"].iloc[-1]),
            "Rule Recommendation": str(df["rule_recommendation"].iloc[-1]),
            "Darvas Breakout %": round(float(df["darvas_breakout_pct"].iloc[-1]), 2)
            if "darvas_breakout_pct" in df.columns else 0.0,
            "Darvas Signal": "✅" if int(df.get("darvas_signal", pd.Series([0])).iloc[-1]) == 1 else "❌",
            "Market Stage": str(df.get("market_stage", pd.Series(["Neutral/Transition"])).iloc[-1]),
            "Market Sub-Stage": str(df.get("market_substage", pd.Series(["NEUTRAL_RANGE"])).iloc[-1]),

            # ✅ SINGLE SOURCE OF TRUTH
            **entry_prices,
        }

        return result

    except Exception as e:
        print(f"⚠️ Error analyzing {symbol}: {e}")
        return None

def detect_smc_accumulation_breakout(df: pd.DataFrame) -> bool:
    recent = df.tail(20)
    if recent.empty:
        return False
    tight_range = recent["high"].max() - recent["low"].min() < 0.05 * recent["close"].iloc[-1]
    breakout = df.iloc[-1]["close"] > recent["high"].max()
    volume_spike = df.iloc[-1]["volume"] > 1.5 * recent["volume"].mean()
    return bool(tight_range and breakout and volume_spike)



def _add_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(df["close"], errors="coerce")
    open_ = pd.to_numeric(df["open"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")

    ema20 = pd.to_numeric(df.get("EMA_20"), errors="coerce")
    ema50 = pd.to_numeric(df.get("EMA_50"), errors="coerce")
    ema200 = pd.to_numeric(df.get("EMA_200"), errors="coerce")
    rsi = pd.to_numeric(df.get("RSI_14"), errors="coerce")
    macd = pd.to_numeric(df.get("MACD"), errors="coerce")
    macd_signal = pd.to_numeric(df.get("MACD_signal"), errors="coerce")
    adx = pd.to_numeric(df.get("ADX_14"), errors="coerce")
    vwap = pd.to_numeric(df.get("vwap_support"), errors="coerce")

    volume_avg_20 = volume.rolling(20, min_periods=5).mean()
    vol_ratio = volume / volume_avg_20.replace(0, np.nan)

    rolling_20_high = high.rolling(20, min_periods=5).max()
    rolling_20_low = low.rolling(20, min_periods=5).min()

    # ===============================
    # ✅ BUILD FEATURES IN DICT
    # ===============================
    new_cols = {}

    # Trend
    new_cols["HIGHER_HIGH_HIGHER_LOW"] = ((high > high.shift(1)) & (low > low.shift(1))).astype(int)
    new_cols["LOWER_HIGH_LOWER_LOW"] = ((high < high.shift(1)) & (low < low.shift(1))).astype(int)

    # EMA stacking
    new_cols["EMA_STACKED_BULLISH"] = ((ema20 > ema50) & (ema50 > ema200)).astype(int)
    new_cols["EMA_STACKED_BEARISH"] = ((ema20 < ema50) & (ema50 < ema200)).astype(int)

    # Range
    new_cols["RANGE_BOUND"] = ((rolling_20_high - rolling_20_low) / close < 0.08).astype(int)
    new_cols["TIGHT_RANGE_CONSOLIDATION"] = ((rolling_20_high - rolling_20_low) / close < 0.04).astype(int)

    # Breakouts
    new_cols["RESISTANCE_BREAKOUT"] = (close > rolling_20_high.shift(1)).astype(int)
    new_cols["SUPPORT_BREAKDOWN"] = (close < rolling_20_low.shift(1)).astype(int)

    # Volume
    new_cols["HIGH_VOLUME_BREAKOUT"] = ((close > rolling_20_high.shift(1)) & (vol_ratio > 1.5)).astype(int)
    new_cols["VOLUME_SPIKE"] = (vol_ratio > 1.5).astype(int)
    new_cols["VOLUME_DRY_UP"] = (vol_ratio < 0.7).astype(int)

    # Candles
    body = (close - open_).abs()
    range_ = (high - low).replace(0, np.nan)
    lower_wick = (np.minimum(open_, close) - low)
    upper_wick = (high - np.maximum(open_, close))

    new_cols["HAMMER"] = ((lower_wick > 2 * body) & (upper_wick < body)).astype(int)
    new_cols["DOJI"] = ((body / range_) < 0.1).fillna(False).astype(int)

    # VWAP
    new_cols["VWAP_RECLAIM"] = ((close > vwap) & (close.shift(1) <= vwap.shift(1))).astype(int)
    new_cols["VWAP_REJECTION"] = ((high > vwap) & (close < vwap)).astype(int)

    # Gaps
    new_cols["GAP_UP"] = (low > high.shift(1)).astype(int)
    new_cols["GAP_DOWN"] = (high < low.shift(1)).astype(int)

    # ===============================
    # ✅ SINGLE CONCAT (CRITICAL)
    # ===============================
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df


# --- Candle-based entry (last ~6 weeks) ---
def entry_from_recent_candles(df, weeks: int = 6) -> float:
    """
    Heuristic entry from recent candles:
      • Start with confluence of EMA21/EMA50 zone
      • If trend strong (ADX>=25 & EMA_uptrend), prefer pullback to EMA21
      • Else prefer swing-low zone (min low in last N days)
      • Require reversal hint: bullish engulfing or hammer (simple rules)
      • Add small ATR buffer below computed anchor to avoid premature fills
    """
    if df is None or df.empty:
        return float("nan")

    look = min(len(df), weeks * 5)  # ~5 trading days per week
    recent = df.tail(look).copy()

    ema21  = float(recent["EMA_21"].iloc[-1])
    ema50  = float(recent["EMA_50"].iloc[-1])
    adx    = float(recent["ADX_14"].iloc[-1])
    atr    = float(recent["ATR_14"].iloc[-1])
    swing_low = float(recent["low"].min())

    # Reversal hints (very lightweight)
    last = recent.iloc[-1]
    prev = recent.iloc[-2] if len(recent) > 1 else last
    bullish_engulf = (last["close"] > last["open"]) and (prev["close"] < prev["open"]) and (last["close"] >= prev["open"]) and (last["open"] <= prev["close"])
    lower_shadow   = (last["low"] < last["open"]) and (last["low"] < last["close"]) and ((min(last["open"], last["close"]) - last["low"]) > (last["high"] - max(last["open"], last["close"])))
    hammer_like    = lower_shadow and (last["close"] > last["open"])

    ema_confluence = 0.5 * (ema21 + ema50)
    ema_up = bool(recent["EMA_uptrend"].iloc[-1])

    if ema_up and adx >= 25:
        anchor = 0.6 * ema21 + 0.4 * ema_confluence
    else:
        anchor = 0.5 * swing_low + 0.5 * ema_confluence

    # If we saw a bullish reversal, bias entry slightly higher (to get filled in strength)
    if bullish_engulf or hammer_like:
        anchor = 0.7 * anchor + 0.3 * ema21

    # Add a small safety buffer (enter slightly above the anchor)
    entry = anchor + 0.15 * atr
    return round(entry, 2)


def _bullish_engulfing_recent(df):
    if len(df) < 2:
        return False
    prev, curr = df.iloc[-2], df.iloc[-1]
    return (
        (prev["close"] < prev["open"]) and
        (curr["close"] > curr["open"]) and
        (curr["close"] >= prev["open"]) and
        (curr["open"] <= prev["close"])
    )

def _hammer_like_recent(df):
    if len(df) < 1:
        return False
    last = df.iloc[-1]
    c = _as_scalar(last.get("close", last.get("Close", np.nan)), default=np.nan)
    o = _as_scalar(last.get("open", last.get("Open", np.nan)), default=np.nan)
    h = _as_scalar(last.get("high", last.get("High", np.nan)), default=np.nan)
    lo = _as_scalar(last.get("low", last.get("Low", np.nan)), default=np.nan)
    body = abs(c - o)
    lower_wick = (min(o, c) - lo)
    upper_wick = (h - max(o, c))
    return (lower_wick > 2 * body) and (upper_wick < body)


# ================================
# Candle-entry (BUY) price helpers
# ================================



def _safe_last(df, col, default=np.nan):
    try:
        return float(df[col].iloc[-1])
    except Exception:
        return float(default)

def _nanfilter(values):
    out = []
    for v in values:
        try:
            if v is None:
                continue
            v = float(v)
            if np.isfinite(v):
                out.append(v)
        except Exception:
            pass
    return out

def _iqr_filter(vals):
    """Remove extreme outliers via Tukey IQR. Keeps shape small & robust."""
    if len(vals) < 4:
        return vals
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    return [v for v in vals if lo <= v <= hi]

def _weeks_to_days(weeks, max_len):
    return int(max(5, min(max_len, int(weeks) * 5)))  # ~5 trading days per week

def _buffer_by_weeks(weeks):
    """
    Limit price buffer below the composite support; shorter windows want a bigger buffer
    to get filled quickly; longer windows can be more patient.
    """
    mapping = {
        2:  0.20,
        4:  0.15,
        6:  0.12,
        8:  0.10,
        12: 0.08,
        18: 0.06,
        30: 0.05,
    }
    return float(mapping.get(int(weeks), 0.10))

def _compute_liquidity_zones(df: pd.DataFrame, lookback: int = 60, bins: int = 24):
    """
    Approximate volume-based liquidity zones using a volume-weighted price histogram.
    Returns (support_zone_price, resistance_zone_price).
    """
    recent = df.tail(lookback)
    if recent.empty or "volume" not in recent.columns or "close" not in recent.columns:
        return np.nan, np.nan

    typical_price = (recent["high"] + recent["low"] + recent["close"]) / 3.0
    vol = recent["volume"]

    lo = float(typical_price.min())
    hi = float(typical_price.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return np.nan, np.nan

    hist, edges = np.histogram(typical_price, bins=bins, range=(lo, hi), weights=vol)
    if hist.sum() <= 0:
        return np.nan, np.nan

    # take top 3 volume nodes
    idxs = hist.argsort()[::-1][:3]
    centers = 0.5 * (edges[idxs] + edges[idxs + 1])

    cur = recent["close"].iloc[-1]
    below = [c for c in centers if c <= cur]
    above = [c for c in centers if c >= cur]

    support = max(below) if below else np.nan
    resistance = min(above) if above else np.nan
    return float(support), float(resistance)


def _compute_fractals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bill Williams-style fractal highs/lows to identify swing points.
    Populates: fract_high, fract_low, fract_last_low.
    """
    n = len(df)
    if n < 5:
        df["fract_high"] = 0
        df["fract_low"] = 0
        df["fract_last_low"] = np.nan
        return df

    high_fractal = np.zeros(n, dtype=int)
    low_fractal = np.zeros(n, dtype=int)

    for i in range(2, n - 2):
        hwin = df["high"].iloc[i-2:i+3]
        lwin = df["low"].iloc[i-2:i+3]
        if hwin.iloc[2] == hwin.max():
            high_fractal[i] = 1
        if lwin.iloc[2] == lwin.min():
            low_fractal[i] = 1

    df["fract_high"] = high_fractal
    df["fract_low"] = low_fractal

    # last swing low (support)
    idxs_low = np.where(low_fractal == 1)[0]
    last_low_price = np.nan
    if len(idxs_low):
        last_low_idx = idxs_low[-1]
        last_low_price = float(df["low"].iloc[last_low_idx])

    df["fract_last_low"] = last_low_price
    return df


def candle_entry_from_weeks(df: pd.DataFrame, weeks: int = 6) -> float:
    """
    Robust limit BUY from last N weeks of daily candles.
    Key hardening for short windows (2w/4w):
      • enforce min lookback of 20 bars so ATR/BB exist
      • ffill indicators inside the slice
      • safe fallbacks if a component is missing
    """
    try:
        if df is None or len(df) == 0:
            return float("nan")

        # --- Hardened lookback (2w/4w need at least ~20 bars for ATR_14 etc.)
        min_bars = 20
        look = max(min_bars, _weeks_to_days(weeks, len(df)))
        recent = df.tail(look).copy()

        # Forward-fill a few indicators inside the slice (helps right after gaps)
        for col in ["EMA_21", "BB_lower", "ATR_14", "vwap_support", "darvas_low"]:
            if col in recent.columns:
                recent[col] = pd.to_numeric(recent[col], errors="coerce").ffill()

        # Required bits (all safe)
        last_close = _safe_last(recent, "close")
        atr        = _safe_last(recent, "ATR_14", default=np.nan)
        adx        = _safe_last(recent, "ADX_14", default=np.nan)
        ema_up     = bool(recent.get("EMA_uptrend", pd.Series([False])).iloc[-1]) if "EMA_uptrend" in recent.columns else False

        ema21      = _safe_last(recent, "EMA_21")
        vwap_sup   = _safe_last(recent, "vwap_support")
        darvas_lo  = _safe_last(recent, "darvas_low")
        bb_lower   = _safe_last(recent, "BB_lower")
        swing_low  = float(recent["low"].min()) if "low" in recent.columns else np.nan

        # Candidate supports
        candidates = _nanfilter([ema21, vwap_sup, darvas_lo, bb_lower, swing_low])
        if not candidates:
            return float("nan")

        # Light outlier removal
        candidates = _iqr_filter(candidates)

        darvas_high = _safe_last(recent, "darvas_box_high", default=np.nan)
        darvas_sig = int(_safe_last(recent, "darvas_signal", default=0))

        if darvas_sig == 1 and np.isfinite(darvas_high):
            # Retest buy: slightly above box high OR at vwap/ema21 if those are higher
            retest_anchor = max(darvas_high * 1.002, min(ema21, vwap_sup))
            # Use smaller buffer so you don’t miss the fill on a retest
            entry = retest_anchor - 0.10 * atr if np.isfinite(atr) else retest_anchor * 0.995
            entry = min(entry, last_close * 0.999)
            return round(entry, 2)

        # Trend regime weighting
        strong_trend = (adx >= 25) and ema_up
        if strong_trend:
            weights = {"ema21": 0.35, "vwap": 0.30, "darv": 0.15, "bb": 0.10, "swing": 0.10}
        else:
            weights = {"ema21": 0.20, "vwap": 0.30, "darv": 0.15, "bb": 0.20, "swing": 0.15}

        # Additional anchors: fractal last low & liquidity support
        fract_low = float(recent.get("fract_last_low", pd.Series([np.nan])).iloc[-1]) if "fract_last_low" in recent.columns else np.nan
        liq_support = float(recent.get("liq_support", pd.Series([np.nan])).iloc[-1]) if "liq_support" in recent.columns else np.nan

        # Extend weights to include these if present
        if "fract" not in weights:
            # keep overall sum ~1; slight rebalancing
            weights.update({"fract": 0.10, "liq": 0.10})
            # rescale old ones a bit
            for k in ("ema21", "vwap", "darv", "bb", "swing"):
                weights[k] *= 0.8  # shrink a bit to make room

        vals, wts = [], []

        def add(v, key):
            if np.isfinite(v):
                vals.append(float(v))
                wts.append(weights[key])

        add(ema21, "ema21")
        add(vwap_sup, "vwap")
        add(darvas_lo, "darv")
        add(bb_lower, "bb")
        add(swing_low, "swing")
        add(fract_low, "fract")
        add(liq_support, "liq")

        if not vals:
            return float("nan")

        base = float(np.average(vals, weights=wts))
# 61.8% Fib cap (don’t chase)
        high_n = float(recent["high"].max()) if "high" in recent.columns else np.nan
        low_n  = float(recent["low"].min())  if "low"  in recent.columns else np.nan
        if np.isfinite(high_n) and np.isfinite(low_n) and high_n > low_n:
            fib_61 = low_n + 0.618 * (high_n - low_n)
            base = min(base, fib_61)

        # If near support, bias a touch higher to ensure fill
        near_sup = bool(recent.get("near_support", pd.Series([False])).iloc[-1]) if "near_support" in recent.columns else False
        if near_sup and np.isfinite(vwap_sup):
            base = (base * 0.6) + (vwap_sup * 0.4)

        # ATR buffer tuned by weeks; fallback to a tiny % if ATR NaN
        buf_k = _buffer_by_weeks(weeks)
        if np.isfinite(atr) and atr > 0:
            # Volatility regime conditioning: deeper discount in high vol, shallower in low vol
            sym_reg = float(
                recent.get("sym_vol_regime", pd.Series([0])).iloc[-1]
            ) if "sym_vol_regime" in recent.columns else 0
            if sym_reg == 1:  # high vol
                buf_k *= 1.2
            elif sym_reg == -1:  # low vol
                buf_k *= 0.8

            entry = base - buf_k * atr
        else:
            entry = base * 0.99  # ~1% cushion if ATR missing

        # A buy-limit should not be above last close
        if np.isfinite(last_close):
            entry = min(entry, last_close * 0.999)

        # --- If the computed entry somehow still goes NaN for short windows,
        #     use a conservative fallback based on the last ~20 bars.
        if (not np.isfinite(entry)) or entry <= 0:
            if int(weeks) in (2, 4):
                # ultra-robust short-window fallback
                lo = float(recent["low"].min()) if "low" in recent.columns else np.nan
                ema21_f = _safe_last(recent, "EMA_21")
                vwap_f  = _safe_last(recent, "vwap_support")
                bb_f    = _safe_last(recent, "BB_lower")
                # conservative anchor: the lowest of common supports + tiny ATR cushion
                anchor_f = np.nanmin([ema21_f, vwap_f, bb_f, lo])
                if np.isfinite(anchor_f):
                    if np.isfinite(atr) and atr > 0:
                        entry = anchor_f + 0.05 * atr   # small add so we’re above absolute lows
                    else:
                        entry = anchor_f * 1.01         # ~1% over the anchor if ATR missing
            # still bad? give up to NaN and let caller print None


        return round(entry, 2) if np.isfinite(entry) and entry > 0 else float("nan")
    except Exception:
        return float("nan")




def candle_entries_multi(df: pd.DataFrame, weeks_list=(2, 4, 6, 8, 12, 18, 30)):
    """
    Vector convenience: compute entry prices for many windows at once.
    Returns: {weeks: price}
    """
    out = {}
    for w in weeks_list:
        try:
            out[int(w)] = candle_entry_from_weeks(df, int(w))
        except Exception:
            out[int(w)] = float("nan")
    return out



def _classify_market_substage(
    df: pd.DataFrame,
    market_stage: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    Full substage classifier aligned to substages.yml.
    Uses vectorized rules and assigns exactly one substage per row.
    Returns:
      (market_substage, substage_confidence)
    """
    n = len(df)

    close = _safe_series(df, "close")
    open_ = _safe_series(df, "open")
    high = _safe_series(df, "high")
    low = _safe_series(df, "low")
    volume = _safe_series(df, "volume", default=0.0)
    adx = _safe_series(df, "ADX_14", default=0.0)
    rsi = _safe_series(df, "RSI_14", default=50.0)
    ema20 = _safe_series(df, "EMA_20")
    ema21 = _safe_series(df, "EMA_21")
    ema50 = _safe_series(df, "EMA_50")
    ema200 = _safe_series(df, "EMA_200")
    macd_hist = _safe_series(df, "MACD_hist", default=0.0)
    macd_hist_slope = _safe_series(df, "MACD_hist_slope", default=0.0)
    vwap_support = _safe_series(df, "vwap_support")
    volume_avg_20 = _safe_series(df, "volume_avg_20", default=np.nan)
    tight_range = _safe_series(df, "tight_range", default=0.0).fillna(0).astype(bool)
    darvas_signal = _safe_series(df, "darvas_signal", default=0.0).fillna(0).astype(int)
    near_support = _safe_series(df, "near_support", default=0.0).fillna(0).astype(bool)

    stage = _stage_key(market_stage)

    body = (close - open_).abs()
    upper_wick = (high - pd.concat([open_, close], axis=1).max(axis=1)).clip(lower=0)
    lower_wick = (pd.concat([open_, close], axis=1).min(axis=1) - low).clip(lower=0)

    vol_ratio = (volume / volume_avg_20.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ret_5 = close.pct_change(5)
    ret_10 = close.pct_change(10)
    ret_20 = close.pct_change(20)

    rolling_20_high = high.rolling(20, min_periods=5).max()
    rolling_20_low = low.rolling(20, min_periods=5).min()
    rolling_60_high = high.rolling(60, min_periods=20).max()
    rolling_60_low = low.rolling(60, min_periods=20).min()

    stacked_bull = (ema20 > ema50) & (ema50 > ema200) & (close >= ema20)
    stacked_bear = (ema20 < ema50) & (ema50 < ema200) & (close <= ema20)
    breakout = (close > rolling_20_high.shift(1)) | (darvas_signal == 1)
    breakdown = close < rolling_20_low.shift(1)

    near_20d_low = ((close - rolling_20_low) / close.replace(0, np.nan)).abs().fillna(1.0) <= 0.03
    near_20d_high = ((rolling_20_high - close) / close.replace(0, np.nan)).abs().fillna(1.0) <= 0.03
    near_60d_low = ((close - rolling_60_low) / close.replace(0, np.nan)).abs().fillna(1.0) <= 0.05
    near_60d_high = ((rolling_60_high - close) / close.replace(0, np.nan)).abs().fillna(1.0) <= 0.05

    hammer_like = (lower_wick > 2 * body) & (upper_wick <= body)
    shooting_star = (upper_wick > 2 * body) & (lower_wick <= body)

    score = pd.DataFrame(0.0, index=df.index, columns=sorted(VALID_SUBSTAGES))

    def add_score(mask: pd.Series, label: str, weight: float) -> None:
        label = _safe_substage_label(label)
        if label not in score.columns:
            return
        score.loc[mask.fillna(False), label] += float(weight)

    # -------------------------
    # ACCUMULATION
    # -------------------------
    acc = stage.eq("ACCUMULATION")

    capitulation = acc & (ret_5 <= -0.12) & (vol_ratio >= 2.0)
    selling_climax = acc & (ret_5 <= -0.08) & (vol_ratio >= 1.8) & near_20d_low
    automatic_rally = acc & (ret_5 >= 0.06) & (ret_10 > 0) & (close > ema20)
    secondary_test_acc = acc & near_20d_low & (vol_ratio <= 1.1) & (ret_5.between(-0.03, 0.03))
    spring = acc & (low < rolling_20_low.shift(1)) & (close > rolling_20_low.shift(1)) & (lower_wick > body)
    test_of_spring = acc & near_20d_low & (vol_ratio < 1.0) & (close > open_) & (ret_5 > -0.02)
    low_vol_comp = acc & tight_range & (vol_ratio < 0.9)
    range_acc = acc & tight_range & close.between(rolling_20_low * 1.01, rolling_20_high * 0.99)
    accumulation_breakout = acc & breakout & (vol_ratio >= 1.3) & (close > ema20)

    add_score(acc, "BASE_FORMATION", 1.0)
    add_score(range_acc, "RANGE_BOUND_ACCUMULATION", 3.0)
    add_score(low_vol_comp, "LOW_VOLATILITY_COMPRESSION", 4.0)
    add_score(secondary_test_acc, "SECONDARY_TEST (ST_ACC)", 5.0)
    add_score(automatic_rally, "AUTOMATIC_RALLY (AR)", 5.0)
    add_score(selling_climax, "SELLING_CLIMAX (SC)", 7.0)
    add_score(capitulation, "CAPITULATION", 8.0)
    add_score(spring, "SPRING (SHAKEOUT)", 8.0)
    add_score(test_of_spring, "TEST_OF_SPRING", 7.0)
    add_score(accumulation_breakout, "ACCUMULATION_BREAKOUT", 9.0)

    # -------------------------
    # MARKUP
    # -------------------------
    mu = stage.eq("MARKUP")

    early_trend = mu & (ema21 > ema50) & (ema21.shift(5) <= ema50.shift(5))
    breakout_confirmation = mu & breakout & (vol_ratio >= 1.2)
    bull_stack = mu & stacked_bull
    momentum_expansion = mu & (adx >= 25) & (macd_hist > 0) & (macd_hist_slope > 0)
    trend_cont = mu & stacked_bull & (ret_20 > 0) & (vol_ratio.between(0.9, 1.5))
    pullback_support = mu & stacked_bull & near_support & (ret_5 < 0) & (close >= ema50)
    vwap_reclaim = mu & (close > vwap_support) & (close.shift(1) <= vwap_support.shift(1))
    hv_breakout = mu & breakout & (vol_ratio >= 1.8)
    accel_phase = mu & (ret_10 >= 0.10) & (adx >= 30)
    climax_run = mu & (ret_5 >= 0.12) & (rsi >= 78) & (vol_ratio >= 2.0)

    add_score(mu, "TREND_CONTINUATION", 1.0)
    add_score(early_trend, "EARLY_TREND_INITIATION", 4.0)
    add_score(breakout_confirmation, "BREAKOUT_CONFIRMATION", 6.0)
    add_score(bull_stack, "EMA_STACK_FORMATION (BULL_STACKED)", 5.0)
    add_score(momentum_expansion, "MOMENTUM_EXPANSION", 7.0)
    add_score(trend_cont, "TREND_CONTINUATION", 5.0)
    add_score(pullback_support, "PULLBACK_TO_SUPPORT (SMART_MONEY_ENTRY)", 8.0)
    add_score(vwap_reclaim, "VWAP_RECLAIM", 7.0)
    add_score(hv_breakout, "HIGH_VOLUME_BREAKOUT (CTA_BREAKOUT)", 9.0)
    add_score(accel_phase, "ACCELERATION_PHASE (PARABOLIC_MOVE)", 8.0)
    add_score(climax_run, "CLIMAX_RUN (BUYING_FRENZY)", 9.0)

    # -------------------------
    # DISTRIBUTION
    # -------------------------
    dist = stage.eq("DISTRIBUTION")

    buying_climax = dist & (ret_5 >= 0.10) & (rsi >= 75) & (vol_ratio >= 1.8)
    upthrust = dist & (high > rolling_20_high.shift(1)) & (close < rolling_20_high.shift(1)) & shooting_star
    utad = dist & upthrust & near_60d_high & (vol_ratio >= 1.4)
    secondary_test_dist = dist & near_20d_high & (vol_ratio <= 1.1) & (ret_5.between(-0.03, 0.03))
    lower_high = dist & (high < high.shift(10)) & (close < rolling_20_high)
    failed_breakout = dist & breakout.shift(1).fillna(False) & (close < rolling_20_high.shift(1))
    distribution_breakdown = dist & breakdown & (vol_ratio >= 1.2)

    add_score(dist, "RANGE_BOUND_DISTRIBUTION", 1.0)
    add_score(secondary_test_dist, "SECONDARY_TEST (ST_DIST)", 5.0)
    add_score(buying_climax, "BUYING_CLIMAX (BC)", 8.0)
    add_score(upthrust, "UPTHRUST (UT)", 7.0)
    add_score(utad, "UPTHRUST_AFTER_DISTRIBUTION (UTAD)", 9.0)
    add_score(lower_high, "LOWER_HIGH_FORMATION", 6.0)
    add_score(failed_breakout, "FAILED_BREAKOUT", 8.0)
    add_score(distribution_breakdown, "DISTRIBUTION_BREAKDOWN", 9.0)

    # -------------------------
    # MARKDOWN
    # -------------------------
    md = stage.eq("MARKDOWN")

    reversal_init = md & (ema21 < ema50) & (ema21.shift(5) >= ema50.shift(5))
    bear_stack = md & stacked_bear
    momentum_selloff = md & (adx >= 25) & (macd_hist < 0) & (macd_hist_slope < 0)
    dead_cat = md & (ret_5 >= 0.04) & (ret_20 < 0) & (close < ema50)
    cont_down = md & stacked_bear & (ret_20 < 0)
    lower_low_acc = md & (low < rolling_20_low.shift(1)) & (vol_ratio >= 1.5)
    panic_sell = md & (ret_5 <= -0.12) & (vol_ratio >= 2.0)
    exhaustion_bottom = md & near_20d_low & hammer_like & (vol_ratio >= 1.2)

    add_score(md, "CONTINUATION_DOWNTREND", 1.0)
    add_score(reversal_init, "TREND_REVERSAL_INITIATION (BEARISH_BREAKDOWN)", 6.0)
    add_score(bear_stack, "EMA_STACK_FORMATION (BEAR_STACKED)", 5.0)
    add_score(momentum_selloff, "MOMENTUM_SELLOFF", 8.0)
    add_score(dead_cat, "DEAD_CAT_BOUNCE", 6.0)
    add_score(cont_down, "CONTINUATION_DOWNTREND", 5.0)
    add_score(lower_low_acc, "LOWER_LOW_ACCELERATION", 7.0)
    add_score(panic_sell, "PANIC_SELLING", 9.0)
    add_score(exhaustion_bottom, "EXHAUSTION_BOTTOMING", 7.0)

    # if score is None or len(score) == 0:
    #     best_label = "NEUTRAL_RANGE"
    #     best_score = 0.0
    # else:
    #     best_label = score.idxmax()
    #     best_score = score.max()

    best_label = score.idxmax(axis=1)
    best_score = score.max(axis=1)
    score_sum = score.sum(axis=1)

    fallback = pd.Series("NEUTRAL_RANGE", index=df.index, dtype="object")
    out = best_label.where(best_score > 0, fallback)
    out = out.apply(_safe_substage_label)

    substage_confidence = (best_score / score_sum.replace(0, np.nan)).fillna(0.0).clip(0.0, 1.0)

    return out, substage_confidence


# ============================================================================
# Pattern framework (structural + tactical)
# ============================================================================
import math
from typing import Dict, Tuple

def _rolling_local_minima(series: pd.Series, window: int = 5) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s[(s == s.rolling(window, center=True, min_periods=window).min())]

def _rolling_local_maxima(series: pd.Series, window: int = 5) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s[(s == s.rolling(window, center=True, min_periods=window).max())]

def _near(a: float, b: float, tol: float) -> bool:
    if not (np.isfinite(a) and np.isfinite(b)):
        return False
    if b == 0:
        return False
    return abs(a-b)/abs(b) <= tol

def _triple_bottom(df: pd.DataFrame, lookback: int = 180, tol: float = 0.035) -> bool:
    if df is None or df.empty or "close" not in df.columns:
        return False
    d = df.tail(lookback).copy()
    lows = _rolling_local_minima(d["close"], window=7)
    if len(lows) < 3:
        return False
    # Take last ~6 minima, look for 3 clustered
    vals = lows.tail(10).values
    vals = [float(x) for x in vals if np.isfinite(x)]
    if len(vals) < 3:
        return False
    # pick three lowest in recent minima
    vals_sorted = sorted(vals)[:3]
    if len(vals_sorted) < 3:
        return False
    m = sum(vals_sorted)/3.0
    return all(_near(v, m, tol) for v in vals_sorted)

def _double_bottom(df: pd.DataFrame, lookback: int = 120, tol: float = 0.035) -> bool:
    if df is None or df.empty or "close" not in df.columns:
        return False
    d = df.tail(lookback).copy()
    lows = _rolling_local_minima(d["close"], window=7)
    if len(lows) < 2:
        return False
    vals = lows.tail(6).values
    vals = [float(x) for x in vals if np.isfinite(x)]
    if len(vals) < 2:
        return False
    a, b = sorted(vals)[:2]
    return _near(a, b, tol)

def _rounding_bottom(df: pd.DataFrame, lookback: int = 200) -> bool:
    if df is None or df.empty or "close" not in df.columns:
        return False
    d = df.tail(lookback).copy()
    c = pd.to_numeric(d["close"], errors="coerce")
    if c.isna().all():
        return False
    # crude: low in first half, then higher lows in second half + positive slope in last 30
    mid = len(c)//2
    first_min_idx = int(c.iloc[:mid].idxmin())
    second_min = float(c.iloc[mid:].min())
    first_min = float(c.min())
    if not (np.isfinite(first_min) and np.isfinite(second_min)):
        return False
    cond1 = second_min > first_min * 1.02
    # slope
    tail = c.tail(30).dropna()
    if len(tail) < 10:
        return False
    x = np.arange(len(tail))
    slope = np.polyfit(x, tail.values, 1)[0]
    return bool(cond1 and slope > 0)

def _bull_flag(df: pd.DataFrame, lookback: int = 90) -> bool:
    if df is None or df.empty or "close" not in df.columns:
        return False
    d = df.tail(lookback).copy()
    c = pd.to_numeric(d["close"], errors="coerce")
    if len(c.dropna()) < 40:
        return False
    # strong impulse then gentle pullback: last 20 within 5% of 20-high and above 50d ma
    high20 = float(c.tail(20).max())
    close = float(c.iloc[-1])
    ma50 = float(c.rolling(50).mean().iloc[-1])
    if not all(np.isfinite(x) for x in [high20, close, ma50]):
        return False
    return (close >= high20*0.95) and (close >= ma50)

def _cup_handle(df: pd.DataFrame, lookback: int = 260) -> bool:
    if df is None or df.empty or "close" not in df.columns:
        return False
    d = df.tail(lookback).copy()
    c = pd.to_numeric(d["close"], errors="coerce").dropna()
    if len(c) < 120:
        return False
    # crude: left peak near top quartile, middle dip, right peak near left peak, then small handle dip last 20
    left_peak = float(c.iloc[:len(c)//3].max())
    mid_low = float(c.iloc[len(c)//3:2*len(c)//3].min())
    right_peak = float(c.iloc[2*len(c)//3:].max())
    if not all(np.isfinite(x) for x in [left_peak, mid_low, right_peak]):
        return False
    if mid_low > left_peak*0.92:
        return False
    if right_peak < left_peak*0.95:
        return False
    handle_low = float(c.tail(20).min())
    return handle_low > right_peak*0.90

def compute_patterns_full(df: pd.DataFrame, common: dict | None = None) -> Dict[str, bool]:
    """Return boolean flags for all supported patterns."""
    flags: Dict[str, bool] = {}

    flags["TRIPLE_BOTTOM"] = _triple_bottom(df)
    flags["DOUBLE_BOTTOM"] = _double_bottom(df)
    flags["ROUNDING_BOTTOM"] = _rounding_bottom(df)
    flags["BULL_FLAG"] = _bull_flag(df)
    flags["CUP_HANDLE"] = _cup_handle(df)

    # Tactical / indicator-driven patterns from `common` when available
    if common:
        rsi = common.get("RSI14", common.get("rsi14", np.nan))
        vwap_dist = common.get("VWAP_DISTANCE_PCT", np.nan)
        vol_surge = common.get("VOL_SURGE_RATIO", np.nan)
        ema_stack = common.get("DMA_STACK", common.get("dma_stack", ""))
        flags["VWAP_RECLAIM"] = bool(np.isfinite(float(vwap_dist)) and float(vwap_dist) > -0.5 and float(vwap_dist) < 0.5) if vwap_dist is not None else False
        flags["VOL_EXPANSION"] = bool(np.isfinite(float(vol_surge)) and float(vol_surge) >= 1.8) if vol_surge is not None else False
        try:
            flags["RSI_MOMENTUM"] = bool(np.isfinite(float(rsi)) and float(rsi) >= 60)
        except Exception:
            flags["RSI_MOMENTUM"] = False
        flags["BULL_STACKED"] = ("BULL" in str(ema_stack).upper())

    # Darvas breakout if columns exist
    try:
        sig, _pct = darvas_box_signal(df)
        flags["DARVAS_BREAKOUT"] = bool(sig)
    except Exception:
        flags["DARVAS_BREAKOUT"] = False

    return flags

_PATTERN_PRIORITY = [
    "DARVAS_BREAKOUT",
    "SMC_BREAKOUT",
    "CUP_HANDLE",
    "TRIPLE_BOTTOM",
    "DOUBLE_BOTTOM",
    "ROUNDING_BOTTOM",
    "BULL_FLAG",
    "VWAP_RECLAIM",
    "VOL_EXPANSION",
    "RSI_MOMENTUM",
    "MEAN_REVERSION",
    "BULLISH_ENGULFING",
    "HAMMER",
]

def patterns_to_strings(flags: Dict[str, bool]) -> Tuple[str, str]:
    patterns = [k for k, v in flags.items() if bool(v)]
    if not patterns:
        return ("NONE", "NONE")
    # stable ordering by priority then alpha
    ordered = [p for p in _PATTERN_PRIORITY if p in patterns]
    remainder = sorted([p for p in patterns if p not in ordered])
    patterns_str = ", ".join(ordered + remainder)
    primary = (ordered[0] if ordered else sorted(patterns)[0])
    return (patterns_str, primary)

def pattern_strength(flags: Dict[str, bool], common: dict | None = None) -> int:
    """0..100 strength heuristic."""
    score = 0
    # structural
    if flags.get("DARVAS_BREAKOUT"): score += 35
    if flags.get("CUP_HANDLE"): score += 30
    if flags.get("TRIPLE_BOTTOM"): score += 28
    if flags.get("DOUBLE_BOTTOM"): score += 20
    if flags.get("BULL_FLAG"): score += 18
    if flags.get("ROUNDING_BOTTOM"): score += 16
    # tactical confirmations
    if flags.get("SMC_BREAKOUT"): score += 25
    if flags.get("VOL_EXPANSION"): score += 12
    if flags.get("VWAP_RECLAIM"): score += 10
    if flags.get("RSI_MOMENTUM"): score += 8
    if flags.get("MEAN_REVERSION"): score += 8
    # trend alignment
    if common:
        try:
            adx = float(common.get("ADX14", common.get("adx14", np.nan)))
            if np.isfinite(adx) and adx >= 25:
                score += 10
        except Exception:
            pass
    return int(min(100, max(0, score)))



# ============================================================
# Wrapper used by Excel pipeline
# Converts boolean pattern flags → list of active patterns
# ============================================================
def compute_patterns(df: pd.DataFrame, common: dict | None = None):
    """
    Returns list of active pattern names.
    This is the function expected by buy_srs_update_excel.py
    """
    try:
        flags = compute_patterns_full(df, common)

        active = []
        for k, v in flags.items():
            try:
                if bool(v):
                    active.append(k)
            except Exception:
                pass

        return active

    except Exception as e:
        return []


# ============================================================
# Market stage classifier (Wyckoff-lite)
# ============================================================
def compute_market_stage_substage(df: pd.DataFrame):
    try:
        close = df["close"]
        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()

        if len(df) < 220:
            return "Neutral/Transition", "INSUFFICIENT_DATA"

        if ma50.iloc[-1] > ma200.iloc[-1]:
            if close.iloc[-1] > ma50.iloc[-1]:
                return "Accumulation", "ACCUMULATION_BREAKOUT"
            else:
                return "Accumulation", "ACCUMULATION_PULLBACK"
        else:
            if close.iloc[-1] < ma50.iloc[-1]:
                return "Distribution", "DISTRIBUTION_BREAKDOWN"
            else:
                return "Distribution", "DISTRIBUTION_BOUNCE"

    except Exception:
        return "Neutral/Transition", "NEUTRAL_CHOP"
