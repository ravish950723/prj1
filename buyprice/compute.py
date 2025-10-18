
import pandas_ta as ta
from fetching import fetch_data_cached
from institutional_investor import score_institutional_investor
from darvas import darvas_box_signal
import numpy as np
import pandas as pd



DEBUG = False



def _classify_market_stage(df: pd.DataFrame) -> pd.Series:
    """
    Classify each row into a market stage using already-computed columns.
    Stages: Accumulation, Mark-Up, Distribution, Mark-Down, Neutral/Transition.
    Priority: Mark-Down > Distribution > Mark-Up > Accumulation > Neutral/Transition
    """
    n = len(df)
    stage = pd.Series(["Neutral/Transition"] * n, index=df.index, dtype="object")

    # Safe getters (avoid KeyError / dtype surprises)
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

    tight_range      = getcol("tight_range", False, "bool")
    ema_up           = getcol("EMA_uptrend", False, "bool")
    htf              = getcol("HTF_Trend", "NEUTRAL", "str").str.upper()
    strong_trend     = getcol("strong_trend", 0, "int")
    adx              = getcol("ADX_14", 0.0, "float")
    macd_hist_slope  = getcol("MACD_hist_slope", 0.0, "float")
    macd_hist        = getcol("MACD_hist", 0.0, "float")

    # Masks (vectorized)
    mark_down   = (htf == "DOWN") & (~ema_up) & (macd_hist < 0)
    mark_up     = (htf == "UP") & ((strong_trend == 1) | (adx >= 25))
    distribution = (tight_range) & (htf == "UP") & (macd_hist_slope <= 0)
    accumulation = (tight_range) & (~ema_up)

    # Apply in priority order
    stage.loc[accumulation] = "Accumulation"
    stage.loc[mark_up]      = "Mark-Up"
    stage.loc[distribution] = "Distribution"
    stage.loc[mark_down]    = "Mark-Down"

    if DEBUG:
        print("[stage] counts:", stage.value_counts(dropna=False).to_dict())
        print("[stage] masks  :",
              {"accum": int(accumulation.sum()),
               "mark_up": int(mark_up.sum()),
               "dist": int(distribution.sum()),
               "mark_down": int(mark_down.sum())})

    return stage


def compute_indicators(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Compute a broad set of indicators used downstream.
    Assumes df has columns: date, open, high, low, close, volume.
    Returns the same df with indicator columns appended, including 'market_stage'.
    """
    df = df.copy()

    # Normalize core price/vol types early
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")

    # === Core Trend & Momentum Indicators ===
    df["EMA_20"]  = ta.ema(df["close"], length=20)
    df["EMA_21"]  = ta.ema(df["close"], length=21)
    df["EMA_50"]  = ta.ema(df["close"], length=50)
    df["EMA_200"] = ta.ema(df["close"], length=200)

    adx = ta.adx(df["high"], df["low"], df["close"], length=14)
    df["ADX_14"] = pd.to_numeric(adx["ADX_14"], errors="coerce")

    macd = ta.macd(df["close"])
    df["MACD"]            = pd.to_numeric(macd["MACD_12_26_9"], errors="coerce")
    df["MACD_signal"]     = pd.to_numeric(macd["MACDs_12_26_9"], errors="coerce")
    df["MACD_hist"]       = pd.to_numeric(macd["MACDh_12_26_9"], errors="coerce")
    df["MACD_hist_slope"] = df["MACD_hist"].diff().rolling(3).mean()
    df["MACD_crossover"]  = (df["MACD_hist"] > 0) & (df["MACD_hist"].shift(1) < 0)

    df["RSI_14"]   = ta.rsi(df["close"], length=14)
    df["RSI_slope"] = df["RSI_14"].diff().rolling(3).mean()
    df["OBV"]       = ta.obv(df["close"], df["volume"])

    bb = ta.bbands(df["close"], length=20)
    df[["BB_upper", "BB_middle", "BB_lower"]] = bb[["BBU_20_2.0", "BBM_20_2.0", "BBL_20_2.0"]]

    # === Volatility Metrics ===
    df["ATR_14"]   = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["BB_width"] = df["BB_upper"] - df["BB_lower"]
    df["return_std"] = df["close"].pct_change().rolling(10).std()

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

    # === Composite Signal Score ===
    df["signal_score"] = (
        df["EMA_uptrend"].astype(int) * 0.15
        + df["MACD_crossover"].astype(int) * 0.15
        + (df["strong_trend"] > 0).astype(int) * 0.15
        + df["darvas_signal"].fillna(0).astype(int) * 0.10
        + df["tight_range"].astype(int) * 0.10
        + df["volume_surge"].astype(int) * 0.10
        + df["near_support"].astype(int) * 0.05
        + (df["green_candles"] / 3.0).clip(lower=0, upper=1) * 0.05
        + df["above_EMA200"] * 0.05
        + (df["MACD_hist_slope"] > 0).astype(int) * 0.05
    ).fillna(0.0)

    df["refined_buy_signal"] = df["signal_score"] >= 0.75

    # === Confidence Metrics ===
    # score_institutional_investor expects/uses OBV/volume/close
    df["institutional_score"] = score_institutional_investor(df)
    df["volume_weight"] = np.minimum(df["volume"] / df["volume_avg_20"].replace(0, np.nan), 2.0).fillna(0.0)

    df["confidence_score"] = (
        df["institutional_score"].fillna(0) * 0.5
        + df["volume_weight"].fillna(0) * 0.3
        + (df["ADX_14"].fillna(0) / 100.0) * 0.2
    ).clip(0, 1)

    # === Recommendation Assignment ===
    df["recommendation"] = np.where(df["refined_buy_signal"], "BUY", "HOLD")

    # === Trend Levels (string labels for downstream use) ===
    df["HTF_Trend"] = np.where((df["EMA_21"] > df["EMA_50"]), "UP", "DOWN")
    df["ITF_Trend"] = np.where(ta.ema(df["close"], length=8)  > ta.ema(df["close"], length=21), "UP", "DOWN")
    df["LTF_Trend"] = np.where(ta.ema(df["close"], length=5)  > ta.ema(df["close"], length=13), "UP", "DOWN")

    # === FINAL: Market Stage ===
    df["market_stage"] = _classify_market_stage(df)

    if DEBUG:
        print("[compute] market_stage counts:", df["market_stage"].value_counts(dropna=False).to_dict())

    return df


def analyze_symbol_all(symbol: str):
    """
    Return a compact dict of latest metrics for a symbol.
    NOTE: Sector correlation code remains in symbol_analysis.py; here we focus on indicator computation.
    """
    try:
        df = fetch_data_cached(symbol, "3 Y", "1 day", refresh=True)
        df = compute_indicators(df, symbol=symbol)

        # Buy price heuristic: blend of EMA21, vwap_support, prior Darvas low
        ema21        = float(df["EMA_21"].iloc[-1])
        vwap_support = float(df["vwap_support"].iloc[-1])
        darvas_low   = float(df["darvas_low"].iloc[-1]) if "darvas_low" in df.columns else np.nan
        buy_price    = float(np.nanmean([ema21, vwap_support, darvas_low]))
        entries = candle_entries_multi(df, (2, 4, 6, 8, 12, 18, 30))

        result = {
            "Symbol": symbol,
            "VWAP Support": round(vwap_support, 2),
            "ADX": round(float(df["ADX_14"].iloc[-1]), 2),
            "Institutional Score": round(float(df["institutional_score"].iloc[-1]), 2),
            "Volume Weight": round(float(df["volume_weight"].iloc[-1]), 2),
            "Confidence Score": round(float(df["confidence_score"].iloc[-1]), 2),
            "Trend": str(df["HTF_Trend"].iloc[-1]),
            "Recommendation": str(df["recommendation"].iloc[-1]),
            "Darvas Breakout %": round(float(df["darvas_breakout_pct"].iloc[-1]), 2)
            if "darvas_breakout_pct" in df.columns else 0.0,
            "Darvas Signal": "✅" if int(df.get("darvas_signal", pd.Series([0])).iloc[-1]) == 1 else "❌",
            "Refined Buy Price": round(buy_price, 2),
            "Market Stage": str(df.get("market_stage", pd.Series(["Neutral/Transition"])).iloc[-1]),
        }

        # Merge in candle entries
        for weeks, price in entries.items():
            result[f"Candle Entry {weeks}w"] = round(price, 2) if np.isfinite(price) else None

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
    body = abs(last["close"] - last["open"])
    lower_wick = (min(last["open"], last["close"]) - last["low"])
    upper_wick = (last["high"] - max(last["open"], last["close"]))
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

        # Trend regime weighting
        strong_trend = (adx >= 25) and ema_up
        if strong_trend:
            weights = {"ema21": 0.35, "vwap": 0.30, "darv": 0.15, "bb": 0.10, "swing": 0.10}
        else:
            weights = {"ema21": 0.20, "vwap": 0.30, "darv": 0.15, "bb": 0.20, "swing": 0.15}

        vals, wts = [], []
        def add(v, key):
            if np.isfinite(v):
                vals.append(float(v)); wts.append(weights[key])
        add(ema21, "ema21"); add(vwap_sup, "vwap"); add(darvas_lo, "darv"); add(bb_lower, "bb"); add(swing_low, "swing")
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


