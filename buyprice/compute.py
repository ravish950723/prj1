import pandas as pd
import numpy as np
import pandas_ta as ta

from config import symbol_to_sector, sector_etfs
from fetching import fetch_data_cached
from institutional_investor import score_institutional_investor
from darvas import darvas_box_signal

# Toggle to see stage distributions & sanity checks in stdout
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


def analyze_symbol(symbol: str):
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

        return {
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