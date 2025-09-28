import pandas as pd
import numpy as np
import pandas_ta as ta

from config import symbol_to_sector, sector_etfs
from fetching import fetch_data_cached
from institutional_investor import score_institutional_investor
from darvas import darvas_box_signal

DEBUG = False

def compute_indicators(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Compute a broad set of indicators used downstream.
    Assumes df has columns: date, open, high, low, close, volume.
    Returns the same df with indicator columns appended.
    """
    # === Core Trend & Momentum Indicators ===
    df["EMA_20"] = ta.ema(df["close"], length=20)
    df["EMA_21"] = ta.ema(df["close"], length=21)
    df["EMA_50"] = ta.ema(df["close"], length=50)
    df["EMA_200"] = ta.ema(df["close"], length=200)

    adx = ta.adx(df["high"], df["low"], df["close"], length=14)
    df["ADX_14"] = adx["ADX_14"]


    macd = ta.macd(df["close"])
    df["MACD"] = macd["MACD_12_26_9"]
    df["MACD_signal"] = macd["MACDs_12_26_9"]
    df["MACD_hist"] = macd["MACDh_12_26_9"]
    df["MACD_hist_slope"] = df["MACD_hist"].diff().rolling(3).mean()
    df["MACD_crossover"] = (df["MACD_hist"] > 0) & (df["MACD_hist"].shift(1) < 0)

    df["RSI_14"] = ta.rsi(df["close"], length=14)
    df["RSI_slope"] = df["RSI_14"].diff().rolling(3).mean()
    df["OBV"] = ta.obv(df["close"], df["volume"])
    bb = ta.bbands(df["close"], length=20)
    df[["BB_upper", "BB_middle", "BB_lower"]] = bb[["BBU_20_2.0", "BBM_20_2.0", "BBL_20_2.0"]]

    # === Volatility Metrics ===
    df["ATR_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["BB_width"] = df["BB_upper"] - df["BB_lower"]
    df["return_std"] = df["close"].pct_change().rolling(10).std()

    df = df.copy()
    df.replace({None: np.nan, "None": np.nan, "": np.nan}, inplace=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")



    # === Trend Confirmation ===
    for c in ["EMA_20", "EMA_21", "EMA_50", "EMA_200"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")

    df["EMA_21_slope"] = df["EMA_21"].diff().rolling(5).mean()
    df["EMA_uptrend"] = (df["EMA_21"] > df["EMA_50"]) & (df["EMA_21_slope"] > 0)
    df["strong_trend"] = df["ADX_14"] > 25

    df["above_EMA21"] = (df["close"] > df["EMA_21"]).astype(int)
    df["above_EMA50"] = (df["close"] > df["EMA_50"]).astype(int)
    df["above_EMA200"] = (df["close"] > df["EMA_200"]).astype(int)

    # === Behavioral Patterns ===
    df["green_candles"] = (df["close"] > df["open"]).astype(int).rolling(3).sum()
    df["red_candles"] = (df["close"] < df["open"]).astype(int).rolling(3).sum()
    df["volume_trend"] = df["volume"].diff().rolling(3).mean()

    # === Tight Range Detection ===
    df["range_std"] = (df["high"] - df["low"]).rolling(10).std()
    df["tight_range"] = df["range_std"] < df["range_std"].rolling(50).mean() * 0.8

    # === Darvas Box Breakout ===
    df = darvas_box_signal(df)

    # === Volume Surge Confirmation ===
    df["volume_avg_20"] = df["volume"].rolling(20).mean()
    df["volume_surge"] = df["volume"] > 1.5 * df["volume_avg_20"]

    # === VWAP Support (proxy) ===
    df["vwap_support"] = df["low"].rolling(5).min()

    # === Price Near Support Check (VWAP or BB Lower) ===
    df["near_support"] = (df["close"] <= df["vwap_support"] * 1.02) | (df["close"] <= df["BB_lower"] * 1.02)

    # === Sector Rotation Strength (placeholder) ===
    df["sector_outperformance"] = 0.0

    # === Composite Signal Score ===
    df["signal_score"] = (
        df["EMA_uptrend"].astype(int) * 0.15
        + df["MACD_crossover"].astype(int) * 0.15
        + df["strong_trend"].astype(int) * 0.15
        + df["darvas_signal"].astype(int) * 0.10
        + df["tight_range"].astype(int) * 0.10
        + df["volume_surge"].astype(int) * 0.10
        + df["near_support"].astype(int) * 0.05
        + df["green_candles"] / 3 * 0.05
        + df["above_EMA200"] * 0.05
        + df["MACD_hist_slope"].apply(lambda x: 1 if x > 0 else 0) * 0.05
    )
    df["refined_buy_signal"] = df["signal_score"] >= 0.75

    # === Confidence Metrics ===
    df["institutional_score"] = score_institutional_investor(df)
    df["volume_weight"] = np.minimum(df["volume"] / df["volume_avg_20"], 2.0)
    df["confidence_score"] = (
        df["institutional_score"] * 0.5
        + df["volume_weight"] * 0.3
        + df["ADX_14"].fillna(0) / 100 * 0.2
    )

    df["confidence_score"] = np.clip(df["confidence_score"], 0, 1)
    # === Recommendation Assignment ===
    df["recommendation"] = np.where(df["refined_buy_signal"], "BUY", "HOLD")

    # === Trend Levels ===
    df["HTF_Trend"] = (df["EMA_21"] > df["EMA_50"]).replace({True: "UP", False: "DOWN"})
    df["ITF_Trend"] = (ta.ema(df["close"], length=8) > ta.ema(df["close"], length=21)).replace({True: "UP", False: "DOWN"})
    df["LTF_Trend"] = (ta.ema(df["close"], length=5) > ta.ema(df["close"], length=13)).replace({True: "UP", False: "DOWN"})

    return df


def analyze_symbol(symbol: str):
    """Return a compact dict of latest metrics for a symbol.
    NOTE: Sector correlation code fixed to avoid referencing undefined df_etf.
    """
    try:
        df = fetch_data_cached(symbol, "3 Y", "1 day", refresh=True)
        df = compute_indicators(df, symbol=symbol)

        # Smarter Buy Price Estimation Logic (pullback/bounce areas)
        ema21 = df["EMA_21"].iloc[-1]
        vwap_support = df["vwap_support"].iloc[-1]
        darvas_low = df["darvas_low"].iloc[-1] if "darvas_low" in df.columns else np.nan
        buy_price = float(np.nanmean([ema21, vwap_support, darvas_low]))

        institutional_score = float(df["institutional_score"].iloc[-1])
        confidence_score = float(df["confidence_score"].iloc[-1])
        volume_weight = float(df["volume_weight"].iloc[-1])
        adx = float(df["ADX_14"].iloc[-1])
        trend = df["HTF_Trend"].iloc[-1]
        recommendation = df["recommendation"].iloc[-1]

        darvas_breakout_pct = float(round(df["darvas_breakout_pct"].iloc[-1], 2))
        darvas_signal = "✅" if int(df["darvas_signal"].iloc[-1]) == 1 else ""

        # pick sector ETF
        etf_symbol = sector_etfs.get(symbol_to_sector.get(symbol))
        sector_corr = 0.0
        try:
            if etf_symbol and etf_symbol != symbol:
                df_etf = fetch_data_cached(etf_symbol, '3 Y', '1 day')
                df_etf = compute_indicators(df_etf, symbol=etf_symbol)

                s = df[['date', 'close']].rename(columns={'close': 'close_sym'})
                e = df_etf[['date', 'close']].rename(columns={'close': 'close_etf'})
                merged = s.merge(e, on='date', how='inner')

                # after merged = s.merge(e, on='date', how='inner')
                if len(merged) < 21:
                    if DEBUG:
                        print(f"[{symbol}] DEBUG sector-corr: merged_len={len(merged)} (<21) → forcing 0.0")
                    sector_corr = 0.0
                else:
                    rc = (merged['close_sym'].pct_change()
                          .rolling(20)
                          .corr(merged['close_etf'].pct_change()))
                    val = rc.iloc[-1]
                    if pd.isna(val) or not np.isfinite(val):
                        if DEBUG:
                            na_ratio = float(rc.isna().mean())
                            print(f"[{symbol}] DEBUG sector-corr: NaN/inf val; rc_len={len(rc)}, NaN%={na_ratio:.2%}")
                        sector_corr = 0.0
                    else:
                        sector_corr = float(val)
                        if DEBUG and abs(sector_corr) < 1e-6:
                            print(f"[{symbol}] DEBUG sector-corr near zero with valid rc (check ETF mapping)")

        except Exception as e:
            print(f"[{symbol}] sector correlation error: {e}")

        if (sector_corr is None) or (not np.isfinite(sector_corr)):
            sector_corr = 0.0

        return {
            "Symbol": symbol,
            "VWAP Support": round(vwap_support, 2),
            "ADX": round(adx, 2),
            "Institutional Score": round(institutional_score, 2),
            "Volume Weight": round(volume_weight, 2),
            "Confidence Score": round(confidence_score, 2),
            "Sector Correlation": round(sector_corr, 2),
            "Trend": trend,
            "Recommendation": recommendation,
            "Darvas Breakout %": darvas_breakout_pct,
            "Darvas Signal": darvas_signal,
            "Refined Buy Price": round(buy_price, 2),
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
