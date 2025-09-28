import numpy as np
import pandas as pd
from config import symbol_to_sector, sector_etfs
from fetching import fetch_data_cached
from compute import compute_indicators
from backtest import evaluate_backtest_accuracy
from upward import detect_bullish_engulfing, detect_hammer
from fetching import fetch_data_cached
from compute import compute_indicators
from backtest import evaluate_backtest_accuracy
from config import symbol_to_sector, sector_etfs


DEBUG = True

from xgboost import XGBClassifier
import joblib
import os
from sklearn.preprocessing import StandardScaler
import math
import numpy as np
import pandas as pd
# Load pre-trained model if available (can be trained separately)
model = None
model_path = os.path.join(os.path.dirname(__file__), "strong_buy_xgb_model.pkl")
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print("❌ Model file not found.")



def compute_confidence_score(row):
    weights = {
        'institutional_score': 0.25,
        'volume_weight': 0.2,
        'sector_corr': 0.15,
        'signal_score': 0.15,
        'adx': 0.1,
        'volume_surge': 0.1,
        'near_support': 0.05
    }

    score = (
            weights['institutional_score'] * row['institutional_score'] +
            weights['volume_weight'] * row['volume_weight'] +
            weights['sector_corr'] * row['sector_corr'] +
            weights['signal_score'] * row['signal_score'] +
            weights['adx'] * (row['ADX_14'] / 100) +
            weights['volume_surge'] * row['volume_surge'] +
            weights['near_support'] * row['near_support']
    )
    return round(score, 2)


def compute_final_recommendation(row, model):
    try:
        confidence_score = compute_confidence_score(row)
        row['confidence_score'] = confidence_score

        adx = row['ADX_14']
        institutional_score = row['institutional_score']
        volume_weight = row['volume_weight']
        sector_corr = row.get('sector_corr', 0.0)
        signal_score = row['signal_score']

        feature_row = pd.DataFrame([{
            "confidence_score": float(confidence_score),
            "sector_corr": float(sector_corr),
            "signal_score": float(signal_score),
            "adx": float(adx),
            "institutional_score": float(institutional_score),
            "volume_weight": float(volume_weight)
        }])

        model_proba = model.predict_proba(feature_row)[0][1]  # Probability of class 1
        row['model_probability'] = round(model_proba, 2)

        recommendation = 'HOLD'
        if model_proba >= 0.8:
            recommendation = 'STRONG BUY'
        elif model_proba >= 0.6:
            recommendation = 'BUY'

        return recommendation, round(model_proba, 2), round(confidence_score, 2)

    except Exception as e:
        print(f"⚠️ Model prediction failed in compute_final_recommendation: {e}")
        row['model_probability'] = 0.0
        return 'HOLD', 0.0, 0.0

def model_predict_proba(features: dict) -> float:
    global model
    if model:
        try:
            feature_row = pd.DataFrame([features])
            feature_row.columns = feature_row.columns.astype(str)
            feature_row.columns = [str(col) for col in feature_row.columns]
            feature_row.columns = feature_row.columns.map(str)
            return float(model.predict_proba(feature_row)[0][1])
        except Exception as e:
            print(f"⚠️ Model prediction failed in model_predict_proba: {e}")
            return 0.0
    else:
        confidence = features.get("confidence_score", 0)
        signal = features.get("signal_score", 0)
        darvas = features.get("darvas_breakout_pct", 0)
        prob = 0.5 * confidence + 0.3 * signal + 0.2 * darvas
        return min(max(prob, 0.0), 1.0)


def is_strong_buy_v2(recommendation, trend_htf, trend_itf, trend_ltf, confidence_score, sector_corr, signal_score):
    return (
        recommendation == "BUY" and
        trend_htf == "UP" and
        trend_itf == "UP" and
        trend_ltf == "UP" and
        confidence_score >= 0.7 and
        sector_corr >= 0.5 and
        signal_score >= 0.6
    )


def analyze_symbol(symbol: str):
    try:
        df = fetch_data_cached(symbol, '3 Y', '1 day')
        print(f"[{symbol}] Columns before indicators: {df.columns.tolist()}")
        df = compute_indicators(df, symbol=symbol)
        print(f"[{symbol}] Columns after indicators: {df.columns.tolist()}")

        if 'EMA_20' not in df.columns or 'EMA_50' not in df.columns or 'EMA_21' not in df.columns:
            raise KeyError("Required EMA columns missing in compute_indicators output")

        ema21 = df['EMA_21'].iloc[-1]
        ema50 = df['EMA_50'].iloc[-1]
        vwap_support = df['vwap_support'].iloc[-1]
        darvas_low = df['darvas_low'].iloc[-1] if 'darvas_low' in df.columns else np.nan
        bb_lower = df['BB_lower'].iloc[-1] if 'BB_lower' in df.columns else np.nan
        swing_low = df['low'].tail(5).min()

        price_candidates = [ema21, vwap_support, darvas_low, bb_lower, swing_low]
        price_candidates = [p for p in price_candidates if not np.isnan(p)]

        # Outlier rejection
        q1, q3 = np.percentile(price_candidates, [25, 75])
        iqr = q3 - q1
        filtered_prices = [p for p in price_candidates if (q1 - 1.5 * iqr) <= p <= (q3 + 1.5 * iqr)]

        # Fib retracement-based bias adjustment
        recent_high = df['high'].tail(10).max()
        recent_low = df['low'].tail(10).min()
        fib_61 = recent_low + 0.618 * (recent_high - recent_low)

        adx = df['ADX_14'].iloc[-1]
        if adx > 25:
            weights = [0.5, 0.2, 0.1, 0.1, 0.1]
        else:
            weights = [0.2, 0.4, 0.1, 0.1, 0.2]

        valid_prices = filtered_prices if filtered_prices else price_candidates
        if len(valid_prices) != len(weights):
            weights = [1] * len(valid_prices)
        buy_price = round(np.average(valid_prices, weights=weights), 2)

        print(f"[{symbol}] Buy Price Components — EMA21: {ema21:.2f}, VWAP: {vwap_support:.2f}, DarvasLow: {darvas_low:.2f}, BBLower: {bb_lower:.2f}, SwingLow: {swing_low:.2f} → Final: {buy_price:.2f}")

        # Bias toward fib retracement if lower than average
        if fib_61 < buy_price:
            buy_price = round((buy_price + fib_61) / 2, 2)

        institutional_score = df['institutional_score'].iloc[-1]
        confidence_score = df['confidence_score'].iloc[-1]
        volume_weight = df['volume_weight'].iloc[-1]
        trend = df['HTF_Trend'].iloc[-1]
        recommendation = df['recommendation'].iloc[-1]

        darvas_breakout_pct = round(float(df['darvas_breakout_pct'].iloc[-1]), 2)
        darvas_signal = "✅" if bool(df['darvas_signal'].iloc[-1]) else "❌"

        # --- Sector correlation (safe) ---
        sector_corr = 0.0
        etf_symbol = sector_etfs.get(symbol_to_sector.get(symbol))

        if etf_symbol and etf_symbol != symbol:
            try:
                df_etf = fetch_data_cached(etf_symbol, '3 Y', '1 day')
                df_etf = compute_indicators(df_etf, symbol=etf_symbol)

                s = df[['date', 'close']].rename(columns={'close': 'close_sym'})
                e = df_etf[['date', 'close']].rename(columns={'close': 'close_etf'})
                merged = s.merge(e, on='date', how='inner')

                if len(merged) >= 21:
                    rolling_corr = (
                        merged['close_sym'].pct_change()
                        .rolling(20)
                        .corr(merged['close_etf'].pct_change())
                    )
                    val = rolling_corr.iloc[-1]
                    if pd.notna(val) and np.isfinite(val):
                        sector_corr = float(val)
            except Exception as e:
                print(f"[{symbol}] sector correlation error: {e}")

        # ensure scalar, and also add a column if you want it available downstream
        if not np.isfinite(sector_corr):
            sector_corr = 0.0
        df['sector_corr'] = sector_corr  # single scalar broadcast OK

        volume_surge = round(float(df['volume_surge'].iloc[-1]), 2) if 'volume_surge' in df.columns else 0
        near_support = "✅" if bool(df['near_support'].iloc[-1]) else "❌"
        signal_score = round(float(df['signal_score'].iloc[-1]), 2) if 'signal_score' in df.columns else 0

        bullish_engulfing = detect_bullish_engulfing(df)
        hammer = detect_hammer(df)
        candle_signal = "✅" if bullish_engulfing or hammer else "❌"

        hit, max_gain, days_to_peak = evaluate_backtest_accuracy(symbol=symbol, df=df, buy_price=buy_price)
        ten_day_gain = round(max_gain, 2)
        strong_hit = "✅" if hit else "❌"

        trend_itf = df['ITF_Trend'].iloc[-1]
        trend_ltf = df['LTF_Trend'].iloc[-1]

        strong_buy = is_strong_buy_v2(
            recommendation=recommendation,
            trend_htf=trend,
            trend_itf=trend_itf,
            trend_ltf=trend_ltf,
            confidence_score=confidence_score,
            sector_corr=sector_corr,
            signal_score=signal_score
        )

        if model:
            # Build full feature set from latest row
            latest = df.iloc[-1:].copy()
            latest.columns = latest.columns.astype(str)
            latest.columns = [col.strip() for col in latest.columns]

            expected_features = [str(col).strip() for col in model.feature_names_in_]
            for col in expected_features:
                if col not in latest.columns:
                    latest[col] = 0.0

            features = latest.reindex(columns=expected_features).astype(float)
            model_proba = max(0.0, min(1.0, float(model.predict_proba(features)[0][1])))

            model_strong_buy = model_proba >= 0.80  # Lowered threshold
            strong_buy = model_strong_buy
        else:
            model_proba = None
            model_strong_buy = False


        sector_corr = float(df['sector_corr'].iloc[-1] or 0.0)

        return {
            "Symbol": symbol,
            "Refined Buy Price": buy_price,
            "VWAP Support": round(float(vwap_support), 2),
            "ADX": round(float(adx), 2),
            "Institutional Score": round(float(institutional_score), 2),
            "Volume Weight": round(float(volume_weight), 2),
            "Confidence Score": round(float(confidence_score), 2),
            "Sector Correlation": round(float(sector_corr), 2),
            "Trend": trend,
            "HTF_Trend": trend,
            "ITF_Trend": trend_itf,
            "LTF_Trend": trend_ltf,
            "Recommendation": recommendation,
            "Darvas Breakout %": darvas_breakout_pct,
            "Darvas Signal": darvas_signal,
            "Strong Buy": "✅" if strong_buy else "❌",
            "Model Probability": round(model_proba, 4) if model else "N/A",
            "Rule-Based Buy": "✅" if is_strong_buy_v2(
                recommendation=recommendation,
                trend_htf=trend,
                trend_itf=trend_itf,
                trend_ltf=trend_ltf,
                confidence_score=confidence_score,
                sector_corr=sector_corr,
                signal_score=signal_score
            ) else "❌",
            "Model-Driven Buy": "✅" if model_strong_buy else "❌",
            "10D Hit": strong_hit,
            "10D Gain (%)": ten_day_gain,
            "Days to Peak": days_to_peak,
            "Volume Surge": volume_surge,
            "Near Support": near_support,
            "Signal Score": signal_score,
            "Candle Pattern": candle_signal
        }

    except Exception as e:
        print(f"⚠️ Error analyzing {symbol}: {e}")
        return None


def detect_mean_reversion_buy(df):
    last = df.iloc[-1]
    return (
        last['close'] <= last['BB_lower'] and
        last['RSI_14'] < 40 and
        last['MACD_hist'] < 0
    )
