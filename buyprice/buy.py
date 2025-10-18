from config import symbols
from compute import analyze_symbol_all
from fetching import fetch_data_cached
from compute import compute_indicators
from backtest import evaluate_backtest_accuracy
import pandas as pd
import numpy as np
import argparse

from upward import (
    detect_smc_accumulation_breakout,
    detect_mean_reversion_buy,
    detect_bullish_engulfing,
    detect_hammer,
    compute_upward_trend,
    compute_signal_score
)

# ==============================
# Configuration / Thresholds
# ==============================
DEBUG = True
BUY_THRESH = 0.60
STRONG_BUY_THRESH = 0.80

# Fallback tech-score threshold when model prob is low
TECH_BUY_FALLBACK = 0.60


# --- model_loader.py (or top of buy.py) ---
from pathlib import Path
import os, joblib

from pathlib import Path
import joblib, json

HERE = Path(__file__).resolve().parent

def load_model():
    for name in ["strong_buy_xgb_model_calibrated.pkl"]:
        p = HERE / name
        if p.exists():
            m = joblib.load(p)
            print(f"✅ Loaded model: {p.name}")
            return m
    # print("❌ Model file not found.")
    return None

def load_threshold(default=0.25):
    p = HERE / "strong_buy_thresholds.json"
    if p.exists():
        try:
            obj = json.loads(p.read_text())
            thr = float(obj.get("best_f1", {}).get("thr", default))
            print(f"✅ Using threshold (best_f1): {thr:.2f}")
            return thr
        except Exception as e:
            print(f"[WARN] failed to read thresholds: {e}")
    print(f"[INFO] Using default threshold: {default:.2f}")
    return default

model = load_model()
thr_buy = load_threshold()



def get_confidence_band(prob: float) -> str:
    """Map model probability to a human label."""
    if prob >= STRONG_BUY_THRESH:
        return "STRONG BUY"
    if prob >= BUY_THRESH:
        return "BUY"
    if prob >= 0.40:
        return "HOLD"
    return "WATCH"


def _load_fallback_feature_list(path: str = "model_features.txt"):
    try:
        with open(path) as f:
            feats = [line.strip() for line in f if line.strip()]
            return feats if feats else None
    except Exception:
        return None


def _expected_features_for_model(_model, df_sample: pd.DataFrame) -> list:
    """
    Determine the expected feature order for inference:
    1) Prefer model.feature_names_in_
    2) Fallback to model_features.txt (if present)
    3) Last-resort: numeric intersection of df columns
    """
    if _model is not None and hasattr(_model, "feature_names_in_"):
        try:
            names = [str(c).strip() for c in _model.feature_names_in_]
            if len(names) > 0:
                return names
        except Exception:
            pass

    file_feats = _load_fallback_feature_list("model_features.txt")
    if file_feats:
        return file_feats

    return df_sample.select_dtypes(include="number").columns.tolist()


def debug_feature_row(symbol, features_df, expected_cols, note=""):
    """Lightweight feature diagnostics before predict."""
    try:
        row = features_df.iloc[0]
        na = int(row.isna().sum())
        zeros = int((row == 0).sum())
        print(
            f"[{symbol}] DEBUG{f' {note}' if note else ''} → feats:{features_df.shape[1]} | "
            f"NaNs:{na} | Zeros:{zeros} | min:{float(row.min()):.4f} | max:{float(row.max()):.4f}"
        )
        for k in [
            "confidence_score",
            "signal_score",
            "ADX_14",
            "volume_weight",
            "institutional_score",
            "darvas_breakout_pct",
        ]:
            if k in features_df.columns:
                v = row[k]
                try:
                    print(f"   • {k}: {None if pd.isna(v) else float(v):.4f}")
                except Exception:
                    print(f"   • {k}: {v}")
        missing = [c for c in expected_cols if c not in features_df.columns]
        extra = [c for c in features_df.columns if c not in expected_cols]
        if missing:
            print(
                f"   ⚠️ Missing in features (expected by model): {missing[:10]}{'...' if len(missing)>10 else ''}"
            )
        if extra:
            print(
                f"   ⚠️ Extra columns (not expected by model): {extra[:10]}{'...' if len(extra)>10 else ''}"
            )
    except Exception as e:
        print(f"[{symbol}] DEBUG feature dump failed: {e}")


def _predict_proba_for_last_row(symbol: str, df: pd.DataFrame) -> float:
    """Align last row to model schema and get prediction probability. Returns float in [0,1]."""
    global model
    if model is None:
        return 0.0

    features = df.iloc[-1:].copy()
    features.columns = features.columns.astype(str)
    features.columns = [col.strip() for col in features.columns]

    expected = _expected_features_for_model(model, features)
    for col in expected:
        if col not in features.columns:
            features[col] = 0.0
    features = features.reindex(columns=expected).astype(float).fillna(0.0)

    if DEBUG:
        debug_feature_row(symbol, features, expected, note="pre-predict")

    try:
        proba = float(model.predict_proba(features)[0][1])
        # Clamp to [0,1] just in case a custom calibrator or odds sneaks in
        if not (0.0 <= proba <= 1.0):
            print(f"[WARN] Model proba out of [0,1]: {proba:.4f} → clamping")
            proba = max(0.0, min(1.0, proba))
        return proba
    except Exception as e:
        print(f"⚠️ Model prediction failed for {symbol}: {e}")
        return 0.0


def _compute_indicator_snap(df: pd.DataFrame) -> dict:
    """Extract readable technical statuses from the latest row."""
    last = df.iloc[-1]
    snap = {}

    # EMA / Trend
    snap["EMA_Uptrend"] = bool(last.get("EMA_uptrend", False))
    snap["EMA21_Slope"] = float(last.get("EMA_21_slope", np.nan))

    # ADX
    adx = float(last.get("ADX_14", np.nan))
    snap["ADX"] = adx
    snap["ADX_Strength"] = "Strong" if adx >= 25 else "Weak"

    # MACD
    macd_cross = bool(last.get("MACD_crossover", False))
    snap["MACD_Crossover"] = "✅" if macd_cross else "❌"

    # RSI
    rsi = float(last.get("RSI_14", np.nan))
    snap["RSI_14"] = rsi
    if rsi >= 70:
        snap["RSI_State"] = "Overbought"
    elif rsi <= 30:
        snap["RSI_State"] = "Oversold"
    else:
        snap["RSI_State"] = "Neutral"

    # OBV trend (slope)
    try:
        obv_slope = float(df["OBV"].diff().rolling(5).mean().iloc[-1])
    except Exception:
        obv_slope = float("nan")
    snap["OBV_Slope"] = obv_slope
    snap["OBV_Trend"] = "Up" if obv_slope > 0 else ("Down" if obv_slope < 0 else "Flat")

    # Bollinger touch / proximity
    close = float(last.get("close", np.nan))
    bb_lower = float(last.get("BB_lower", np.nan))
    bb_upper = float(last.get("BB_upper", np.nan))
    snap["At_BB_Lower"] = "✅" if close <= bb_lower * 1.01 else "❌"
    snap["At_BB_Upper"] = "✅" if close >= bb_upper * 0.99 else "❌"

    return snap


def _tech_fallback_score(snap: dict, df: pd.DataFrame, smc_breakout: bool, mean_rev: bool) -> float:
    """Score a quick technical composite for fallback when model prob is low."""
    score = 0.0
    # Trend & momentum
    if snap.get("EMA_Uptrend"):
        score += 0.25
    if snap.get("ADX", 0) >= 25:
        score += 0.20
    if snap.get("MACD_Crossover") == "✅":
        score += 0.20
    # RSI sweet-spot (avoid extremes)
    rsi = snap.get("RSI_14") or 50
    if 40 <= rsi <= 60:
        score += 0.10
    # Accumulation via OBV slope
    if snap.get("OBV_Slope", 0) > 0:
        score += 0.10
    # Price-action helpers
    if smc_breakout:
        score += 0.10
    if mean_rev:
        score += 0.05
    # Near key supports
    near_support = bool(df.iloc[-1].get("near_support", False))
    if near_support:
        score += 0.05

    return round(score, 2)


def parse_args():
    parser = argparse.ArgumentParser(description="Run buy analysis with optional CSV output")
    parser.add_argument("--csv", dest="csv", action="store_true", help="Save predictions to CSV")
    parser.add_argument("--no-csv", dest="csv", action="store_false", help="Do not save predictions to CSV")
    parser.set_defaults(csv=True)
    return parser.parse_args()


def main():
    summary = []
    args = parse_args()

    # First pass: analyze and collect base fields
    for symbol in symbols:
        result = analyze_symbol_all(symbol)
        if result:
            ce = {k: result.get(k) for k in
                  ("Candle Entry 2w", "Candle Entry 4w", "Candle Entry 6w", "Candle Entry 8w",
                   "Candle Entry 12w", "Candle Entry 18w", "Candle Entry 30w")}
            print(f"[{symbol}] Candle Entries → {ce}")
            summary.append(result)
        # if result:
        #     summary.append(result)

    if not summary:
        print("⚠️ No valid predictions could be generated.")
        return

    print("Summary of Predictions and Recommendations:")

    # We label these "90D" to match backtest.tail(90) window
    hit_list, gain_list, days_list = [], [], []

    # Augment with signals, model proba, indicators, and backtest
    for res in summary:
        symbol = res["Symbol"]
        try:
            df = fetch_data_cached(symbol, duration='3 Y', bar_size='1 day')
            df = compute_indicators(df, symbol=symbol)

            # Pattern/price-action signals
            df = compute_upward_trend(df)
            signal_score, signal_count, pa_reco = compute_signal_score(df)
            res["Signal Score"] = round(signal_score, 2)
            res["Signal Count"] = signal_count
            res["Signal"] = pa_reco

            # Individual booleans
            smc = detect_smc_accumulation_breakout(df)
            mean_rev = detect_mean_reversion_buy(df)
            res["SMC_Breakout"] = smc
            res["Mean_Reversion"] = mean_rev
            res["Bullish_Engulfing"] = detect_bullish_engulfing(df)
            res["Hammer"] = detect_hammer(df)
            res["Trend_Strength"] = int(df.iloc[-1].get("trend_strength", 0))

            # Technical snapshot (readable for the table)
            snap = _compute_indicator_snap(df)
            res.update({
                "EMA Uptrend": "✅" if snap["EMA_Uptrend"] else "❌",
                "EMA21 Slope": round(snap["EMA21_Slope"], 4) if not np.isnan(snap["EMA21_Slope"]) else np.nan,
                "ADX Strength": snap["ADX_Strength"],
                "MACD Cross": snap["MACD_Crossover"],
                "RSI": round(snap["RSI_14"], 1) if not np.isnan(snap["RSI_14"]) else np.nan,
                "RSI State": snap["RSI_State"],
                "OBV Trend": snap["OBV_Trend"],
                "At BB Lower": snap["At_BB_Lower"],
            })

            # Simple rule-based SMA cross + confidence filter
            df["SMA_20"] = df["close"].rolling(window=20).mean()
            df["SMA_50"] = df["close"].rolling(window=50).mean()
            df["SMA_cross_signal"] = (df["SMA_20"] > df["SMA_50"]) & (df["SMA_20"].shift(1) <= df["SMA_50"].shift(1))
            rule_based_buy = bool(df["SMA_cross_signal"].iloc[-1]) and float(res["Confidence Score"]) > BUY_THRESH
            res["Rule-Based Buy"] = "✅" if rule_based_buy else "❌"

            # Model probability (schema-safe)
            prob = _predict_proba_for_last_row(symbol, df)
            res["Model Probability"] = round(prob, 2)
            res["Model-Driven Buy"] = "✅" if prob >= STRONG_BUY_THRESH else "❌"
            res["Confidence Band"] = get_confidence_band(prob)

            # Fallback technical BUY when model is low
            if prob < BUY_THRESH:
                tech_score = _tech_fallback_score(snap, df, smc, mean_rev)
                res["Tech Fallback Score"] = tech_score
                if tech_score >= TECH_BUY_FALLBACK and signal_count >= 2:
                    res["Signal"] = "BUY (Tech Fallback)"
                else:
                    # keep pa_reco from compute_signal_score
                    pass
            else:
                res["Tech Fallback Score"] = 0.0

            # Backtest (matches evaluate_backtest_accuracy's 90-bar window)
            buy_price = res["Refined Buy Price"]
            hit, gain, days_to_peak = evaluate_backtest_accuracy(symbol, df, buy_price)

        except Exception as e:
            print(f"⚠️ Could not evaluate 90D backtest for {symbol}: {e}")
            hit, gain, days_to_peak = False, 0.0, -1
            res.setdefault("Model Probability", 0.0)
            res.setdefault("Model-Driven Buy", "❌")
            res.setdefault("Confidence Band", "WATCH")


        hit_list.append("✅" if hit else "❌")
        gain_list.append(round(gain, 2))
        days_list.append(days_to_peak if days_to_peak >= 0 else "N/A")

    # Prepare final DataFrame
    df_summary = pd.DataFrame(summary)
    for col in [
        "Candle Entry 2w", "Candle Entry 4w", "Candle Entry 6w", "Candle Entry 8w",
        "Candle Entry 12w", "Candle Entry 18w", "Candle Entry 30w"
    ]:
        if col not in df_summary.columns:
            df_summary[col] = np.nan

    df_summary["90D Hit"] = hit_list
    df_summary["90D Gain (%)"] = gain_list
    df_summary["Days to Peak"] = days_list

    # Rank primarily by model probability; could add composite later
    df_summary.sort_values(by="Model Probability", ascending=False, inplace=True)

    columns_to_display = [
        "Symbol", "Refined Buy Price", "Candle Entry 2w", "Candle Entry 4w",
        "Candle Entry 6w", "Candle Entry 8w", "Candle Entry 12w", "Candle Entry 18w", "Candle Entry 30w",
        "VWAP Support", "ADX", "Institutional Score",
        "Volume Weight", "Confidence Score", "Sector Correlation",
        "Trend", "Recommendation", "Darvas Breakout %", "Darvas Signal",
        "Rule-Based Buy", "Model-Driven Buy", "Model Probability",
        "Confidence Band", "Tech Fallback Score", "Signal",
        "90D Hit", "90D Gain (%)", "Days to Peak",
        "EMA Uptrend", "EMA21 Slope", "ADX Strength", "MACD Cross", "RSI", "RSI State",
        "OBV Trend", "At BB Lower",
        "Volume Surge", "Near Support", "Signal Score",
        "SMC_Breakout", "Mean_Reversion", "Bullish_Engulfing", "Hammer", "Trend_Strength",
        "Market Stage",
    ]

    # Fill optional columns if missing
    for col in ["Volume Surge", "Near Support", "Signal Score"]:
        if col not in df_summary.columns:
            df_summary[col] = "N/A"

    safe_cols = [c for c in columns_to_display if c in df_summary.columns]
    missing = [c for c in columns_to_display if c not in df_summary.columns]
    if missing:
        print(f"⚠️ Skipping missing columns: {missing}")

    print(df_summary[safe_cols].to_string(index=False))

    if args.csv:
        df_summary[safe_cols].to_csv("predictions_summary.csv", index=False, encoding="utf-8-sig")
        print("\n✅ Predictions saved to predictions_summary.csv")
    else:
        print("\n💡 Skipped saving CSV as per user request")
    # df_summary[safe_cols].to_csv("predictions_summary.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
