import pandas as pd
import numpy as np
from compute import compute_indicators
from fetching import fetch_data_cached
from config import symbols as target_symbols

training_rows = []
window_size = 60  # rolling window size

for symbol in target_symbols:
    df = fetch_data_cached(symbol, duration="3 Y", bar_size="1 day")
    df = compute_indicators(df, symbol)
    df = df.dropna()

    # Ensure all expected model features are present
    try:
        with open("model_features.txt") as f:
            expected_features = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        expected_features = None  # Proceed anyway if not present

    for i in range(window_size, len(df) - 10):
        slice_df = df.iloc[i - window_size:i].copy()

        entry_price = df.iloc[i]['close']
        future_window = df.iloc[i+1:i+31]  # next 30 bars

        if len(future_window) < 10:
            continue  # not enough data for forward view

        max_gain = ((future_window['high'].max() - entry_price) / entry_price) * 100
        max_drawdown = ((entry_price - future_window['low'].min()) / entry_price) * 100

        ema_uptrend = df.iloc[i]['EMA_21'] > df.iloc[i]['EMA_50']
        volume_confirm = df.iloc[i]['volume_weight'] > 1
        adx_strong = df.iloc[i]['ADX_14'] >= 20

        strong_buy = (
            max_gain >= 15 and
            max_drawdown <= 10 and
            ema_uptrend and
            volume_confirm and
            adx_strong
        )

        features = df.iloc[i].copy()

        # Drop non-numeric or non-feature columns if present
        for col in ['recommendation', 'HTF_Trend', 'ITF_Trend', 'LTF_Trend']:
            if col in features:
                del features[col]

        # Add missing expected features as 0.0
        if expected_features:
            for col in expected_features:
                if col not in features:
                    features[col] = 0.0

        features['strong_buy'] = int(strong_buy)
        training_rows.append(features)

train_df = pd.DataFrame(training_rows)
train_df.to_csv("train_data.csv", index=False)
print(f"✅ Training data saved: {len(train_df)} rows")

if train_df.empty:
    print("⚠️ train_df is empty — no rows generated. Skipping stats/SMOTE.")
    import sys
    sys.exit(0)


# === Optional: Print label distribution and feature correlation ===
label_dist = train_df['strong_buy'].value_counts(normalize=True) * 100
print("\n🔍 Label Distribution (%):\n", label_dist)

# Only use numeric features for correlation
numeric_features = train_df.select_dtypes(include='number')
correlations = numeric_features.drop(columns=['strong_buy']).corrwith(numeric_features['strong_buy']).sort_values(key=abs, ascending=False)
print("\n📊 Top Correlated Features:\n", correlations.head(10))

# === Apply SMOTE for class balancing ===
try:
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import StandardScaler

    numeric_cols = train_df.drop(columns=['strong_buy']).select_dtypes(include='number').columns
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_df[numeric_cols])
    y = train_df['strong_buy']

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    balanced_df = pd.DataFrame(X_resampled, columns=numeric_cols)
    balanced_df['strong_buy'] = y_resampled
    balanced_df.to_csv("train_data_balanced.csv", index=False)
    print(f"✅ SMOTE applied: {len(balanced_df)} balanced rows saved to train_data_balanced.csv")

    # Save feature list for inference
    with open("model_features.txt", "w") as f:
        for col in numeric_cols:
            f.write(col + "\n")
except ImportError:
    print("⚠️ imbalanced-learn (imblearn) is not installed. Skipping SMOTE step.")
