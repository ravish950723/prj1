import pandas as pd
from xgboost import XGBClassifier
import joblib

# Train with named features
X_df = pd.DataFrame([
    [0.9, 0.8, 0.7, 25, 0.85, 1.1],
    [0.65, 0.4, 0.55, 18, 0.6, 0.9],
    [0.95, 0.7, 0.8, 35, 0.9, 1.2],
    [0.6, 0.3, 0.4, 15, 0.5, 0.8]
], columns=["confidence_score", "sector_corr", "signal_score", "adx", "institutional_score", "volume_weight"])

y = [1, 0, 1, 0]

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_df, y)

joblib.dump(model, "strong_buy_xgb_model.pkl")
print("✅ Model saved with feature names")
