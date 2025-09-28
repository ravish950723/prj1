# train_strong_buy_model.py — time-series friendly (Variant A)
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.calibration import CalibratedClassifierCV
import joblib

# 1) Load ORIGINAL chronological data (no SMOTE)
df = pd.read_csv("train_data.csv")

# Keep time order to avoid leakage
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

X = df.drop(columns=["strong_buy"]).select_dtypes(include="number")
y = df["strong_buy"].astype(int)

# 2) Chronological split with a minimum of 50%/50% and both classes in holdout
split = int(len(df) * 0.8)
while split > int(len(df) * 0.5) and y.iloc[split:].nunique() < 2:
    split -= max(1, int(0.02 * len(df)))  # step back 2% (>=1 row)

X_tr, y_tr = X.iloc[:split], y.iloc[:split]
X_va, y_va = X.iloc[split:], y.iloc[split:]

print("Train dist:", y_tr.value_counts().to_dict(), " | Holdout dist:", y_va.value_counts().to_dict())

# 3) Handle imbalance with class weight
neg = (y_tr == 0).sum()
pos = (y_tr == 1).sum()
scale_pos_weight = neg / max(pos, 1)

base_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=800,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=2.0,
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight,
)

# 4) Fit base model on TRAIN only
base_model.fit(X_tr, y_tr)

# Quick checks @0.30 and @0.50
proba_base = base_model.predict_proba(X_va)[:, 1]
for thr in (0.30, 0.50):
    pred_base = (proba_base >= thr).astype(int)
    print(f"\nHoldout base report @{thr:.2f}:")
    print(classification_report(y_va, pred_base, digits=3, zero_division=0))
print("Holdout AUC (base):", roc_auc_score(y_va, proba_base))
print("Holdout PR-AUC (base):", average_precision_score(y_va, proba_base))

# 5) Probability calibration via CV on TRAIN (preferred)
cal_model = CalibratedClassifierCV(
    estimator=base_model,  # use fitted base model
    method="isotonic",
    cv=3
)
cal_model.fit(X_tr, y_tr)

proba_cal = cal_model.predict_proba(X_va)[:, 1]
for thr in (0.30, 0.50):
    pred_cal = (proba_cal >= thr).astype(int)
    print(f"\nHoldout calibrated report @{thr:.2f}:")
    print(classification_report(y_va, pred_cal, digits=3, zero_division=0))
print("Holdout AUC (cal):", roc_auc_score(y_va, proba_cal))
print("Holdout PR-AUC (cal):", average_precision_score(y_va, proba_cal))

# 6) Preserve feature names for inference
setattr(cal_model, "feature_names_in_", X.columns.to_numpy())

# 7) Save calibrated model
joblib.dump(cal_model, "strong_buy_xgb_model.pkl")
print("✅ Saved calibrated strong_buy_xgb_model.pkl")
