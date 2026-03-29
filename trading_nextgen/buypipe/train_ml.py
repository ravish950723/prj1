from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "train_data.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = BASE_DIR / "cache"
EPS_CACHE_DIR = CACHE_DIR / "eps"
EPS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "ml_model.pkl"
THRESHOLD_PATH = MODEL_DIR / "thresholds.json"
METRICS_PATH = MODEL_DIR / "training_metrics.json"
FEATURES_PATH = MODEL_DIR / "feature_columns.json"
DEFAULT_VALID_RATIO = 0.20
MIN_PRECISION_TARGET = 0.50
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)
LABEL_CANDIDATES = ["strong_buy", "label", "target", "strong_buy_label", "y", "target_15pct_45d"]
DATE_CANDIDATES = ["date", "datetime", "timestamp", "bar_date", "trade_date"]
KNOWN_LEAKAGE_COLUMNS = {
    "label", "target", "y", "strong_buy_label", "target_15pct_45d", "buy", "avoid",
    "future_return", "future_ret", "future_pct_return", "future_gain", "future_profit",
    "future_high", "future_low", "future_close", "future_open", "future_max_high",
    "future_min_low", "future_peak", "future_trough", "future_return_max", "future_drawdown_min",
    "days_to_peak", "days_to_target", "hit_15pct_45d", "achieved_target", "next_close",
    "next_return", "next_day_return", "fwd_return", "forward_return", "forward_ret",
    "lookahead_return", "max_return_next_", "min_return_next_",
}
IDENTIFIER_COLUMNS = {"symbol", "ticker", "sector", "industry", "company", "name", "cusip", "isin", "row_id", "id"}
ENABLE_DL = False


def build_base_estimator():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85,
            min_child_weight=3, reg_alpha=0.2, reg_lambda=1.5, objective="binary:logistic",
            eval_metric="logloss", random_state=42, n_jobs=4,
        ), "xgboost"
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_depth=8, learning_rate=0.05, max_iter=500, random_state=42), "hist_gradient_boosting"


@dataclass
class ThresholdResult:
    thr: float
    precision: float
    recall: float
    f1: float
    predicted_positives: int


@dataclass
class MLReport:
    model_type: str
    rows_total: int
    rows_train: int
    rows_valid: int
    positive_rate_total: float
    positive_rate_train: float
    positive_rate_valid: float
    threshold_used: float
    train_precision: float
    train_recall: float
    train_f1: float
    valid_precision: float
    valid_recall: float
    valid_f1: float
    valid_auc: Optional[float]
    valid_pr_auc: Optional[float]
    best_f1_threshold: Dict
    best_precision_target_threshold: Dict
    best_recall_threshold: Dict
    leakage_columns_removed: List[str]
    feature_count: int


def detect_label_column(df: pd.DataFrame) -> str:
    for col in LABEL_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(f"No label column found. Expected one of: {LABEL_CANDIDATES}")


def detect_date_column(df: pd.DataFrame) -> str:
    for col in DATE_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(f"No date column found for time-based split. Expected one of: {DATE_CANDIDATES}")


def safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=False)


def is_probably_leakage(col: str) -> bool:
    c = col.lower().strip()
    if c in KNOWN_LEAKAGE_COLUMNS:
        return True
    bad_tokens = ["future", "forward", "fwd", "next_", "nextday", "next_day", "lookahead", "lead_", "lead", "target", "label", "outcome", "achieved", "days_to_"]
    return any(tok in c for tok in bad_tokens)


def is_feature_column(col: str, label_col: str, date_col: str) -> bool:
    c = col.lower().strip()
    if col == label_col or col == date_col:
        return False
    if c in IDENTIFIER_COLUMNS:
        return False
    if is_probably_leakage(c):
        return False
    return True


def get_numeric_feature_frame(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)
    numeric_cols = []
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            numeric_cols.append(col)
        else:
            coerced = pd.to_numeric(X[col], errors="coerce")
            if coerced.notna().mean() > 0.90:
                X[col] = coerced
                numeric_cols.append(col)
    return X[numeric_cols]


def time_based_split(df: pd.DataFrame, date_col: str, valid_ratio: float = DEFAULT_VALID_RATIO) -> Tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work[date_col] = safe_to_datetime(work[date_col])
    work = work.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    split_idx = max(1, int(len(work) * (1.0 - valid_ratio)))
    split_idx = min(split_idx, len(work) - 1)
    return work.iloc[:split_idx].copy(), work.iloc[split_idx:].copy()


def sanitize_target(y: pd.Series) -> pd.Series:
    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
    return y.clip(lower=0, upper=1)


def compute_binary_metrics(y_true: np.ndarray, probs: np.ndarray, thr: float) -> Dict[str, float]:
    preds = (probs >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1": float(f1), "predicted_positives": int(preds.sum())}


def evaluate_thresholds(y_true: np.ndarray, probs: np.ndarray) -> Tuple[ThresholdResult, ThresholdResult, ThresholdResult]:
    best_f1 = None
    best_precision_target = None
    best_recall = None
    rows = []
    for thr in THRESHOLD_GRID:
        m = compute_binary_metrics(y_true, probs, float(thr))
        row = ThresholdResult(float(thr), m["precision"], m["recall"], m["f1"], m["predicted_positives"])
        rows.append(row)
        if best_f1 is None or row.f1 > best_f1.f1:
            best_f1 = row
        if best_recall is None or row.recall > best_recall.recall or (math.isclose(row.recall, best_recall.recall) and row.precision > best_recall.precision):
            best_recall = row
        if row.precision >= MIN_PRECISION_TARGET:
            if best_precision_target is None or (row.f1, row.recall, -row.thr) > (best_precision_target.f1, best_precision_target.recall, -best_precision_target.thr):
                best_precision_target = row
    if best_precision_target is None:
        best_precision_target = max(rows, key=lambda x: (x.precision, x.f1, x.recall, -x.thr))
    return best_f1, best_precision_target, best_recall


def save_json(path: Path, data: dict):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def build_ml_pipeline(base_estimator):
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", CalibratedClassifierCV(estimator=base_estimator, method="sigmoid", cv=3))])


def train() -> dict:
    df = pd.read_csv(DATA_PATH)
    if df.empty:
        raise ValueError("Training data is empty.")
    label_col = detect_label_column(df)
    date_col = detect_date_column(df)
    df[label_col] = sanitize_target(df[label_col])
    df[date_col] = safe_to_datetime(df[date_col])
    df = df.dropna(subset=[date_col]).copy()

    candidate_cols = [c for c in df.columns if is_feature_column(c, label_col=label_col, date_col=date_col)]
    X_full = get_numeric_feature_frame(df, candidate_cols)
    if X_full.empty:
        raise ValueError("No numeric feature columns remain after leakage guard.")
    kept_feature_cols = list(X_full.columns)
    removed_cols = sorted([c for c in df.columns if c not in kept_feature_cols and c not in {label_col, date_col}])

    work = df[[date_col, label_col]].join(X_full).copy()
    train_df, valid_df = time_based_split(work, date_col=date_col, valid_ratio=DEFAULT_VALID_RATIO)
    X_train, y_train = train_df[kept_feature_cols].copy(), sanitize_target(train_df[label_col])
    X_valid, y_valid = valid_df[kept_feature_cols].copy(), sanitize_target(valid_df[label_col])
    if y_train.nunique() < 2 or y_valid.nunique() < 2:
        raise ValueError("Training or validation split has only one class.")

    base_estimator, model_name = build_base_estimator()
    ml_pipeline = build_ml_pipeline(base_estimator)
    ml_pipeline.fit(X_train, y_train)
    train_probs = ml_pipeline.predict_proba(X_train)[:, 1]
    valid_probs = ml_pipeline.predict_proba(X_valid)[:, 1]
    best_f1_thr, best_precision_target_thr, best_recall_thr = evaluate_thresholds(y_valid.values, valid_probs)

    # 🔥 robust threshold selection
    if best_precision_target_thr.predicted_positives > 0:
        threshold_used = float(best_precision_target_thr.thr)
    elif best_recall_thr.predicted_positives > 0:
        threshold_used = float(best_recall_thr.thr)
    else:
        # fallback: percentile-based threshold to force some coverage
        threshold_used = float(np.percentile(valid_probs, 97))

    # final safety clamp
    threshold_used = float(np.clip(threshold_used, 0.05, 0.90))

    print(
        f"[THRESHOLD DEBUG] used={threshold_used:.4f} | "
        f"best_f1={best_f1_thr.thr:.4f}/{best_f1_thr.predicted_positives} | "
        f"best_precision={best_precision_target_thr.thr:.4f}/{best_precision_target_thr.predicted_positives} | "
        f"best_recall={best_recall_thr.thr:.4f}/{best_recall_thr.predicted_positives}"
    )

    train_m = compute_binary_metrics(y_train.values, train_probs, threshold_used)
    valid_m = compute_binary_metrics(y_valid.values, valid_probs, threshold_used)
    valid_auc = float(roc_auc_score(y_valid.values, valid_probs))
    valid_pr_auc = float(average_precision_score(y_valid.values, valid_probs))

    ml_report = MLReport(
        model_type=f"{model_name}_calibrated",
        rows_total=int(len(work)), rows_train=int(len(train_df)), rows_valid=int(len(valid_df)),
        positive_rate_total=float(work[label_col].mean()), positive_rate_train=float(y_train.mean()), positive_rate_valid=float(y_valid.mean()),
        threshold_used=threshold_used,
        train_precision=float(train_m["precision"]), train_recall=float(train_m["recall"]), train_f1=float(train_m["f1"]),
        valid_precision=float(valid_m["precision"]), valid_recall=float(valid_m["recall"]), valid_f1=float(valid_m["f1"]),
        valid_auc=valid_auc, valid_pr_auc=valid_pr_auc,
        best_f1_threshold=asdict(best_f1_thr), best_precision_target_threshold=asdict(best_precision_target_thr), best_recall_threshold=asdict(best_recall_thr),
        leakage_columns_removed=removed_cols, feature_count=len(kept_feature_cols),
    )

    joblib.dump(ml_pipeline, MODEL_PATH)
    save_json(THRESHOLD_PATH, {"threshold": threshold_used, "min_precision_target": MIN_PRECISION_TARGET, "eps_cache_dir": str(EPS_CACHE_DIR), "date_split_column": date_col, "label_column": label_col})
    save_json(FEATURES_PATH, {"feature_columns": kept_feature_cols, "label_column": label_col, "date_column": date_col, "eps_cache_dir": str(EPS_CACHE_DIR)})
    final_out = {"ml": asdict(ml_report), "dl": {"enabled": False, "note": "DL disabled until label quality improves."}}
    save_json(METRICS_PATH, final_out)
    print(final_out)
    return final_out


def train_ml_model(source_path: str | None = None, out_dir: str | None = None, threshold: float | None = None,
                   valid_frac: float = DEFAULT_VALID_RATIO) -> dict:
    global DATA_PATH, MODEL_DIR, MODEL_PATH, THRESHOLD_PATH, METRICS_PATH, FEATURES_PATH
    if source_path:
        DATA_PATH = Path(source_path)
    if out_dir:
        MODEL_DIR = Path(out_dir)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_PATH = MODEL_DIR / "ml_model.pkl"
        THRESHOLD_PATH = MODEL_DIR / "thresholds.json"
        METRICS_PATH = MODEL_DIR / "training_metrics.json"
        FEATURES_PATH = MODEL_DIR / "feature_columns.json"
    return train()


if __name__ == "__main__":
    train_ml_model()
