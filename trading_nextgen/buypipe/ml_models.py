from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd

from .utils import safe_float


def load_ml_model(project_root: Path):
    for name in ["strong_buy_xgb_model_calibrated.pkl", "strong_buy_xgb_model.pkl"]:
        p = project_root / name
        if p.exists():
            try:
                return joblib.load(p)
            except Exception:
                continue
    return None


def _normalize_row_for_model(row: Dict[str, Any], feats: list[str]) -> Dict[str, float]:
    out = {}
    for f in feats:
        candidates = [
            f,
            f.lower(),
            f.upper(),
            f.replace("_", " "),
            f.replace("_", " ").title(),
        ]
        val = 0.0
        for c in candidates:
            if c in row:
                val = safe_float(row.get(c), 0.0)
                break
        out[f] = val
    return out


def predict_ml_probability(model, row: Dict[str, Any]) -> Tuple[float, str]:
    if model is None:
        return np.nan, "missing_model"
    if not hasattr(model, "predict_proba"):
        return np.nan, "predict_proba_unavailable"

    feats = [str(c) for c in getattr(model, "feature_names_in_", [])]
    if not feats:
        return np.nan, "missing_feature_contract"

    vals = _normalize_row_for_model(row, feats)
    X = pd.DataFrame([vals], columns=feats).astype(float)

    try:
        p = float(np.clip(model.predict_proba(X)[0][1], 0.0, 1.0))
        return p, "ok"
    except Exception as exc:
        return np.nan, f"predict_failed:{exc}"