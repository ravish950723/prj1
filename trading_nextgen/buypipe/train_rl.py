from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np

from .train_dataset import build_training_frame, load_source_frame, DEFAULT_FEATURES


def train_rl_policy(source_path: str, out_dir: str) -> dict:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    src = load_source_frame(source_path)
    df = build_training_frame(src, label_col='strong_buy')

    try:
        from rl_policy import train_offline_bandit, ThresholdActionSpace
    except Exception:
        metrics = {'enabled': False, 'reason': 'rl_policy.py not available in PYTHONPATH'}
        with open(out_path / 'rl_training_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        return metrics

    X_base = df[DEFAULT_FEATURES].astype(float).to_numpy()
    y = df['strong_buy'].astype(int).to_numpy()

    # model probability proxy for bandit training if ML model not yet present
    proba = np.clip(
        0.15
        + 0.25 * np.tanh(df['QUANT_COMPOSITE_SCORE'].astype(float).to_numpy())
        + 0.20 * np.tanh(df['Signal Score'].astype(float).to_numpy())
        + 0.20 * np.tanh(df['Confidence Score'].astype(float).to_numpy())
        + 0.20 * np.tanh(df['Institutional Score'].astype(float).to_numpy()),
        0.01,
        0.99,
    )

    feature_order = DEFAULT_FEATURES + ['MODEL_PROBA']
    X = np.hstack([X_base, proba.reshape(-1, 1)])
    policy = train_offline_bandit(
        X=X,
        y=y,
        proba=proba,
        feature_order=feature_order,
        action_space=ThresholdActionSpace(),
        fp_penalty=0.30,
        miss_penalty=0.08,
    )
    joblib.dump(policy, out_path / 'rl_threshold_policy.pkl')
    metrics = {'enabled': True, 'rows': int(len(df)), 'feature_count': len(feature_order)}
    with open(out_path / 'rl_training_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    return metrics
