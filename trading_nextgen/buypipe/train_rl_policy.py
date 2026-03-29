
from __future__ import annotations
# from .train_dataset import normalize_feature_columns


import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from rl_policy import train_offline_bandit, ThresholdActionSpace

HERE = Path(__file__).resolve().parent

DEFAULT_MODEL = HERE / "strong_buy_xgb_model_calibrated.pkl"
DEFAULT_DATA = HERE / "train_data.csv"
DEFAULT_FEATURES = HERE / "model_features.txt"
DEFAULT_OUT = HERE / "rl_threshold_policy.pkl"


def _load_feature_order(path: Path) -> list[str]:
    txt = path.read_text().strip().splitlines()
    feats = [t.strip() for t in txt if t.strip()]
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--features", default=str(DEFAULT_FEATURES))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--fp-penalty", type=float, default=0.35, help="Penalty for false positives (took trade when label=0)")
    ap.add_argument("--miss-penalty", type=float, default=0.05, help="Penalty for missed positives (skipped when label=1)")
    args = ap.parse_args()

    data_path = Path(args.data)
    model_path = Path(args.model)
    feats_path = Path(args.features)
    out_path = Path(args.out)

    df = pd.read_csv(data_path)
    if "strong_buy" not in df.columns:
        raise SystemExit("train_data.csv must include 'strong_buy' label column")

    # Load model (calibrated preferred). If your environment can't unpickle calibrated model,
    # fall back to the raw XGB model.
    try:
        model = joblib.load(model_path)
    except Exception as e:
        fallback = model_path.with_name("strong_buy_xgb_model.pkl")
        if fallback.exists():
            print(f"[WARN] Failed to load {model_path.name} ({e}). Falling back to {fallback.name}")
            model = joblib.load(fallback)
        else:
            raise

    # Determine feature order:
    # 1) Prefer model.feature_names_in_ if present (source of truth)
    # 2) Else use model_features.txt
    if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
        base_feature_order = [str(x) for x in list(model.feature_names_in_)]
    else:
        base_feature_order = _load_feature_order(feats_path)

    feature_order = base_feature_order + ["MODEL_PROBA"]

    # Drop empty rows (some pipelines leave a blank first row)
    df = df.dropna(how="all").copy()

    # Ensure all feature columns exist; fill missing with 0.
    # Also support common aliases from your generator:
    #  - adx <- ADX_14
    #  - sector_corr <- sector_outperformance (proxy)
    aliases = {
        "adx": ["ADX_14", "ADX14", "adx_14"],
        "sector_corr": ["sector_outperformance", "sector_relative_strength"],
    }
    for c in base_feature_order:
        if c in df.columns:
            continue
        # try aliases
        if c in aliases:
            for alt in aliases[c]:
                if alt in df.columns:
                    df[c] = df[alt]
                    break
        if c not in df.columns:
            df[c] = 0.0

    X_base = df[base_feature_order].astype(float).fillna(0.0).to_numpy()

    y = df["strong_buy"].astype(int).fillna(0).to_numpy()

        # calibrated model could be a CalibratedClassifierCV or similar.
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_base)[:, 1]
    else:
        raise SystemExit("Model does not support predict_proba()")

    X = np.hstack([X_base, proba.reshape(-1,1)])

    policy = train_offline_bandit(
        X=X,
        y=y,
        proba=proba,
        feature_order=feature_order,
        action_space=ThresholdActionSpace(),
        fp_penalty=args.fp_penalty,
        miss_penalty=args.miss_penalty,
    )

    policy.save(str(out_path))
    print(f"[OK] Saved RL threshold policy: {out_path}")
    print(f"[INFO] Meta: {policy.meta}")


if __name__ == "__main__":
    main()
