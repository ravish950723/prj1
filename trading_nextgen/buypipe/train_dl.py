from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

from .train_dataset import build_training_frame, load_source_frame, DEFAULT_FEATURES


class MLP(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def _chrono_split_indices(src: pd.DataFrame, valid_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    n = len(src)
    if n < 10:
        idx = np.arange(n)
        split = max(1, int(n * (1.0 - valid_frac)))
        return idx[:split], idx[split:]

    if "date" in src.columns:
        order = pd.to_datetime(src["date"], errors="coerce")
        sorter = np.argsort(order.fillna(pd.Timestamp.min).to_numpy())
    else:
        sorter = np.arange(n)

    split = int(n * (1.0 - valid_frac))
    split = min(max(split, 1), n - 1)
    train_idx = sorter[:split]
    valid_idx = sorter[split:]
    return train_idx, valid_idx


def train_dl_model(source_path: str, out_dir: str, epochs: int = 30, valid_frac: float = 0.2) -> dict:
    if torch is None:
        metrics = {"enabled": False, "reason": "torch not installed"}
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(out_dir) / "dl_training_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        return metrics

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    src = load_source_frame(source_path)
    df = build_training_frame(src, label_col="strong_buy")

    X = df[DEFAULT_FEATURES].astype(float).to_numpy(dtype=np.float32)
    y = df["strong_buy"].astype(np.float32).to_numpy().reshape(-1, 1)

    train_idx, valid_idx = _chrono_split_indices(src, valid_frac=valid_frac)

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_valid = X[valid_idx]
    y_valid = y[valid_idx]

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-6

    X_train_n = (X_train - mean) / std
    X_valid_n = (X_valid - mean) / std

    model = MLP(X_train_n.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    xt = torch.tensor(X_train_n, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)

    model.train()
    for _ in range(int(epochs)):
        opt.zero_grad()
        logits = model(xt)
        loss = loss_fn(logits, yt)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        train_probs = torch.sigmoid(model(torch.tensor(X_train_n, dtype=torch.float32))).cpu().numpy().reshape(-1)
        valid_probs = torch.sigmoid(model(torch.tensor(X_valid_n, dtype=torch.float32))).cpu().numpy().reshape(-1)

    thr = 0.62
    train_pred = (train_probs >= thr).astype(int)
    valid_pred = (valid_probs >= thr).astype(int)

    train_y = y_train.reshape(-1).astype(int)
    valid_y = y_valid.reshape(-1).astype(int)

    train_precision = float(((train_pred == 1) & (train_y == 1)).sum() / max((train_pred == 1).sum(), 1))
    train_recall = float(((train_pred == 1) & (train_y == 1)).sum() / max((train_y == 1).sum(), 1))

    valid_precision = float(((valid_pred == 1) & (valid_y == 1)).sum() / max((valid_pred == 1).sum(), 1))
    valid_recall = float(((valid_pred == 1) & (valid_y == 1)).sum() / max((valid_y == 1).sum(), 1))

    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_names": DEFAULT_FEATURES,
            "mean": mean,
            "std": std,
        },
        out_path / "dl_lstm_model.pt",
    )

    metrics = {
        "enabled": True,
        "rows_total": int(len(df)),
        "rows_train": int(len(X_train)),
        "rows_valid": int(len(X_valid)),
        "epochs": int(epochs),
        "threshold_used": thr,
        "train_precision": train_precision,
        "train_recall": train_recall,
        "valid_precision": valid_precision,
        "valid_recall": valid_recall,
    }

    with open(out_path / "dl_training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics