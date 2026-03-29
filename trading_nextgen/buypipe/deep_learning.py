from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from .train_dl import MLP
from .utils import safe_float, logistic

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None


class _LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        logits = self.head(out[:, -1, :])
        return logits


def _sequence_features(df: pd.DataFrame, seq_len: int = 32) -> np.ndarray:
    use_cols = [c for c in ['close', 'volume', 'EMA_20', 'EMA_50', 'EMA_200', 'RSI_14', 'MACD', 'MACD_signal', 'ATR_14'] if c in df.columns]
    if not use_cols:
        return np.zeros((1, seq_len, 1), dtype=np.float32)
    data = df[use_cols].copy().tail(seq_len)
    data = data.apply(pd.to_numeric, errors='coerce').fillna(method='ffill').fillna(0.0)
    arr = data.to_numpy(dtype=np.float32)
    if len(arr) < seq_len:
        pad = np.zeros((seq_len - len(arr), arr.shape[1]), dtype=np.float32)
        arr = np.vstack([pad, arr])
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True) + 1e-6
    arr = (arr - mean) / std
    return arr.reshape(1, seq_len, arr.shape[1])


def predict_dl_probability(project_root: Path, df: pd.DataFrame, row: dict, use_dl: bool = True) -> float:
    if not use_dl:
        return 0.0

    ckpt = project_root / "dl_lstm_model.pt"
    if torch is not None and ckpt.exists():
        try:
            state = torch.load(ckpt, map_location="cpu")
            feature_names = state.get("feature_names", [])
            mean = state.get("mean")
            std = state.get("std")

            vals = np.array(
                [[safe_float(row.get(f, row.get(f.lower(), row.get(f.upper(), 0.0)))) for f in feature_names]],
                dtype=np.float32,
            )
            vals = (vals - mean) / (std + 1e-6)

            model = MLP(vals.shape[1])  # import same model class used in training
            model.load_state_dict(state["state_dict"])
            model.eval()

            with torch.no_grad():
                logits = model(torch.tensor(vals))
                prob = torch.sigmoid(logits).cpu().numpy().reshape(-1)[0]
            return float(np.clip(prob, 0.0, 1.0))
        except Exception:
            pass

    proxy = logistic(
        3.5 * safe_float(row.get("QUANT_COMPOSITE_SCORE"), 0.5) +
        2.0 * safe_float(row.get("Signal Score"), 0.0) +
        1.5 * safe_float(row.get("Confidence Score"), 0.0) +
        0.5 * safe_float(row.get("Institutional Score"), 0.0) -
        0.03 * abs(safe_float(row.get("RSI"), 50.0) - 55.0)
    )
    return float(np.clip(proxy, 0.0, 1.0))