from __future__ import annotations

from typing import Any
import numpy as np


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def safe_str(v: Any, default: str = 'N/A') -> str:
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def yn(flag: bool) -> str:
    return 'Y' if bool(flag) else 'N'


def logistic(x: float) -> float:
    try:
        return float(1.0 / (1.0 + np.exp(-float(x))))
    except Exception:
        return 0.5
