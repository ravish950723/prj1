
"""
RL / Bandit policy for adapting buy thresholds to market regime.

This is *offline* contextual bandit training:
- State: recent features (model probability, trend/volatility/liquidity indicators, macro regime)
- Action: pick a buy threshold bucket (and optionally a strong-buy threshold bucket)
- Reward: encourages precision (avoid false positives) while still capturing true positives

Why this helps "accuracy":
- Your XGB model outputs P(strong_buy). A fixed threshold (e.g., 0.50/0.65) is suboptimal across regimes.
- The bandit learns when to be strict (raise threshold) vs permissive (lower threshold) based on context.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import joblib

try:
    from sklearn.linear_model import Ridge
except Exception:  # pragma: no cover
    Ridge = None


@dataclass
class ThresholdActionSpace:
    buy_thresholds: Tuple[float, ...] = (0.45, 0.50, 0.55, 0.60)
    strong_buy_thresholds: Tuple[float, ...] = (0.60, 0.65, 0.70, 0.75)


def _to_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        xf = float(x)
        if np.isnan(xf) or np.isinf(xf):
            return default
        return xf
    except Exception:
        return default


def build_state_vector(state: Dict[str, float], feature_order: List[str]) -> np.ndarray:
    return np.array([_to_float(state.get(k, 0.0)) for k in feature_order], dtype=float)


class RLThresholdPolicy:
    """
    Stores per-action reward models and chooses thresholds that maximize predicted reward.

    Trained models:
      - buy_models[a] predicts expected reward if we *use* buy_thresholds[a]
      - strong_models[b] predicts expected reward if we *use* strong_buy_thresholds[b]
    """
    def __init__(
        self,
        feature_order: List[str],
        action_space: Optional[ThresholdActionSpace] = None,
        buy_models: Optional[List[object]] = None,
        strong_models: Optional[List[object]] = None,
        meta: Optional[dict] = None,
    ):
        self.feature_order = feature_order
        self.action_space = action_space or ThresholdActionSpace()
        self.buy_models = buy_models or []
        self.strong_models = strong_models or []
        self.meta = meta or {}

    def choose_thresholds(self, state: Dict[str, float]) -> Tuple[float, float]:
        x = build_state_vector(state, self.feature_order).reshape(1, -1)

        buy_thr = self.action_space.buy_thresholds[1]  # default 0.50
        strong_thr = self.action_space.strong_buy_thresholds[1]  # default 0.65

        if self.buy_models:
            preds = np.array([float(m.predict(x)[0]) for m in self.buy_models], dtype=float)
            buy_thr = self.action_space.buy_thresholds[int(np.argmax(preds))]

        if self.strong_models:
            preds = np.array([float(m.predict(x)[0]) for m in self.strong_models], dtype=float)
            strong_thr = self.action_space.strong_buy_thresholds[int(np.argmax(preds))]

        # Safety: strong threshold should not be <= buy threshold
        if strong_thr <= buy_thr:
            strong_thr = min(0.95, buy_thr + 0.10)

        return float(buy_thr), float(strong_thr)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "RLThresholdPolicy":
        return joblib.load(path)


def train_offline_bandit(
    X: np.ndarray,
    y: np.ndarray,
    proba: np.ndarray,
    feature_order: List[str],
    action_space: Optional[ThresholdActionSpace] = None,
    fp_penalty: float = 0.35,
    miss_penalty: float = 0.05,
) -> RLThresholdPolicy:
    """
    Offline bandit training using "counterfactual" rewards derived from labels and model probability.

    Reward design (per sample, per threshold action):
      - If proba >= threshold: we "take trade"
          reward = +1 if y==1 else -fp_penalty
      - Else: we "skip trade"
          reward = 0 if y==0 else -miss_penalty   (penalize missing true positives lightly)

    This pushes the policy to improve precision (accuracy of positives) while not collapsing recall.
    """
    if Ridge is None:
        raise RuntimeError("scikit-learn not available; cannot train RLThresholdPolicy")

    action_space = action_space or ThresholdActionSpace()

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    proba = np.asarray(proba, dtype=float)

    def reward_for_threshold(thr: float) -> np.ndarray:
        take = proba >= thr
        r = np.zeros_like(proba, dtype=float)
        # take
        r[take & (y == 1)] = 1.0
        r[take & (y == 0)] = -float(fp_penalty)
        # skip
        r[(~take) & (y == 1)] = -float(miss_penalty)
        r[(~take) & (y == 0)] = 0.0
        return r

    buy_models = []
    for thr in action_space.buy_thresholds:
        r = reward_for_threshold(thr)
        model = Ridge(alpha=1.0, random_state=0)
        model.fit(X, r)
        buy_models.append(model)

    strong_models = []
    for thr in action_space.strong_buy_thresholds:
        r = reward_for_threshold(thr)
        model = Ridge(alpha=1.0, random_state=0)
        model.fit(X, r)
        strong_models.append(model)

    meta = dict(fp_penalty=float(fp_penalty), miss_penalty=float(miss_penalty))
    return RLThresholdPolicy(
        feature_order=feature_order,
        action_space=action_space,
        buy_models=buy_models,
        strong_models=strong_models,
        meta=meta,
    )
