from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _safe_num(v, default=np.nan):
    try:
        if v is None or v == "":
            return default
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def derive_best_risk_reward_frame(df: pd.DataFrame) -> pd.Series:
    candidates = []
    for col in ["Best_Risk_Reward", "LONG_RR_RATIO", "SHORT_RR_RATIO", "Risk/Reward T1", "Risk/Reward T2"]:
        if col in df.columns:
            candidates.append(pd.to_numeric(df[col], errors="coerce"))
    if {"Reward_%", "Risk_%"}.issubset(df.columns):
        risk = pd.to_numeric(df["Risk_%"], errors="coerce")
        reward = pd.to_numeric(df["Reward_%"], errors="coerce")
        rr = reward / risk.replace(0, np.nan)
        candidates.append(rr)
    if not candidates:
        return pd.Series(np.nan, index=df.index, name="Best_Risk_Reward")
    out = pd.concat(candidates, axis=1).max(axis=1, skipna=True)
    out.name = "Best_Risk_Reward"
    return out


def add_rr_smart_money_training_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Best_Risk_Reward"] = derive_best_risk_reward_frame(out)

    order_flow = pd.to_numeric(out.get("Order Flow Score", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    inst_flow = pd.to_numeric(out.get("Institutional Flow Score", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    absorption = pd.to_numeric(out.get("Absorption Score", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)

    out["Smart_Money_Flow"] = 0.45 * order_flow + 0.35 * inst_flow + 0.20 * absorption
    out["RR_x_SmartMoney"] = out["Best_Risk_Reward"].clip(lower=0).fillna(0.0) * out["Smart_Money_Flow"]
    out["RR_Quality_Num"] = np.select(
        [
            out["Best_Risk_Reward"] >= 3.0,
            out["Best_Risk_Reward"] >= 2.0,
            out["Best_Risk_Reward"] >= 1.5,
        ],
        [1.0, 0.75, 0.5],
        default=0.0,
    )

    current = pd.to_numeric(out.get("Current Price"), errors="coerce")
    entry = pd.to_numeric(out.get("ML Entry Target", out.get("Refined Buy Price")), errors="coerce")
    atr = pd.to_numeric(out.get("ATR14"), errors="coerce")
    out["Entry_Efficiency"] = ((current - entry) / atr.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    out["Entry_Efficiency"] = out["Entry_Efficiency"].clip(-5, 5)

    # Safe fill for model ingestion
    for col in ["Best_Risk_Reward", "Smart_Money_Flow", "RR_x_SmartMoney", "RR_Quality_Num", "Entry_Efficiency"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    return out


def rr_smart_money_feature_columns(existing: Iterable[str] | None = None) -> list[str]:
    base = list(existing or [])
    extra = [
        "Best_Risk_Reward",
        "Smart_Money_Flow",
        "RR_x_SmartMoney",
        "RR_Quality_Num",
        "Entry_Efficiency",
    ]
    for c in extra:
        if c not in base:
            base.append(c)
    return base
