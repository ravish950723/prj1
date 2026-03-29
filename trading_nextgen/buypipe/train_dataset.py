from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUANT_YML = PROJECT_ROOT / "configs" / "quant.yml"


def load_quant_features(path: Path | str = QUANT_YML) -> List[str]:
    p = Path(path)
    feats: List[str] = []
    seen = set()
    for raw in p.read_text(encoding="utf-8").splitlines():
        col = raw.strip()
        if not col:
            continue
        if col not in seen:
            seen.add(col)
            feats.append(col)
    return feats


DEFAULT_FEATURES = load_quant_features()

ALIASES = {
    "open": ["open", "Open"],
    "high": ["high", "High"],
    "low": ["low", "Low"],
    "close": ["close", "Close"],
    "volume": ["volume", "Volume"],

    "EMA_20": ["EMA_20", "EMA20"],
    "EMA_21": ["EMA_21", "EMA21"],
    "EMA_50": ["EMA_50", "EMA50"],
    "EMA_200": ["EMA_200", "EMA200"],

    "ADX_14": ["ADX_14", "ADX14", "ADX"],
    "MACD_signal": ["MACD_signal", "MACD_SIGNAL"],
    "MACD_hist": ["MACD_hist", "MACD_HIST"],
    "RSI_14": ["RSI_14", "RSI"],

    "signal_score": ["signal_score", "Signal Score"],
    "confidence_score": ["confidence_score", "Confidence Score", "CONFIDENCE_SCORE"],
    "institutional_score": ["institutional_score", "Institutional Score"],
    "volume_weight": ["volume_weight", "Volume Weight"],
    "market_stage": ["market_stage", "Market Stage"],
    "market_substage": ["market_substage", "Market Sub-Stage"],
    "substage_confidence": ["substage_confidence", "Substage Confidence"],

    "volume_pressure": ["volume_pressure", "Volume Pressure"],
    "sym_vol_regime": ["sym_vol_regime", "Sym Vol Regime"],
    "VIX_vol_regime": ["VIX_vol_regime", "VIX Vol Regime"],
    "Order Flow Score": ["Order Flow Score"],
    "Institutional Flow Score": ["Institutional Flow Score"],
    "Absorption Score": ["Absorption Score"],
    "Stealth Accumulation Score": ["Stealth Accumulation Score"],
    "Stealth Distribution Score": ["Stealth Distribution Score"],
    "Order Flow Imbalance": ["Order Flow Imbalance"],
    "Buy Pressure": ["Buy Pressure"],
    "Sell Pressure": ["Sell Pressure"],
    "BID_ASK_SPREAD_PCT": ["BID_ASK_SPREAD_PCT"],
    "L2 Imbalance": ["L2 Imbalance"],
    "Dollar Volume": ["Dollar Volume"],
    "Dollar Volume Z20": ["Dollar Volume Z20"],
    "CMF_20": ["CMF_20"],
    "ADL_DELTA": ["ADL_DELTA"],
}


def normalize_feature_columns(df: pd.DataFrame) -> pd.DataFrame:

    if df is None or df.empty:
        return df

    out = df.copy()

    # 🔧 Step 1 — Build reverse lookup map
    reverse_map = {}
    for canon, variants in ALIASES.items():
        for v in variants:
            reverse_map[v] = canon

    # 🔧 Step 2 — Normalize column names
    new_cols = {}
    for col in out.columns:
        c = str(col).strip()

        # direct match
        if c in reverse_map:
            new_cols[col] = reverse_map[c]
            continue

        # case-insensitive match
        cl = c.lower()
        for k in reverse_map:
            if cl == str(k).lower():
                new_cols[col] = reverse_map[k]
                break
        else:
            # fallback → keep original
            new_cols[col] = c

    out.rename(columns=new_cols, inplace=True)

    # 🔧 Step 3 — Ensure all canonical columns exist
    for canon in ALIASES.keys():
        if canon not in out.columns:
            out[canon] = 0.0

    # 🔧 Step 4 — Convert numeric safely
    for c in out.columns:
        try:
            out[c] = pd.to_numeric(out[c], errors="ignore")
        except Exception:
            pass

    return out

def _pick(df: pd.DataFrame, col: str):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    for alt in ALIASES.get(col, []):
        if alt in df.columns:
            return pd.to_numeric(df[alt], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def build_training_frame(df: pd.DataFrame, label_col: str = "strong_buy") -> pd.DataFrame:
    df = normalize_feature_columns(df.copy())
    out = pd.DataFrame(index=df.index)

    # derive one-hot asset-type columns if only ASSET_TYPE exists
    asset_type = pd.Series(df["ASSET_TYPE"], index=df.index) if "ASSET_TYPE" in df.columns else pd.Series("",
                                                                                                          index=df.index)

    if "ASSET_TYPE_STOCK" not in df.columns:
        out["ASSET_TYPE_STOCK"] = asset_type.astype(str).str.upper().eq("STOCK").astype(int)

    if "ASSET_TYPE_ETF" not in df.columns:
        out["ASSET_TYPE_ETF"] = asset_type.astype(str).str.upper().eq("ETF").astype(int)

    for col in DEFAULT_FEATURES:
        if col in ("ASSET_TYPE_STOCK", "ASSET_TYPE_ETF") and col in out.columns:
            continue
        out[col] = _pick(df, col)

    if label_col in df.columns:
        out[label_col] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
    else:
        gain = pd.to_numeric(df.get("90D Gain (%)"), errors="coerce").fillna(0.0)
        d2p = pd.to_numeric(df.get("Days to Peak"), errors="coerce").fillna(9999)
        out[label_col] = ((gain >= 15.0) & (d2p <= 45)).astype(int)

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out[label_col] = out[label_col].astype(int)
    return out


def load_source_frame(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(p, sheet_name=0)
    return pd.read_csv(p)