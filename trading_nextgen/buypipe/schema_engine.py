from __future__ import annotations

import pandas as pd

from .config_loader import get_columns_config


def apply_output_schema(df: pd.DataFrame) -> pd.DataFrame:
    cfg = get_columns_config()
    rename_map = cfg.get("rename_map", {})
    required = cfg.get("required_columns", [])
    ordered = cfg.get("output_columns", [])

    out = df.copy()
    out = out.rename(columns=rename_map)

    for col in required:
        if col not in out.columns:
            out[col] = pd.NA

    existing = [c for c in ordered if c in out.columns]
    remaining = [c for c in out.columns if c not in existing]
    return out[existing + remaining]