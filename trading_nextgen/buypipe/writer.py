from __future__ import annotations

from pathlib import Path
from typing import List
import pandas as pd


def write_output(df: pd.DataFrame, out_path: str, columns: List[str], template_path: str | None = None) -> None:
    out = df.copy()

    # Add only truly missing columns
    for c in columns:
        if c not in out.columns:
            out[c] = pd.NA

    out = out.loc[:, columns]

    # Only fill object (string) columns
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].fillna("N/A")

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="Summary", index=False)