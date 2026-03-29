from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_FORCE_REFRESH,
    DEFAULT_OUT,
    DEFAULT_SCHEMA,
    DEFAULT_SYMBOL_LIMIT,
    DEFAULT_TTL_MINUTES,
    DEFAULT_USE_DL,
    DEFAULT_USE_ML,
    DEFAULT_USE_RL,
    symbols as config_symbols,
)
from .ib_level2 import IBLevel2Config, fetch_level2_snapshot, merge_level2_into_row
from .schema import align_row_to_schema, build_default_row, load_schema_config
from .scoring import apply_scoring
from .writer import write_output


import numpy as np
import pandas as pd

def _is_default_like(v) -> bool:
    # None
    if v is None:
        return True

    # pandas / numpy missing
    try:
        if pd.isna(v) and not isinstance(v, (list, tuple, dict, np.ndarray, pd.Series)):
            return True
    except Exception:
        pass

    # strings
    if isinstance(v, str):
        return v.strip() in {"", "N/A", "NA", "NONE", "NULL"}

    # bool
    if isinstance(v, (bool, np.bool_)):
        return v is False

    # numbers
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v) == 0.0

    # numpy arrays
    if isinstance(v, np.ndarray):
        if v.size == 0:
            return True
        try:
            flat = v.reshape(-1)
            return all(_is_default_like(x) for x in flat.tolist())
        except Exception:
            return False

    # pandas series
    if isinstance(v, pd.Series):
        if v.empty:
            return True
        try:
            return all(_is_default_like(x) for x in v.tolist())
        except Exception:
            return False

    # containers
    if isinstance(v, (list, tuple)):
        if len(v) == 0:
            return True
        return all(_is_default_like(x) for x in v)

    if isinstance(v, dict):
        return len(v) == 0

    return False


def _non_default_keys(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        try:
            if not _is_default_like(v):
                out[k] = v
        except Exception:
            # never let debug logging crash pipeline
            out[k] = f"<unprintable:{type(v).__name__}>"
    return out



def _load_symbols(template: str | None) -> List[str]:
    syms = [str(s).strip().upper() for s in (config_symbols or []) if str(s).strip()]
    if template and Path(template).exists():
        try:
            xls = pd.read_excel(template, sheet_name=0)
            if "Symbol" in xls.columns:
                syms.extend(
                    [
                        str(s).strip().upper()
                        for s in xls["Symbol"].dropna().tolist()
                        if str(s).strip()
                    ]
                )
        except Exception:
            pass
    return list(dict.fromkeys(syms))


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if value == "":
            return default
        return float(value)
    except Exception:
        return default




def _is_exportable_row(row: dict) -> tuple[bool, str]:
    data_source = str(row.get("DATA_SOURCE") or "").strip().upper()
    current_price = _safe_float(row.get("Current Price"), 0.0)

    if data_source == "ERROR":
        return False, "DATA_SOURCE=ERROR"

    if current_price <= 0:
        return False, "Current Price <= 0"

    return True, ""


def run_pipeline(
    template: str,
    schema_path: str,
    out_path: str,
    ttl_minutes: int = 240,
    force_refresh: bool = False,
    limit: int = 0,
    use_ml: bool = True,
    use_dl: bool = True,
    use_rl: bool = True,
    use_level2: bool = True,
) -> pd.DataFrame:
    project_root = Path(__file__).resolve().parent.parent

    schema_cfg = load_schema_config(schema_path)
    columns = schema_cfg["columns"]
    columns_meta = schema_cfg["columns_meta"]
    rename_map = schema_cfg.get("rename_map", {})

    default_row = build_default_row(columns_meta)
    rows: list[dict] = []

    # single source of truth
    from .build_master_features import build_master_rows

    master_rows = build_master_rows(
        template=template,
        ttl_minutes=ttl_minutes,
        force_refresh=force_refresh,
        limit=limit,
    )

    l2_cfg = IBLevel2Config()

    for idx, (base_row, df) in enumerate(master_rows, start=1):
        # Reset all per-symbol state inside the loop
        row = dict(default_row)
        l2_data = {}
        symbol = ""

        try:
            if isinstance(base_row, dict):
                row.update(base_row)

            symbol = str(row.get("Symbol", "")).strip().upper()

            if not symbol:
                row["DATA_SOURCE"] = "ERROR"
                row["Error Detail"] = "Missing Symbol in base_row"
                print(f"[ROW {idx}] Missing Symbol in base_row; skipping")
                continue

            print(f"\n{'=' * 70}")
            print(f"[{idx}] PROCESSING SYMBOL: {symbol}")
            print(f"{'=' * 70}")

            # Level 2 enrichment
            if use_level2:
                try:
                    l2 = fetch_level2_snapshot(symbol, l2_cfg)
                    l2_data = merge_level2_into_row(row, l2) or {}

                    if isinstance(l2_data, dict):
                        row.update({k: v for k, v in l2_data.items() if v is not None})

                    row["l2_status"] = (
                        "ok" if l2.get("L2 Available") else f"unavailable:{l2.get('L2 Error', '')}"
                    )
                except Exception as exc:
                    row["l2_status"] = f"failed:{exc}"
                    print(f"[{symbol}] Level2 FAILED: {exc}")
                    traceback.print_exc()

            # Scoring / ML / DL / RL
            try:
                row = apply_scoring(
                    project_root,
                    row,
                    df,
                    use_ml=use_ml,
                    use_dl=use_dl,
                    use_rl=use_rl,
                )
            except Exception as exc:
                row["DATA_SOURCE"] = "ERROR"
                row["Error Detail"] = f"apply_scoring failed: {exc}"
                print(f"[{symbol}] apply_scoring FAILED: {exc}")
                traceback.print_exc()
                continue

            # Rename fields after scoring
            row = {rename_map.get(k, k): v for k, v in row.items()}

            # Final schema alignment
            row = align_row_to_schema(row, columns_meta)

            # Diagnostics
            print(f"[{symbol}] DATA_SOURCE: {row.get('DATA_SOURCE')}")
            print(f"[{symbol}] Current Price: {row.get('Current Price')}")
            print(f"[{symbol}] Refined Buy Price: {row.get('Refined Buy Price')}")
            print(f"[{symbol}] Market Stage: {row.get('Market Stage')}")
            print(f"[{symbol}] Market Sub-Stage: {row.get('Market Sub-Stage')}")
            print(f"[{symbol}] Model Probability: {row.get('Model Probability')}")
            print(f"[{symbol}] L2 Status: {row.get('l2_status')}")
            print(f"[{symbol}] ROW NON-DEFAULT KEYS: {_non_default_keys(row)}")
            print(f"[{symbol}] TOTAL ROWS BUILT SO FAR: {len(rows)}")

            exportable, reason = _is_exportable_row(row)
            if not exportable:
                print(f"[SKIP] {symbol}: {reason}; not exporting default/incomplete row")
                continue

            rows.append(row)
            print(f"[APPEND] {symbol}: row added successfully")

        except Exception as exc:
            print(f"[{symbol or f'ROW-{idx}'}] UNHANDLED PIPELINE ERROR: {exc}")
            traceback.print_exc()
            continue

    out = pd.DataFrame(rows)

    if out.empty:
        print("No valid rows were built. Output DataFrame is empty.")
    else:
        print("\nFINAL OUTPUT PREVIEW:")
        print(out.head(3).T)

    write_output(out, out_path=out_path, columns=columns, template_path=template)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default=str(Path("/mnt/data/predictions_summary_out.xlsx")))
    ap.add_argument("--schema", default=DEFAULT_SCHEMA)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--ttl", type=int, default=DEFAULT_TTL_MINUTES)
    ap.add_argument("--refresh", action="store_true", default=DEFAULT_FORCE_REFRESH)
    ap.add_argument("--limit", type=int, default=DEFAULT_SYMBOL_LIMIT)
    ap.add_argument("--no-ml", action="store_true")
    ap.add_argument("--no-dl", action="store_true")
    ap.add_argument("--no-rl", action="store_true")
    ap.add_argument("--no-level2", action="store_true")
    args = ap.parse_args()

    run_pipeline(
        template=args.template,
        schema_path=args.schema,
        out_path=args.out,
        ttl_minutes=args.ttl,
        force_refresh=args.refresh,
        limit=args.limit,
        use_ml=DEFAULT_USE_ML and not args.no_ml,
        use_dl=DEFAULT_USE_DL and not args.no_dl,
        use_rl=DEFAULT_USE_RL and not args.no_rl,
        use_level2=not args.no_level2,
    )

    print(f"Saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())