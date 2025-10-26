
# cache_utils.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Tuple
from datetime import datetime, timezone

import pandas as pd

from config import CACHE_DIR

# ---------------------------
# TTL defaults (in minutes)
# ---------------------------
# These are sensible defaults; you can override by passing max_age_minutes to load_cache().
TTL_BY_TIMEFRAME = {
    "hourly": 10,    # refresh frequently
    "daily": 30,     # typical EOD or intraday runs
    "weekly": 24*60, # 1 day
    "monthly": 24*60 # treat like weekly for now
}

def _safe_symbol(symbol) -> str:
    return str(symbol).replace("/", "_").replace(" ", "_").upper()

def _paths(symbol: str, timeframe: str) -> Dict[str, Path]:
    sym = _safe_symbol(symbol)
    base_dir = Path(CACHE_DIR) / sym
    base_dir.mkdir(parents=True, exist_ok=True)
    parquet = base_dir / f"{timeframe}.parquet"
    csv = base_dir / f"{timeframe}.csv"
    meta = base_dir / f"{timeframe}.meta.json"
    return {"dir": base_dir, "parquet": parquet, "csv": csv, "meta": meta}

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _read_meta(meta_path: Path) -> Dict:
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _write_meta(meta_path: Path, meta: Dict) -> None:
    meta_copy = dict(meta)
    meta_copy["updated_at"] = _utcnow_iso()
    meta_path.write_text(json.dumps(meta_copy, indent=2), encoding="utf-8")

def _read_df(parquet_path: Path, csv_path: Path) -> Optional[pd.DataFrame]:
    # Prefer Parquet
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception:
            pass
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception:
            pass
    return None

def _write_df(df: pd.DataFrame, parquet_path: Path, csv_path: Path) -> None:
    # normalize date column name
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception as e:
        print(f"[CACHE] Parquet write failed ({e}); falling back to CSV only.")
    try:
        df.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"[CACHE] CSV write failed: {e}")

def is_cache_stale(symbol: str, timeframe: str, max_age_minutes: Optional[int] = None) -> bool:
    """Return True if cache is missing or older than TTL."""
    p = _paths(symbol, timeframe)
    parquet_path, csv_path, meta_path = p["parquet"], p["csv"], p["meta"]
    # No files -> stale
    if not parquet_path.exists() and not csv_path.exists():
        return True

    meta = _read_meta(meta_path)
    ttl = TTL_BY_TIMEFRAME.get(timeframe, 60) if max_age_minutes is None else max_age_minutes

    # If meta has updated_at, use it; otherwise use file mtime
    updated_at_iso = meta.get("updated_at")
    if updated_at_iso:
        try:
            updated_at = datetime.fromisoformat(updated_at_iso)
        except Exception:
            updated_at = None
    else:
        try:
            mtime = max(parquet_path.stat().st_mtime if parquet_path.exists() else 0,
                        csv_path.stat().st_mtime if csv_path.exists() else 0)
            updated_at = datetime.fromtimestamp(mtime, tz=timezone.utc)
        except Exception:
            updated_at = None

    if updated_at is None:
        return True

    age_min = (datetime.now(timezone.utc) - updated_at).total_seconds() / 60.0
    return age_min > ttl

def load_cache(symbol: str, timeframe: str, max_age_minutes: Optional[int] = None) -> Optional[pd.DataFrame]:
    """
    Load a cached DataFrame if present and not stale (per TTL).
    Returns None if cache is missing or stale.
    """
    if is_cache_stale(symbol, timeframe, max_age_minutes=max_age_minutes):
        return None

    p = _paths(symbol, timeframe)
    df = _read_df(p["parquet"], p["csv"])
    if df is not None and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
    return df

def save_cache(symbol: str, timeframe: str, df_new: pd.DataFrame) -> Path:
    """
    Merge df_new with any existing cache on 'date' column, drop duplicates,
    and write Parquet/CSV + meta. Returns the file path written (Parquet if present).
    """
    p = _paths(symbol, timeframe)
    parquet_path, csv_path, meta_path = p["parquet"], p["csv"], p["meta"]

    # Normalize new data
    df_new = df_new.copy()
    if "date" in df_new.columns:
        df_new["date"] = pd.to_datetime(df_new["date"], errors="coerce")
    # Try to merge with existing
    existing = _read_df(parquet_path, csv_path)
    if existing is not None:
        if "date" in existing.columns:
            existing["date"] = pd.to_datetime(existing["date"], errors="coerce")
        try:
            merged = pd.concat([existing, df_new], ignore_index=True)
            if "date" in merged.columns:
                merged = merged.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last").sort_values("date")
            else:
                merged = merged.drop_duplicates().reset_index(drop=True)
            df_final = merged.reset_index(drop=True)
        except Exception as e:
            print(f"[CACHE] Merge failed, overwriting: {e}")
            df_final = df_new
    else:
        df_final = df_new

    _write_df(df_final, parquet_path, csv_path)

    # Update meta with simple stats
    meta = _read_meta(meta_path)
    meta.update({
        "symbol": _safe_symbol(symbol),
        "timeframe": timeframe,
        "rows": int(len(df_final)),
        "first_date": str(df_final["date"].min()) if "date" in df_final.columns and len(df_final) else None,
        "last_date":  str(df_final["date"].max()) if "date" in df_final.columns and len(df_final) else None,
    })
    _write_meta(meta_path, meta)

    return parquet_path if parquet_path.exists() else csv_path

def clear_cache_file(symbol: str, timeframe: str) -> None:
    """Delete cache files for a given symbol/timeframe (parquet/csv/meta)."""
    p = _paths(symbol, timeframe)
    for key in ("parquet", "csv", "meta"):
        try:
            if p[key].exists():
                p[key].unlink()
        except Exception as e:
            print(f"[CACHE] Failed to remove {p[key]}: {e}")
