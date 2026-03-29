from __future__ import annotations

from typing import Tuple
import pandas as pd

from .fetching import fetch_data_daily_with_fallback


def fetch_symbol_daily(symbol: str, ttl_minutes: int = 240, force_refresh: bool = False) -> Tuple[pd.DataFrame, str, str | None]:
    df, source, err = fetch_data_daily_with_fallback(
        symbol=symbol,
        bar_spec='10 Y',
        bar_size='1 day',
        ttl_minutes=ttl_minutes,
        require_today=False,
        force_refresh=force_refresh,
    )
    return normalize_ohlcv(df), source, err


def normalize_ohlcv(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or getattr(df, 'empty', True):
        return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    out = df.copy()
    cmap = {str(c).lower(): c for c in out.columns}
    rename = {}
    for src, dst in [('date', 'date'), ('datetime', 'date'), ('time', 'date'), ('open', 'open'), ('high', 'high'), ('low', 'low'), ('close', 'close'), ('volume', 'volume')]:
        if src in cmap:
            rename[cmap[src]] = dst
    out = out.rename(columns=rename)
    for col in ['date', 'open', 'high', 'low', 'close', 'volume']:
        if col not in out.columns:
            out[col] = pd.NA
    out['date'] = pd.to_datetime(out['date'], errors='coerce')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        out[col] = pd.to_numeric(out[col], errors='coerce')
    out = out.dropna(subset=['date', 'open', 'high', 'low', 'close']).sort_values('date').drop_duplicates(subset=['date'], keep='last').reset_index(drop=True)
    return out
