from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

BASE_DIR = Path(__file__).resolve().parent
CACHE_ROOT = BASE_DIR / "cache"
EPS_CACHE_DIR = CACHE_ROOT / "eps"
SENTIMENT_CACHE_DIR = CACHE_ROOT / "sentiment"
for _p in (CACHE_ROOT, EPS_CACHE_DIR, SENTIMENT_CACHE_DIR):
    _p.mkdir(parents=True, exist_ok=True)

_MEM_CACHE: dict[str, dict] = {}


def _cache_key(cache_dir: Path, symbol: str) -> str:
    return f"{cache_dir.name}:{symbol.upper().strip()}"


def _cache_file(cache_dir: Path, symbol: str) -> Path:
    return cache_dir / f"{symbol.upper().strip()}.json"


def load_cached_payload(cache_dir: Path, symbol: str, ttl_hours: float | None = None) -> Optional[Dict[str, Any]]:
    symbol = symbol.upper().strip()
    key = _cache_key(cache_dir, symbol)
    if key in _MEM_CACHE:
        return _MEM_CACHE[key]

    fp = _cache_file(cache_dir, symbol)
    if not fp.exists():
        return None

    try:
        if ttl_hours is not None and ttl_hours > 0:
            age_seconds = time.time() - fp.stat().st_mtime
            if age_seconds > float(ttl_hours) * 3600.0:
                return None
        data = json.loads(fp.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            _MEM_CACHE[key] = data
            return data
    except Exception:
        return None
    return None


def save_cached_payload(cache_dir: Path, symbol: str, data: Dict[str, Any]) -> None:
    symbol = symbol.upper().strip()
    key = _cache_key(cache_dir, symbol)
    fp = _cache_file(cache_dir, symbol)
    fp.write_text(json.dumps(data), encoding="utf-8")
    _MEM_CACHE[key] = data


def get_cached_payload(
    symbol: str,
    fetch_fn: Callable[[str], Optional[Dict[str, Any]]],
    *,
    cache_dir: Path,
    sleep_sec: float = 0.8,
    ttl_hours: float | None = None,
    force_refresh: bool = False,
    log_prefix: str = "[CACHE]",
) -> Dict[str, Any]:
    symbol = symbol.upper().strip()

    if not force_refresh:
        cached = load_cached_payload(cache_dir, symbol, ttl_hours=ttl_hours)
        if cached is not None:
            return cached

    print(f"{log_prefix} Fetching for {symbol}")
    data = fetch_fn(symbol) or {}
    if isinstance(data, dict) and data:
        save_cached_payload(cache_dir, symbol, data)
    if sleep_sec > 0:
        time.sleep(float(sleep_sec))
    return data if isinstance(data, dict) else {}


def get_eps_data(symbol: str, fetch_fn: Callable[[str], Optional[Dict[str, Any]]], sleep_sec: float = 0.8,
                 ttl_hours: float | None = 24 * 7, force_refresh: bool = False) -> Dict[str, Any]:
    return get_cached_payload(
        symbol,
        fetch_fn,
        cache_dir=EPS_CACHE_DIR,
        sleep_sec=sleep_sec,
        ttl_hours=ttl_hours,
        force_refresh=force_refresh,
        log_prefix="[EPS]",
    )


def get_sentiment_data(symbol: str, fetch_fn: Callable[[str], Optional[Dict[str, Any]]], sleep_sec: float = 0.4,
                       ttl_hours: float | None = 24, force_refresh: bool = False) -> Dict[str, Any]:
    return get_cached_payload(
        symbol,
        fetch_fn,
        cache_dir=SENTIMENT_CACHE_DIR,
        sleep_sec=sleep_sec,
        ttl_hours=ttl_hours,
        force_refresh=force_refresh,
        log_prefix="[SENTIMENT]",
    )
