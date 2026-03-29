from __future__ import annotations

import os
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

try:
    import requests
except Exception:
    requests = None


AV_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()

ETF_LIKE_HINTS = {
    "QQQ", "SPY", "IWM", "DIA", "BUG", "CPER", "GLD", "SLV", "IBIT", "BITO",
    "ARKQ", "ARKK", "ARKX", "QTUM", "WGMI", "CRPT", "SOXX", "XLK", "XLE",
    "XLF", "XLV", "XLI", "XLP", "XLY", "XLB", "XLU", "XLRE", "XLC",
    "HACK", "CIBR", "ROBO", "BOTZ", "ESPO", "ITA", "PPA", "GDX", "GDXJ"
}


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if not np.isfinite(v):
            return default
        return v
    except Exception:
        return default


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    try:
        return float(max(lo, min(hi, x)))
    except Exception:
        return 0.0


def _av_get(params: Dict[str, Any]) -> Dict[str, Any]:
    if requests is None:
        raise RuntimeError("requests is not installed. Install it with: pip install requests")
    if not AV_KEY:
        raise RuntimeError("Missing ALPHA_VANTAGE_API_KEY environment variable")

    url = "https://www.alphavantage.co/query"
    resp = requests.get(url, params={**params, "apikey": AV_KEY}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    print(f"[EPS] Fetching Alpha Vantage for {params.get('symbol')}")

    if not isinstance(data, dict):
        raise RuntimeError("Alpha Vantage returned non-dict response")

    if "Error Message" in data:
        raise RuntimeError(str(data["Error Message"]))
    if "Information" in data:
        raise RuntimeError(str(data["Information"]))
    if "Note" in data:
        raise RuntimeError(str(data["Note"]))

    return data


def detect_asset_type(symbol: str) -> str:
    s = str(symbol).upper().strip()
    if s in ETF_LIKE_HINTS:
        return "ETF"

    try:
        data = _av_get({"function": "OVERVIEW", "symbol": s})
        etf_flag = str(data.get("ETF", "")).strip().lower()
        if etf_flag == "true":
            return "ETF"

        asset_type = str(data.get("AssetType", "")).strip().upper()
        if asset_type == "ETF":
            return "ETF"

        exchange = str(data.get("Exchange", "")).strip()
        sector = str(data.get("Sector", "")).strip()
        industry = str(data.get("Industry", "")).strip()

        if exchange and (sector or industry):
            return "STOCK"
    except Exception:
        pass

    return "STOCK"


def fetch_quarterly_eps_alpha_vantage(symbol: str) -> pd.DataFrame:
    data = _av_get({"function": "EARNINGS", "symbol": str(symbol).upper().strip()})
    rows = data.get("quarterlyEarnings", []) or []
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if "fiscalDateEnding" in df.columns:
        df["fiscalDateEnding"] = pd.to_datetime(df["fiscalDateEnding"], errors="coerce")

    for c in ["reportedEPS", "estimatedEPS", "surprise"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "surprisePercentage" in df.columns:
        df["surprisePercentage"] = (
            df["surprisePercentage"].astype(str).str.replace("%", "", regex=False)
        )
        df["surprisePercentage"] = pd.to_numeric(df["surprisePercentage"], errors="coerce")

    return df.sort_values("fiscalDateEnding").reset_index(drop=True)


def compute_stock_eps_features(symbol: str) -> Dict[str, Any]:
    try:
        df = fetch_quarterly_eps_alpha_vantage(symbol)
    except Exception:
        return {
            "ASSET_TYPE": "STOCK",
            "EPS_AVAILABLE": 0,
            "eps_inc_2q": 0,
            "eps_inc_3q": 0,
            "eps_inc_4q": 0,
            "eps_growth_qoq": 0.0,
            "eps_surprise_pct_last": 0.0,
            "eps_quality_score": 0.0,
            "ETF_PROXY_GROWTH_SCORE": 0.0,
        }

    if df.empty or "reportedEPS" not in df.columns:
        return {
            "ASSET_TYPE": "STOCK",
            "EPS_AVAILABLE": 0,
            "eps_inc_2q": 0,
            "eps_inc_3q": 0,
            "eps_inc_4q": 0,
            "eps_growth_qoq": 0.0,
            "eps_surprise_pct_last": 0.0,
            "eps_quality_score": 0.0,
            "ETF_PROXY_GROWTH_SCORE": 0.0,
        }

    eps = pd.to_numeric(df["reportedEPS"], errors="coerce").dropna()
    if eps.empty:
        return {
            "ASSET_TYPE": "STOCK",
            "EPS_AVAILABLE": 0,
            "eps_inc_2q": 0,
            "eps_inc_3q": 0,
            "eps_inc_4q": 0,
            "eps_growth_qoq": 0.0,
            "eps_surprise_pct_last": 0.0,
            "eps_quality_score": 0.0,
            "ETF_PROXY_GROWTH_SCORE": 0.0,
        }

    diffs = eps.diff()
    inc_flags = (diffs > 0).astype(int)

    def trailing_inc(n: int) -> int:
        if len(inc_flags) < n:
            return 0
        return int(inc_flags.tail(n).sum() == n)

    last_eps = _safe_float(eps.iloc[-1], 0.0)
    prev_eps = _safe_float(eps.iloc[-2], 0.0) if len(eps) >= 2 else 0.0
    eps_growth_qoq = ((last_eps - prev_eps) / abs(prev_eps)) if prev_eps not in (0.0, -0.0) else 0.0

    last_surprise = 0.0
    if "surprisePercentage" in df.columns:
        sp = pd.to_numeric(df["surprisePercentage"], errors="coerce").dropna()
        if not sp.empty:
            last_surprise = _safe_float(sp.iloc[-1], 0.0)

    # Quality score in roughly [0, 1]
    quality = (
        0.35 * trailing_inc(4) +
        0.20 * trailing_inc(3) +
        0.10 * trailing_inc(2) +
        0.20 * max(0.0, _clamp(eps_growth_qoq, -1.0, 1.0)) +
        0.15 * max(0.0, _clamp(last_surprise / 100.0, -1.0, 1.0))
    )

    return {
        "ASSET_TYPE": "STOCK",
        "EPS_AVAILABLE": 1,
        "eps_inc_2q": trailing_inc(2),
        "eps_inc_3q": trailing_inc(3),
        "eps_inc_4q": trailing_inc(4),
        "eps_growth_qoq": round(float(eps_growth_qoq), 4),
        "eps_surprise_pct_last": round(float(last_surprise), 4),
        "eps_quality_score": round(float(quality), 4),
        "ETF_PROXY_GROWTH_SCORE": 0.0,
    }


def compute_etf_proxy_features(symbol: str, price_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    d = price_df.copy() if price_df is not None else pd.DataFrame()
    if d.empty or "close" not in d.columns:
        return {
            "ASSET_TYPE": "ETF",
            "EPS_AVAILABLE": 0,
            "eps_inc_2q": 0,
            "eps_inc_3q": 0,
            "eps_inc_4q": 0,
            "eps_growth_qoq": 0.0,
            "eps_surprise_pct_last": 0.0,
            "eps_quality_score": 0.0,
            "ETF_PROXY_GROWTH_SCORE": 0.0,
        }

    close = pd.to_numeric(d["close"], errors="coerce")
    volume = pd.to_numeric(d["volume"], errors="coerce") if "volume" in d.columns else pd.Series(dtype=float)

    ret_20 = 0.0
    ret_60 = 0.0
    if len(close.dropna()) >= 21 and _safe_float(close.iloc[-21], 0.0) != 0.0:
        ret_20 = (float(close.iloc[-1]) / float(close.iloc[-21])) - 1.0
    if len(close.dropna()) >= 61 and _safe_float(close.iloc[-61], 0.0) != 0.0:
        ret_60 = (float(close.iloc[-1]) / float(close.iloc[-61])) - 1.0

    vol_ratio = 0.0
    if not volume.empty:
        v20 = _safe_float(volume.tail(20).mean(), 0.0)
        v60 = _safe_float(volume.tail(60).mean(), 0.0)
        if v60 > 0:
            vol_ratio = v20 / v60

    score = (
        0.45 * _clamp(ret_60, -1.0, 1.0) +
        0.35 * _clamp(ret_20, -1.0, 1.0) +
        0.20 * _clamp(vol_ratio - 1.0, -1.0, 1.0)
    )

    proxy_score = round(float(score), 4)

    return {
        "ASSET_TYPE": "ETF",
        "EPS_AVAILABLE": 0,
        "eps_inc_2q": 0,
        "eps_inc_3q": 0,
        "eps_inc_4q": 0,
        "eps_growth_qoq": 0.0,
        "eps_surprise_pct_last": 0.0,
        "eps_quality_score": proxy_score,
        "ETF_PROXY_GROWTH_SCORE": proxy_score,
    }


def compute_eps_features(symbol: str, price_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    asset_type = detect_asset_type(symbol)
    if asset_type == "ETF":
        return compute_etf_proxy_features(symbol, price_df)
    return compute_stock_eps_features(symbol)


# Backward-compatible wrappers if old code calls these
def fetch_quarterly_eps(symbol: str) -> pd.DataFrame:
    asset_type = detect_asset_type(symbol)
    if asset_type == "ETF":
        return pd.DataFrame()
    return fetch_quarterly_eps_alpha_vantage(symbol)


def eps_growth_flags(symbol: str, price_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    return compute_eps_features(symbol, price_df)


def fetch_market_sentiment(symbol: str) -> Dict[str, Any]:
    # Placeholder. Keep interface stable.
    return {
        "news_sentiment_score": 0.0,
        "news_positive_ratio": 0.0,
        "news_article_count": 0,
    }