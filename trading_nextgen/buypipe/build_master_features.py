from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .compute import compute_indicators
from .compute import compute_mean_reversion, generate_exit_reason
from .compute_orderflow_institutional import _add_orderflow_institutional_layer
from .config import IB_HOST, IB_PORT, IB_CLIENT_ID
from .fetching import fetch_data_daily_with_fallback
from .l2_ibkr import fetch_l2_snapshot

# Optional real IBKR L2 support
try:
    from ib_insync import IB, Stock
except Exception:
    IB = None
    Stock = None


# ============================================================
# Fetching
# ============================================================

def fetch_daily(
        symbol: str,
        years: int = 10,
        ttl_minutes: int = 240,
        force_refresh: bool = False,
):
    try:
        df, source, err = fetch_data_daily_with_fallback(
            symbol=symbol,
            bar_spec=f"{years} Y",
            bar_size="1 day",
            ttl_minutes=ttl_minutes,
            require_today=False,
            force_refresh=force_refresh,
        )

        if df is None or getattr(df, "empty", True):
            return None, source or "ERROR", err or "empty dataframe"

        src = str(source or "").upper()
        if "YFINANCE" in src or "YAHOO" in src:
            return None, source, "yfinance source is disabled by policy"

        return df, source, err or ""
    except Exception as e:
        return None, "ERROR", str(e)


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    colmap = {c.lower(): c for c in df.columns}

    def get(*names):
        for n in names:
            if n in colmap:
                return colmap[n]
        return None

    dcol = get("date", "datetime", "time")
    if dcol and dcol != "date":
        df = df.rename(columns={dcol: "date"})

    for src, dst in [
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
    ]:
        c = get(src, src.capitalize(), src.upper())
        if c and c != dst:
            df = df.rename(columns={c: dst})

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")

    return df


# ============================================================
# Helpers
# ============================================================

def safe_log(msg: str) -> None:
    try:
        print(msg)
    except Exception:
        pass


def _try_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def _safe_last(series, default=np.nan):
    try:
        if series is None or len(series) == 0:
            return default
        v = series.iloc[-1]
        return float(v) if pd.notna(v) else default
    except Exception:
        return default


def _safe_str(v, default: str = "") -> str:
    try:
        if v is None:
            return default
        s = str(v).strip()
        return s if s else default
    except Exception:
        return default


def _safe_bool(v, default: bool = False) -> bool:
    try:
        if v is None:
            return default
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s in {"y", "yes", "true", "1"}:
            return True
        if s in {"n", "no", "false", "0", ""}:
            return False
        return bool(v)
    except Exception:
        return default


def _clip01(x):
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0


def _safe_float_or_nan(v) -> float:
    try:
        if v is None:
            return np.nan
        out = float(v)
        return out if np.isfinite(out) else np.nan
    except Exception:
        return np.nan


def _grade_from_score(score: float) -> str:
    s = _clip01(score)
    if s >= 0.85:
        return "A"
    if s >= 0.70:
        return "B"
    if s >= 0.55:
        return "C"
    return "D"


def _holding_period_from_stage(stage: str, substage: str) -> str:
    s = _safe_str(stage).upper()
    ss = _safe_str(substage).upper()

    if "ACCUMULATION" in s and any(k in ss for k in ["BASE", "SPRING", "SECONDARY_TEST", "COMPRESSION"]):
        return "6-12 weeks"
    if "MARK-UP" in s or "MARKUP" in s:
        return "4-10 weeks"
    if "MARK-DOWN" in s or "MARKDOWN" in s:
        return "0-2 weeks"
    if "DISTRIBUTION" in s:
        return "1-3 weeks"
    return "2-6 weeks"


# ============================================================
# VWAP
# ============================================================

def _rolling_vwap(price: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    pv = price * volume
    pv_sum = pv.rolling(window, min_periods=max(5, window // 4)).sum()
    vol_sum = volume.rolling(window, min_periods=max(5, window // 4)).sum()
    return pv_sum / vol_sum.replace(0, np.nan)


def compute_vwap_and_reversion(df: pd.DataFrame) -> dict:
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")

    typical = (high + low + close) / 3.0

    vwap_20 = _rolling_vwap(typical, volume, 20)
    vwap_60 = _rolling_vwap(typical, volume, 60)
    vwap_252 = _rolling_vwap(typical, volume, 252)

    current = _safe_last(close)
    v20 = _safe_last(vwap_20)
    v60 = _safe_last(vwap_60)
    v252 = _safe_last(vwap_252)
    std20 = _safe_last(close.rolling(20).std(), 0.0)

    vwap_now = v60 if np.isfinite(v60) else v20
    vwap_support = v20 if np.isfinite(v20) else vwap_now

    z = (current - vwap_now) / (std20 + 1e-9) if np.isfinite(current) and np.isfinite(vwap_now) else np.nan

    return {
        "VWAP": float(vwap_now) if np.isfinite(vwap_now) else np.nan,
        "VWAP Support": float(vwap_support) if np.isfinite(vwap_support) else np.nan,
        "VWAP_20D": float(v20) if np.isfinite(v20) else np.nan,
        "VWAP_60D": float(v60) if np.isfinite(v60) else np.nan,
        "VWAP_252D": float(v252) if np.isfinite(v252) else np.nan,
        "VWAP_DISTANCE_PCT": float((current / vwap_now) - 1.0) if np.isfinite(current) and np.isfinite(
            vwap_now) and vwap_now != 0 else np.nan,
        "VWAP Deviation Z": float(z) if pd.notna(z) else np.nan,
        "Mean Reversion State": (
            "DEEP_UNDERVALUE" if pd.notna(z) and z < -2.0 else
            "UNDER_VWAP" if pd.notna(z) and z < -0.75 else
            "OVERHEATED" if pd.notna(z) and z > 2.0 else
            "NEUTRAL"
        ),
    }


# ============================================================
# Real IBKR Level 2
# ============================================================

def _l2_quality_from_rows(rows: int, spread_pct: float, total_size: float) -> str:
    if rows >= 8 and np.isfinite(spread_pct) and spread_pct < 0.08 and total_size > 2000:
        return "DEEP"
    if rows >= 5 and np.isfinite(spread_pct) and spread_pct < 0.15 and total_size > 500:
        return "GOOD"
    if rows >= 3:
        return "FAIR"
    return "THIN"


def _l2_weighted_price(levels):
    px_sz = [(p, s) for p, s in levels if np.isfinite(p) and np.isfinite(s) and s > 0]
    if not px_sz:
        return np.nan
    num = sum(p * s for p, s in px_sz)
    den = sum(s for _, s in px_sz)
    return num / den if den > 0 else np.nan


def _l2_slope(levels):
    px = [p for p, s in levels if np.isfinite(p)]
    if len(px) < 2:
        return np.nan
    return float(px[0] - px[-1])


def fetch_l2_snapshot(
        symbol: str,
        host: str = "127.0.0.1",
        port: int = 7496,
        client_id: int = 103,
        exchange: str = "ARCA",
        currency: str = "USD",
        num_rows: int = 10,
        timeout_sec: float = 2.5,
) -> Dict[str, Any]:
    """
    Real IBKR market depth via TWS/Gateway.
    Requires ib_insync + TWS/Gateway running + market depth permissions.
    """
    if IB is None or Stock is None:
        return {
            "L2 Status": "IB_INSYNC_NOT_INSTALLED",
            "L2 Quality": np.nan,
            "L2 Rows Requested": num_rows,
            "L2 Rows Received": 0,
        }

    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id, readonly=True, timeout=timeout_sec)

        contract = Stock(symbol, exchange, currency)
        qualified = ib.qualifyContracts(contract)
        if not qualified:
            return {
                "L2 Status": "CONTRACT_NOT_QUALIFIED",
                "L2 Quality": np.nan,
                "L2 Rows Requested": num_rows,
                "L2 Rows Received": 0,
            }

        contract = qualified[0]
        ticker = ib.reqMktDepth(contract, numRows=num_rows, isSmartDepth=False)

        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            ib.sleep(0.1)
            if len(ticker.domBids) > 0 or len(ticker.domAsks) > 0:
                break

        bids = []
        asks = []

        for row in list(ticker.domBids)[:num_rows]:
            bids.append((_safe_float_or_nan(row.price), _safe_float_or_nan(row.size)))

        for row in list(ticker.domAsks)[:num_rows]:
            asks.append((_safe_float_or_nan(row.price), _safe_float_or_nan(row.size)))

        ib.cancelMktDepth(contract)

        best_bid = bids[0][0] if bids else np.nan
        best_ask = asks[0][0] if asks else np.nan
        spread_abs = best_ask - best_bid if np.isfinite(best_bid) and np.isfinite(best_ask) else np.nan
        spread_pct = (spread_abs / ((best_ask + best_bid) / 2.0)) * 100.0 if np.isfinite(spread_abs) and (
                best_ask + best_bid) > 0 else np.nan

        total_bid_size = float(sum(s for _, s in bids if np.isfinite(s)))
        total_ask_size = float(sum(s for _, s in asks if np.isfinite(s)))
        imbalance = (
            (total_bid_size - total_ask_size) / (total_bid_size + total_ask_size)
            if (total_bid_size + total_ask_size) > 0 else np.nan
        )

        bid_vwap_5 = _l2_weighted_price(bids[:5])
        ask_vwap_5 = _l2_weighted_price(asks[:5])

        microprice = np.nan
        if np.isfinite(best_bid) and np.isfinite(best_ask) and bids and asks:
            bid_sz_1 = bids[0][1]
            ask_sz_1 = asks[0][1]
            denom = bid_sz_1 + ask_sz_1
            if np.isfinite(denom) and denom > 0:
                microprice = (best_ask * bid_sz_1 + best_bid * ask_sz_1) / denom

        rows_received = max(len(bids), len(asks))
        quality = _l2_quality_from_rows(rows_received, spread_pct, total_bid_size + total_ask_size)

        return {
            "L2 Status": "OK" if rows_received > 0 else "NO_DEPTH",
            "L2 Exchange": exchange,
            "L2 Rows Requested": num_rows,
            "L2 Rows Received": rows_received,
            "L2 Best Bid": best_bid,
            "L2 Best Ask": best_ask,
            "BID_ASK_SPREAD_PCT": spread_pct,
            "L2 Spread Abs": spread_abs,
            "L2 Total Bid Size": total_bid_size,
            "L2 Total Ask Size": total_ask_size,
            "L2 Imbalance": imbalance,
            "L2 Bid VWAP 5": bid_vwap_5,
            "L2 Ask VWAP 5": ask_vwap_5,
            "L2 Microprice": microprice,
            "L2 Bid Slope": _l2_slope(bids),
            "L2 Ask Slope": _l2_slope(asks),
            "L2 Quality": quality,
        }

    except Exception as e:
        return {
            "L2 Status": f"ERROR: {type(e).__name__}",
            "L2 Quality": np.nan,
            "L2 Rows Requested": num_rows,
            "L2 Rows Received": 0,
            "L2 Error Detail": str(e),
        }
    finally:
        try:
            if ib.isConnected():
                ib.disconnect()
        except Exception:
            pass


# ============================================================
# Derived packs
# ============================================================

def _compute_gap_and_flow_stats(df: pd.DataFrame) -> dict:
    close = pd.to_numeric(df["close"], errors="coerce")
    open_ = pd.to_numeric(df["open"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")

    prev_close = close.shift(1)
    gap_pct = ((open_ - prev_close) / prev_close.replace(0, np.nan)) * 100.0

    down_day = close < prev_close
    up_day = close > prev_close
    high_vol = volume > volume.rolling(20, min_periods=5).mean()

    distribution_days = ((down_day) & (high_vol)).rolling(20, min_periods=5).sum()
    accumulation_days = ((up_day) & (high_vol)).rolling(20, min_periods=5).sum()

    return {
        "GAP_PCT": float(gap_pct.iloc[-1]) if len(gap_pct) else np.nan,
        "DISTRIBUTION_DAYS_20D": float(distribution_days.iloc[-1]) if len(distribution_days) else np.nan,
        "ACCUMULATION_DAYS_20D": float(accumulation_days.iloc[-1]) if len(accumulation_days) else np.nan,
    }


def _compute_liquidity_score(df: pd.DataFrame) -> float:
    volume = pd.to_numeric(df["volume"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    dollar_vol = close * volume
    avg_dollar_vol = dollar_vol.rolling(20, min_periods=5).mean()
    dv = _safe_last(avg_dollar_vol)

    if not np.isfinite(dv):
        return np.nan
    if dv >= 100_000_000:
        return 1.0
    if dv >= 20_000_000:
        return 0.8
    if dv >= 5_000_000:
        return 0.6
    if dv >= 1_000_000:
        return 0.4
    return 0.2


def _compute_sector_correlation(df: pd.DataFrame, qqq_df: Optional[pd.DataFrame]) -> float:
    if qqq_df is None or qqq_df.empty:
        return np.nan
    try:
        lhs = pd.to_numeric(df["close"], errors="coerce").pct_change()
        rhs = pd.to_numeric(qqq_df["close"], errors="coerce").pct_change()
        joined = pd.concat([lhs.rename("lhs"), rhs.rename("rhs")], axis=1).dropna().tail(60)
        if len(joined) < 10:
            return np.nan
        return float(joined["lhs"].corr(joined["rhs"]))
    except Exception:
        return np.nan


def _compute_weekly_pattern_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "Whether Weekly chart has got higher up wicks volume.": np.nan,
            "How much % high Weekly chart is from previous lower volume.": np.nan,
        }

    w = df.resample("W").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    ).dropna()

    if len(w) < 3:
        return {
            "Whether Weekly chart has got higher up wicks volume.": np.nan,
            "How much % high Weekly chart is from previous lower volume.": np.nan,
        }

    last = w.iloc[-1]
    prev = w.iloc[-2]

    body_top = max(last["open"], last["close"])
    upper_wick = last["high"] - body_top
    full_range = max(last["high"] - last["low"], 1e-9)

    higher_up_wick_vol = int((upper_wick / full_range > 0.35) and (last["volume"] > prev["volume"]))
    high_from_prev_low_vol = ((last["high"] / max(prev["high"], 1e-9)) - 1.0) * 100.0

    return {
        "Whether Weekly chart has got higher up wicks volume.": higher_up_wick_vol,
        "How much % high Weekly chart is from previous lower volume.": float(high_from_prev_low_vol),
    }


def _compute_candle_entries(df: pd.DataFrame) -> dict:
    try:
        from .compute import candle_entries_multi
        entries = candle_entries_multi(df, weeks_list=(2, 4, 6, 8, 12, 18, 30))
        return {
            "Candle Entry 2w": float(entries.get(2, np.nan)),
            "Candle Entry 4w": float(entries.get(4, np.nan)),
            "Candle Entry 6w": float(entries.get(6, np.nan)),
            "Candle Entry 8w": float(entries.get(8, np.nan)),
            "Candle Entry 12w": float(entries.get(12, np.nan)),
            "Candle Entry 18w": float(entries.get(18, np.nan)),
            "Candle Entry 30w": float(entries.get(30, np.nan)),
        }
    except Exception:
        low = pd.to_numeric(df["low"], errors="coerce")
        out = {}
        windows = {
            "Candle Entry 2w": 10,
            "Candle Entry 4w": 20,
            "Candle Entry 6w": 30,
            "Candle Entry 8w": 40,
            "Candle Entry 12w": 60,
            "Candle Entry 18w": 90,
            "Candle Entry 30w": 150,
        }
        for col, win in windows.items():
            if len(low) >= 3:
                out[col] = float(low.rolling(win, min_periods=min(win, max(5, len(low) // 4))).min().iloc[-1])
            else:
                out[col] = np.nan
        return out


def _compute_volume_pack(df: pd.DataFrame) -> dict:
    vol = pd.to_numeric(df["volume"], errors="coerce")
    v20 = vol.rolling(20, min_periods=1).mean()
    v60 = vol.rolling(60, min_periods=1).mean()
    curr = float(vol.iloc[-1])
    avg20 = float(v20.iloc[-1]) if len(v20) else np.nan
    avg60 = float(v60.iloc[-1]) if len(v60) else np.nan
    vol_weight = curr / (avg20 + 1e-9) if np.isfinite(avg20) else np.nan

    return {
        "VOL_TODAY": curr,
        "AVG_VOL_20D": avg20,
        "AVG_VOL_60D": avg60,
        "Volume Weight": float(vol_weight) if np.isfinite(vol_weight) else np.nan,
        "Volume Surge": float(curr / (avg20 + 1e-9)) if np.isfinite(avg20) else np.nan,
        "VOL_TREND_5D": float(vol.pct_change(5).iloc[-1]) if len(vol) >= 6 and pd.notna(
            vol.pct_change(5).iloc[-1]) else np.nan,
    }


def _compute_sentiment_pack(out: dict) -> dict:
    rsi = _safe_float_or_nan(out.get("RSI"))
    stage = _safe_str(out.get("Market Stage")).upper()
    reg_score = _safe_float_or_nan(out.get("Regime Composite Score"))
    if not np.isfinite(reg_score):
        reg_score = _safe_float_or_nan(out.get("stage_strength_score"))

    if "MARK-UP" in stage or "MARKUP" in stage:
        label = "Bullish"
    elif "MARK-DOWN" in stage or "MARKDOWN" in stage:
        label = "Bearish"
    elif np.isfinite(rsi) and rsi < 30:
        label = "Oversold"
    else:
        label = "Neutral"

    conf = _clip01(abs(reg_score) if np.isfinite(reg_score) else 0.0)
    return {
        "Sentiment Label": label,
        "Sentiment Confidence": round(conf, 4),
    }


def _compute_pattern_pack(out: dict) -> dict:
    flags: List[str] = []

    def _add_flag(name: str) -> None:
        if name and name not in flags:
            flags.append(name)

    raw_pattern = _safe_str(out.get("Pattern Detected") or out.get("Pattern_Detected"))
    if raw_pattern and raw_pattern.upper() not in {"NONE", "NAN"}:
        for part in [p.strip() for p in raw_pattern.split(",") if p.strip()]:
            _add_flag(part.upper().replace(" ", "_"))

    if _safe_bool(out.get("Darvas Breakout Active")) or _safe_str(out.get("Darvas Signal")).upper() == "Y":
        _add_flag("DARVAS_BREAKOUT")

    smc_state = _safe_str(out.get("SMC Structure State")).upper()
    if smc_state in {"BULLISH_BREAK", "BREAKOUT", "BULLISH_BREAKOUT"} or _safe_bool(out.get("SMC_Breakout")):
        _add_flag("SMC_BREAKOUT")

    mr_state = _safe_str(out.get("Mean Reversion State")).upper()
    if mr_state in {"DEEP_UNDERVALUE", "UNDER_VWAP"} or _safe_bool(out.get("Mean_Reversion")):
        _add_flag("MEAN_REVERSION")

    if _safe_bool(out.get("Breakout")):
        _add_flag("BREAKOUT")

    if _safe_bool(out.get("DipReclaim")):
        _add_flag("VWAP_RECLAIM")

    if _safe_bool(out.get("Bullish_Engulfing")):
        _add_flag("BULLISH_ENGULFING")

    if _safe_bool(out.get("Hammer")):
        _add_flag("HAMMER")

    primary = flags[0] if flags else "NONE"
    pattern_list = ", ".join(flags) if flags else "NONE"
    return {
        "Patterns": pattern_list,
        "Primary Pattern": primary,
        "Pattern Count": len(flags),
        "Pattern Detected": pattern_list,
    }


def _compute_technical_snapshot(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    close = pd.to_numeric(df["close"], errors="coerce")

    def last_num(col, default=np.nan):
        series = df.get(col, pd.Series([default] * len(df), index=df.index))
        return float(pd.to_numeric(series, errors="coerce").iloc[-1])

    rsi = last_num("RSI_14")
    obv = pd.to_numeric(df.get("OBV", pd.Series(np.nan, index=df.index)), errors="coerce")
    obv_trend = np.nan
    if len(obv.dropna()) >= 5:
        obv_trend = float(obv.iloc[-1] - obv.iloc[-5])

    bb_lower = last_num("BB_lower")
    price = float(close.iloc[-1])

    dma20 = float(close.rolling(20, min_periods=5).mean().iloc[-1])
    dma50 = float(close.rolling(50, min_periods=10).mean().iloc[-1])
    dma100 = float(close.rolling(100, min_periods=20).mean().iloc[-1])
    dma150 = float(close.rolling(150, min_periods=30).mean().iloc[-1])
    dma200 = float(close.rolling(200, min_periods=40).mean().iloc[-1])

    return {
        "ADX Strength": last_num("ADX_14"),
        "RSI State": (
            "OVERSOLD" if np.isfinite(rsi) and rsi < 30 else
            "OVERBOUGHT" if np.isfinite(rsi) and rsi > 70 else
            "NEUTRAL"
        ),
        "OBV Trend": (
            "UP" if np.isfinite(obv_trend) and obv_trend > 0 else
            "DOWN" if np.isfinite(obv_trend) and obv_trend < 0 else
            np.nan
        ),
        "Volume Pressure": last_num("volume_pressure"),
        "Bullish_Engulfing": _safe_bool(last.get("Bullish_Engulfing")),
        "Hammer": _safe_bool(last.get("Hammer")),
        "Breakout": _safe_bool(last.get("HIGH_VOLUME_BREAKOUT", False)) or _safe_bool(
            last.get("RESISTANCE_BREAKOUT", False)),
        "Pattern Detected": last.get("Pattern_Detected", np.nan) if isinstance(last, pd.Series) else np.nan,
        "DipReclaim": _safe_bool(last.get("VWAP_RECLAIM", False)),
        "SMC_Breakout": _safe_bool(last.get("SMC_Breakout", False)),
        "Mean_Reversion": _safe_bool(last.get("Mean_Reversion", False)),
        "Trend_Strength": last_num("TREND_EFFICIENCY_20"),
        "Darvas Breakout %": last_num("darvas_breakout_pct"),
        "Darvas Signal": "Y" if _safe_bool(last.get("darvas_signal", False)) else np.nan,
        "Near Support": _safe_bool(last.get("near_support", False)),
        "At BB Lower": bool(np.isfinite(bb_lower) and np.isfinite(price) and price <= bb_lower * 1.02),
        "DMA20": dma20,
        "DMA50": dma50,
        "DMA100": dma100,
        "DMA150": dma150,
        "DMA200": dma200,
        "PCT_FROM_DMA20": float((price / dma20) - 1.0) if np.isfinite(dma20) and dma20 != 0 else np.nan,
        "PCT_FROM_DMA50": float((price / dma50) - 1.0) if np.isfinite(dma50) and dma50 != 0 else np.nan,
        "PCT_FROM_DMA100": float((price / dma100) - 1.0) if np.isfinite(dma100) and dma100 != 0 else np.nan,
        "PCT_FROM_DMA150": float((price / dma150) - 1.0) if np.isfinite(dma150) and dma150 != 0 else np.nan,
        "PCT_FROM_DMA200": float((price / dma200) - 1.0) if np.isfinite(dma200) and dma200 != 0 else np.nan,
        "DMA_STACK": (
            "BULLISH" if np.isfinite(dma20) and np.isfinite(dma50) and np.isfinite(
                dma200) and dma20 > dma50 > dma200 else
            "BEARISH" if np.isfinite(dma20) and np.isfinite(dma50) and np.isfinite(
                dma200) and dma20 < dma50 < dma200 else
            np.nan
        ),
        "ABOVE_DMA20": bool(np.isfinite(dma20) and price > dma20),
        "ABOVE_DMA50": bool(np.isfinite(dma50) and price > dma50),
        "Whether the current DMA is greater than 50 DMA.": int(
            np.isfinite(dma20) and np.isfinite(dma50) and dma20 > dma50),
        "Whether the current DMA is greater than 100 DMA.": int(
            np.isfinite(dma20) and np.isfinite(dma100) and dma20 > dma100),
        "Whether the current DMA is greater than 150 DMA.": int(
            np.isfinite(dma20) and np.isfinite(dma150) and dma20 > dma150),
        "Whether the current DMA is greater than 200.": int(
            np.isfinite(dma20) and np.isfinite(dma200) and dma20 > dma200),
        "REL_STRENGTH_20D_VS_QQQ": last_num("REL_STRENGTH_20D_VS_QQQ"),
        "Sym Vol Regime": last_num("sym_vol_regime"),
        "VIX Vol Regime": last.get("VIX_regime", np.nan),
    }


def _compute_entry_execution_pack(out: dict) -> dict:
    current = _safe_float_or_nan(out.get("Current Price"))
    buy = _safe_float_or_nan(out.get("Refined Buy Price"))
    atr = _safe_float_or_nan(out.get("ATR14"))
    add_on = _safe_float_or_nan(out.get("Add_On_Dip_Price"))

    if not np.isfinite(current) or not np.isfinite(buy):
        return {
            "ML Entry Target": np.nan,
            "ML Entry Mode": np.nan,
            "ML Entry Bias ATR": np.nan,
            "Execution Action": np.nan,
            "Signal": np.nan,
            "Invalidation_Level": np.nan,
            "Risk_%": np.nan,
            "Reward_%": np.nan,
            "Atr Trailing Stop": np.nan,
            "Exit Now": np.nan,
            "Smart Money Flow": np.nan,
            "Smart Money Entry Bias": np.nan,
            "Entry Aggression": np.nan,
        }

    order_flow = _safe_float_or_nan(out.get("Order Flow Score"))
    inst_flow = _safe_float_or_nan(out.get("Institutional Flow Score"))
    absorption = _safe_float_or_nan(out.get("Absorption Score"))

    flow_strength = (
        0.45 * _clip01(order_flow)
        + 0.35 * _clip01(inst_flow)
        + 0.20 * _clip01(absorption)
    )

    # Conservative default stop/target anchored on ATR
    inval = buy - (1.2 * atr if np.isfinite(atr) else buy * 0.03)
    reward = current + (2.0 * atr if np.isfinite(atr) else current * 0.06)
    risk_pct = ((buy - inval) / buy) * 100.0 if np.isfinite(inval) and buy != 0 else np.nan
    reward_pct = ((reward - buy) / buy) * 100.0 if np.isfinite(reward) and buy != 0 else np.nan

    rr_existing = _safe_float_or_nan(out.get("Best_Risk_Reward"))
    rr_est = rr_existing
    if not np.isfinite(rr_est):
        if np.isfinite(risk_pct) and risk_pct > 0 and np.isfinite(reward_pct):
            rr_est = reward_pct / risk_pct
        elif np.isfinite(atr) and atr > 0:
            risk_abs = buy - inval
            reward_abs = reward - buy
            rr_est = reward_abs / risk_abs if risk_abs > 0 else np.nan

    # Smart-money / RR aware entry optimization
    optimized_buy = buy
    entry_bias = 0.0
    aggression = "NEUTRAL"

    if np.isfinite(add_on):
        optimized_buy = min(optimized_buy, add_on)

    if np.isfinite(atr) and atr > 0:
        if flow_strength >= 0.62 and np.isfinite(rr_est) and rr_est >= 2.2:
            # Strong institutional support + good RR: allow a slightly more aggressive fill
            optimized_buy = min(current, max(buy, optimized_buy + 0.20 * atr))
            entry_bias = 0.20
            aggression = "AGGRESSIVE"
        elif flow_strength <= 0.35 or (np.isfinite(rr_est) and rr_est < 1.5):
            # Weak flow or poor RR: wait deeper / demand better price
            optimized_buy = min(optimized_buy, buy - 0.35 * atr)
            entry_bias = -0.35
            aggression = "DEFENSIVE"
        elif flow_strength >= 0.50 and np.isfinite(rr_est) and rr_est >= 1.8:
            optimized_buy = min(current, max(buy, optimized_buy + 0.10 * atr))
            entry_bias = 0.10
            aggression = "MODERATE_AGGRESSIVE"
        else:
            optimized_buy = min(optimized_buy, buy - 0.10 * atr)
            entry_bias = -0.10
            aggression = "MODERATE_DEFENSIVE"

    optimized_buy = float(max(0.01, optimized_buy))

    premium = (current / optimized_buy) - 1.0 if optimized_buy > 0 else np.nan
    if np.isfinite(premium) and premium <= 0.025:
        mode = "AT_LEVEL"
        exec_action = "BUY_NOW"
    elif np.isfinite(premium) and premium <= 0.07:
        mode = "PULLBACK"
        exec_action = "BUY_ON_PULLBACK"
    else:
        mode = "WAIT_FOR_DIP"
        exec_action = "WAIT"

    inval = optimized_buy - (1.2 * atr if np.isfinite(atr) else optimized_buy * 0.03)
    reward = current + (2.2 * atr if np.isfinite(atr) and flow_strength >= 0.55 else 2.0 * atr if np.isfinite(atr) else current * 0.06)
    risk_pct = ((optimized_buy - inval) / optimized_buy) * 100.0 if np.isfinite(inval) and optimized_buy != 0 else np.nan
    reward_pct = ((reward - optimized_buy) / optimized_buy) * 100.0 if np.isfinite(reward) and optimized_buy != 0 else np.nan

    return {
        "ML Entry Target": optimized_buy,
        "ML Entry Mode": mode,
        "ML Entry Bias ATR": float((current - optimized_buy) / atr) if np.isfinite(atr) and atr != 0 else np.nan,
        "Execution Action": exec_action,
        "Signal": out.get("Recommendation", np.nan),
        "Invalidation_Level": inval,
        "Risk_%": risk_pct,
        "Reward_%": reward_pct,
        "Atr Trailing Stop": current - (2.0 * atr) if np.isfinite(atr) else np.nan,
        "Exit Now": False,
        "Smart Money Flow": round(float(flow_strength), 6),
        "Smart Money Entry Bias": entry_bias,
        "Entry Aggression": aggression,
    }


def _final_cleanup_pack(out: dict, df: pd.DataFrame) -> dict:
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")

    current = _safe_float_or_nan(out.get("Current Price"))
    atr = _safe_float_or_nan(out.get("ATR14"))
    buy = _safe_float_or_nan(out.get("Refined Buy Price"))

    # L2 proxy only when real L2 missing
    spread_pct = (
        ((high.iloc[-1] - low.iloc[-1]) / current) * 100
        if np.isfinite(current) and len(df) else np.nan
    )

    dollar_vol = (close * volume).rolling(20, min_periods=5).mean()
    avg_dollar_vol = _safe_float_or_nan(dollar_vol.iloc[-1])

    if np.isfinite(avg_dollar_vol):
        if avg_dollar_vol > 100_000_000:
            l2_quality = "DEEP"
        elif avg_dollar_vol > 20_000_000:
            l2_quality = "GOOD"
        elif avg_dollar_vol > 5_000_000:
            l2_quality = "FAIR"
        else:
            l2_quality = "THIN"
    else:
        l2_quality = np.nan

    buy_low = buy
    buy_high = buy * 1.02 if np.isfinite(buy) else np.nan

    add = _safe_float_or_nan(out.get("Add_On_Dip_Price"))
    add_high = add * 1.015 if np.isfinite(add) else np.nan

    trailing = current - 2 * atr if np.isfinite(current) and np.isfinite(atr) else np.nan

    conf = _safe_float_or_nan(out.get("Confidence Score"))
    setup_quality = "STRONG" if conf >= 0.7 else "MODERATE" if conf >= 0.5 else "WEAK"

    exec_action = _safe_str(out.get("Execution Action")).upper()
    entry_timing = "NOW" if exec_action == "BUY_NOW" else "PULLBACK" if exec_action == "BUY_ON_PULLBACK" else "WAIT"

    return {
        "Execution Action": out.get("Execution Action"),
        "Signal": out.get("Signal", out.get("Recommendation")),
        "Buy Range Low": buy_low,
        "Buy Range High": buy_high,
        "Add Range Low": add,
        "Add Range High": add_high,
        "Trailing Stop": trailing,
        "Setup Quality": setup_quality,
        "Entry Timing": entry_timing,
        "L2 Status": out.get("L2 Status", "NOT_WIRED"),
        "L2 Quality": out.get("L2 Quality", l2_quality),
        "Approx Spread %": out.get("BID_ASK_SPREAD_PCT", spread_pct),
        "Intraday Range %": spread_pct,
        "Avg Dollar Volume 20D": avg_dollar_vol,
    }


def backfill_output_columns(symbol: str, df: pd.DataFrame, out: dict, qqq_df: Optional[pd.DataFrame] = None) -> dict:
    close = pd.to_numeric(df["close"], errors="coerce")

    if not out.get("_entry_locked"):
        entries = _compute_candle_entries(df)

        for k, v in entries.items():
            out[k] = v

        out["_entry_locked"] = True

    # ============================================================
    # 🔥 PREVENT FLAT CANDLE ENTRY LADDER (VERY IMPORTANT)
    # ============================================================

    vals = [out.get(k) for k in [
        "Candle Entry 2w", "Candle Entry 4w", "Candle Entry 6w",
        "Candle Entry 8w", "Candle Entry 12w", "Candle Entry 18w", "Candle Entry 30w"
    ] if np.isfinite(out.get(k, np.nan))]

    if len(set([round(v, 2) for v in vals])) <= 1:
        for i, k in enumerate([
            "Candle Entry 2w", "Candle Entry 4w", "Candle Entry 6w",
            "Candle Entry 8w", "Candle Entry 12w", "Candle Entry 18w", "Candle Entry 30w"
        ]):
            if np.isfinite(out.get(k, np.nan)):
                out[k] *= (1 - 0.002 * i)

    out.update(_compute_volume_pack(df))
    out.update(_compute_sentiment_pack(out))
    out.update(_compute_technical_snapshot(df))
    out.update(_compute_pattern_pack(out))
    out.update(_compute_gap_and_flow_stats(df))
    out.update(_compute_weekly_pattern_stats(df))

    out["LIQUIDITY_SCORE"] = _compute_liquidity_score(df)
    out["Sector Correlation"] = _compute_sector_correlation(df, qqq_df)

    stage = _safe_str(out.get("Market Stage"), "UNKNOWN")
    substage = _safe_str(out.get("Market Sub-Stage"), "UNKNOWN")
    substage_conf = out.get("Substage Confidence", np.nan)

    try:
        substage_conf_val = float(substage_conf)
    except Exception:
        substage_conf_val = np.nan

    if not np.isfinite(substage_conf_val):
        stage_strength_fallback = _safe_float_or_nan(out.get("stage_strength_score"))
        substage_conf_val = _clip01(stage_strength_fallback if np.isfinite(stage_strength_fallback) else 0.0)

    out["Substage Confidence"] = round(float(substage_conf_val), 4)

    out["ADXInstitutional Score"] = _safe_float_or_nan(out.get("Institutional Score"))
    out["Trend"] = _safe_str(
        out.get("HTF_Trend")
        or out.get("Trend")
        or out.get("Market Mode")
        or out.get("Market Stage")
    )

    vol_weight = _safe_float_or_nan(out.get("Volume Weight"))
    inst = _safe_float_or_nan(out.get("Institutional Score"))
    stage_strength = _safe_float_or_nan(out.get("stage_strength_score"))

    vw_term = 0.20 * _clip01(vol_weight / 2.0) if np.isfinite(vol_weight) else 0.0
    inst_term = 0.30 * _clip01(inst) if np.isfinite(inst) else 0.0
    stage_term = 0.35 * _clip01(stage_strength) if np.isfinite(stage_strength) else 0.0
    sub_term = 0.15 * _clip01(float(out.get("Substage Confidence") or 0.0))

    sig_score = stage_term + inst_term + vw_term + sub_term
    out["Signal Score"] = round(sig_score, 6)
    out["signal_score"] = round(sig_score, 6)

    conf_score = 0.0
    conf_score += 0.30 * _clip01(float(out.get("Substage Confidence") or 0.0))
    if np.isfinite(stage_strength):
        conf_score += 0.30 * _clip01(stage_strength)
    if np.isfinite(inst):
        conf_score += 0.20 * _clip01(inst)
    if np.isfinite(vol_weight):
        conf_score += 0.20 * _clip01(vol_weight / 2.0)

    out["Confidence Score"] = round(float(conf_score), 6)
    out["Confidence Grade"] = _grade_from_score(float(conf_score))

    refined_buy = _safe_float_or_nan(out.get("Refined Buy Price"))
    if np.isfinite(refined_buy):
        out["Primary_Entry_Price"] = refined_buy
        out["Primary_Entry_Source"] = "Refined Buy Price"
    else:
        out["Primary_Entry_Price"] = _safe_last(close)
        out["Primary_Entry_Source"] = "Current Price"

    out["Add_On_Dip_Price"] = (
            out.get("Candle Entry 2w")
            or out.get("Candle Entry 4w")
            or out.get("Refined Buy Price")
            or _safe_last(close)
    )

    out["Expected Holding Period"] = _holding_period_from_stage(stage, substage)
    out["Momentum Expected Holding Period"] = out["Expected Holding Period"]
    out["Momentum Confidence Grade"] = out.get("Confidence Grade", np.nan)
    out["Momentum Decision Reason"] = out.get("Decision Reason", np.nan)
    out["Momentum Recommendation"] = out.get("Recommendation", np.nan)

    out.update(_compute_entry_execution_pack(out))
    out.update(_final_cleanup_pack(out, df))

    out.setdefault("Buy_Window_Status", "WAIT")
    out.setdefault("Position_Size Class", out.get("Position_Size_Class", "SMALL"))
    out.setdefault("Position_Size_Class", "SMALL")
    out.setdefault("Decision Reason", "pending_scoring")

    # Keep truly external fields unavailable unless actually sourced
    out.setdefault("News Sentiment Score", np.nan)
    out.setdefault("News Positive Ratio", np.nan)
    out.setdefault("News Article Count", np.nan)
    out.setdefault("BORROW_FEE_PCT", np.nan)
    out.setdefault("EPS_AVAILABLE", np.nan)
    out.setdefault("eps_growth_qoq", np.nan)
    out.setdefault("L2 Status", np.nan)

    out["Market Stage"] = stage
    out["Market Sub-Stage"] = substage

    pattern_detected = _safe_str(out.get("Pattern Detected"))
    if not pattern_detected:
        fallback_patterns = _safe_str(out.get("Patterns"))
        if fallback_patterns:
            out["Pattern Detected"] = fallback_patterns
        else:
            out["Pattern Detected"] = "NONE"

    return out


# ============================================================
# Core feature build
# ============================================================

def compute_symbol_features(
        symbol: str,
        df: pd.DataFrame,
        qqq_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "Symbol": symbol,
        "ASOF_DATE": datetime.utcnow().strftime("%Y-%m-%d"),
        "DATA_SOURCE": "N/A",
    }

    if df is None or df.empty:
        return out

    try:
        df = compute_indicators(df.copy(), symbol)
        df = _add_orderflow_institutional_layer(df)
    except Exception as e:
        print(f"[{symbol}] compute_indicators failed: {e}")

    try:
        last = df.iloc[-1]

        out["Order Flow Score"] = float(last.get("Order Flow Score", 0))
        out["Institutional Flow Score"] = float(last.get("Institutional Flow Score", 0))
        out["Absorption Score"] = float(last.get("Absorption Score", 0))
        out["Stealth Accumulation Score"] = float(last.get("Stealth Accumulation Score", 0))
        out["Stealth Distribution Score"] = float(last.get("Stealth Distribution Score", 0))
        last = df.iloc[-1]

        # existing assignments
        out["Order Flow Score"] = float(last.get("Order Flow Score", 0))
        out["Institutional Flow Score"] = float(last.get("Institutional Flow Score", 0))
        out["Absorption Score"] = float(last.get("Absorption Score", 0))

        # order_flow = _safe_float_or_nan(out.get("Order Flow Score"))
        # inst_flow = _safe_float_or_nan(out.get("Institutional Flow Score"))
        # absorption = _safe_float_or_nan(out.get("Absorption Score"))

        # flow_inst = (
        #         0.45 * _clip01(order_flow) +
        #         0.35 * _clip01(inst_flow) +
        #         0.20 * _clip01(absorption)
        # )
        #
        #
        # flow_inst = np.clip(flow_inst * 1.6, 0, 1)

        # print("FLOW DEBUG:", out.get("Order Flow Score"), out.get("Institutional Flow Score"),
        #       out.get("Absorption Score"))
        # print("INST DEBUG:", out["Institutional Score"], out["Order Flow Score"], out["Institutional Flow Score"],
        #       out["Absorption Score"])

        close = pd.to_numeric(df["close"], errors="coerce")
        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        volume = pd.to_numeric(df["volume"], errors="coerce")

        current_price = float(close.iloc[-1])
        ema21_series = close.ewm(span=21, adjust=False).mean()
        ema50_series = close.ewm(span=50, adjust=False).mean()
        atr14 = (high - low).rolling(14).mean()

        vwap_pack = compute_vwap_and_reversion(df)

        support_20 = low.rolling(20).min().iloc[-1]
        ema21 = float(ema21_series.iloc[-1])
        vwap_support = float(vwap_pack.get("VWAP Support", np.nan))

        refined_buy = max(
            support_20 if pd.notna(support_20) else np.nan,
            ema21 * 0.97 if np.isfinite(ema21) else np.nan,
            vwap_support * 0.99 if np.isfinite(vwap_support) else np.nan,
        )
        if np.isfinite(refined_buy) and refined_buy > current_price * 0.98:
            refined_buy = current_price * 0.97

        out.update(
            {
                "Symbol": symbol,
                "Current Price": current_price,
                "VWAP": vwap_pack.get("VWAP"),
                "VWAP Support": vwap_pack.get("VWAP Support"),
                "VWAP_20D": vwap_pack.get("VWAP_20D"),
                "VWAP_60D": vwap_pack.get("VWAP_60D"),
                "VWAP_252D": vwap_pack.get("VWAP_252D"),
                "VWAP_DISTANCE_PCT": vwap_pack.get("VWAP_DISTANCE_PCT"),
                "VWAP Deviation Z": vwap_pack.get("VWAP Deviation Z"),
                "Mean Reversion State": vwap_pack.get("Mean Reversion State"),
                "EMA21": float(ema21_series.iloc[-1]),
                "EMA50": float(ema50_series.iloc[-1]),
                "RSI": float(last.get("RSI_14", np.nan)),
                "ADX": float(last.get("ADX_14", np.nan)),
                "ATR14": float(last.get("ATR_14", _safe_last(atr14, np.nan))),
                "ATR14_PCT": float(last.get("ATR14_PCT", np.nan)),
                "Volume Surge": float(last.get("VOL_SURGE_RATIO", np.nan)),
                "VOL_SURGE_RATIO": float(last.get("VOL_SURGE_RATIO", np.nan)),
                "VOL_TODAY": float(last.get("VOL_TODAY", volume.iloc[-1])),
                "AVG_VOL_20D": float(last.get("AVG_VOL_20D", volume.rolling(20, min_periods=1).mean().iloc[-1])),
                "VOL_TREND_5D": float(
                    last.get("VOL_TREND_5D", volume.pct_change(5).iloc[-1] if len(volume) >= 6 else np.nan)),
                "Refined Buy Price": float(refined_buy) if np.isfinite(refined_buy) else np.nan,
                "Market Stage": last.get("market_stage", "UNKNOWN"),
                "Market Sub-Stage": last.get("market_substage", "UNKNOWN"),
                "Substage Confidence": float(last.get("substage_confidence", np.nan)),
                "stage_strength_score": float(last.get("stage_strength_score", np.nan)),
                "HTF_Trend": last.get("HTF_Trend", np.nan),
                "ITF_Trend": last.get("ITF_Trend", np.nan),
                "LTF_Trend": last.get("LTF_Trend", np.nan),
                "Near Support": bool(last.get("near_support", False)),
                "EMA Uptrend": bool(last.get("EMA_uptrend", False)),
                "EMA21 Slope": float(last.get("EMA_21_slope", np.nan)),
                "MACD Cross": bool(last.get("MACD_Crossover", False)),
                "MACD_SIGNAL": float(last.get("MACD_signal", np.nan)),
                "MACD_HIST": float(last.get("MACD_hist", np.nan)),
                "QUANT_COMPOSITE_SCORE": float(last.get("QUANT_COMPOSITE_SCORE", np.nan)),
                "REL_STRENGTH_20D_VS_QQQ": float(last.get("REL_STRENGTH_20D_VS_QQQ", np.nan)),
                "PCT_FROM_DMA50": float(last.get("PCT_FROM_DMA50", np.nan)),
                "PCT_FROM_DMA200": float(last.get("PCT_FROM_DMA200", np.nan)),
                "TREND_EFFICIENCY_20": float(last.get("TREND_EFFICIENCY_20", np.nan)),
                "volume_pressure": float(last.get("volume_pressure", np.nan)),
                "Darvas Breakout Active": bool(last.get("darvas_signal", False)),
                "Darvas Top": float(last.get("Darvas Top", np.nan)) if "Darvas Top" in last.index else np.nan,
                "Darvas Bottom": float(last.get("Darvas Bottom", np.nan)) if "Darvas Bottom" in last.index else np.nan,
                "SMC Structure State": last.get("SMC Structure State", np.nan),
                "SMC Breakout Level": float(
                    last.get("SMC Breakout Level", np.nan)) if "SMC Breakout Level" in last.index else np.nan,
                "SMC Invalidation Level": float(
                    last.get("SMC Invalidation Level", np.nan)) if "SMC Invalidation Level" in last.index else np.nan,
                "Rule Recommendation": last.get("rule_recommendation", "HOLD"),
                "ASSET_TYPE": "ETF" if symbol in {"QQQ", "CPER", "BUG"} else "STOCK",
            }
        )
    except Exception as e:
        print(f"[{symbol}] CORE FEATURE BUILD FAILED: {e}")

    # Real IBKR L2
    try:
        l2_data = fetch_l2_snapshot(
            symbol=symbol,
            host=IB_HOST,
            port=IB_PORT,
            client_id=IB_CLIENT_ID,
            exchange="ARCA",
            currency="USD",
            num_rows=10,
            timeout_sec=2.5,
        )
        if isinstance(l2_data, dict):
            out.update(l2_data)
    except Exception as e:
        out["L2 Status"] = f"ERROR: {type(e).__name__}"
        out["L2 Error Detail"] = str(e)

    try:
        import symbol_analysis as sa
        if hasattr(sa, "compute_symbol_indicators"):
            res = _try_call(sa.compute_symbol_indicators, df.copy())
            if isinstance(res, dict):
                out.update(res)
        elif hasattr(sa, "compute_indicators"):
            res = _try_call(sa.compute_indicators, df.copy())
            if isinstance(res, dict):
                out.update(res)
    except Exception:
        pass

    final_inst = np.nan
    try:
        from institutional_investor import score_institutional_investor
        legacy_inst = _safe_float_or_nan(out.get("Institutional Score"))

        order_flow = _safe_float_or_nan(out.get("Order Flow Score"))
        inst_flow = _safe_float_or_nan(out.get("Institutional Flow Score"))
        absorption = _safe_float_or_nan(out.get("Absorption Score"))

        # Combine properly
        flow_inst = (
                0.40 * _clip01(order_flow) +
                0.35 * _clip01(inst_flow) +
                0.25 * _clip01(absorption)
        )

        # Blend legacy + new (IMPORTANT)
        if np.isfinite(legacy_inst):
            final_inst = 0.5 * legacy_inst + 0.5 * flow_inst
        else:
            final_inst = flow_inst

    except Exception:
        pass

    out["Institutional Score"] = final_inst
    print("INST DEBUG:", out["Institutional Score"], ...)

    try:
        from darvas import darvas_box_signal
        tmp = _try_call(darvas_box_signal, df.copy())
        if isinstance(tmp, pd.DataFrame) and not tmp.empty:
            out["Darvas Signal"] = "Y" if int(tmp.get("darvas_signal", pd.Series([0])).iloc[-1]) == 1 else ""
            out["Darvas Breakout %"] = float(tmp.get("darvas_breakout_pct", pd.Series([0.0])).iloc[-1])
    except Exception:
        pass

    try:
        from upward import (
            detect_smc_accumulation_breakout,
            detect_mean_reversion_buy,
            detect_bullish_engulfing,
            detect_hammer,
            compute_upward_trend,
        )
        out["SMC_Breakout"] = "Y" if bool(_try_call(detect_smc_accumulation_breakout, df.copy())) else ""
        out["Bullish_Engulfing"] = "Y" if bool(_try_call(detect_bullish_engulfing, df.copy())) else ""
        out["Hammer"] = "Y" if bool(_try_call(detect_hammer, df.copy())) else ""
        mr_signal = _try_call(detect_mean_reversion_buy, df.copy())
        mr_score = compute_mean_reversion(df)

        out["Mean_Reversion_Flag"] = "Y" if bool(mr_signal) else ""
        out["Mean_Reversion"] = mr_score

        tr = _try_call(compute_upward_trend, df.copy())
        if isinstance(tr, (int, float, np.floating)):
            out["Trend_Strength"] = float(tr)
        elif isinstance(tr, dict):
            out.update(tr)
    except Exception:
        pass

    try:
        import eps_features as eps
        if hasattr(eps, "compute_eps_features"):
            res = _try_call(eps.compute_eps_features, symbol)
            if isinstance(res, dict):
                out.update(res)
    except Exception:
        pass

    try:
        import macro_features as mf
        if hasattr(mf, "compute_macro_features"):
            res = _try_call(mf.compute_macro_features)
            if isinstance(res, dict):
                out.update(res)
    except Exception:
        pass

    try:
        from rank_long_short import (
            load_rank_config,
            compute_common_indicators_rls,
            score_long,
            score_short,
            detect_long_setup,
            detect_short_setup,
            build_long_plan,
            build_short_plan,
        )
        cfg = load_rank_config()
        ind = _try_call(compute_common_indicators_rls, df.copy(), qqq_df, cfg)
        if isinstance(ind, dict):
            out.update(ind)

        ls = _try_call(detect_long_setup, df.copy(), cfg)
        ss = _try_call(detect_short_setup, df.copy(), cfg)
        if isinstance(ls, dict):
            out.update(ls)
        if isinstance(ss, dict):
            out.update(ss)

        lscore = _try_call(score_long, df.copy(), cfg, out)
        sscore = _try_call(score_short, df.copy(), cfg, out)
        if isinstance(lscore, dict):
            out.update(lscore)
        if isinstance(sscore, dict):
            out.update(sscore)

        lp = _try_call(build_long_plan, df.copy(), cfg, out)
        sp = _try_call(build_short_plan, df.copy(), cfg, out)
        if isinstance(lp, dict):
            out.update(lp)
        if isinstance(sp, dict):
            out.update(sp)
    except Exception:
        pass

    for k in [
        "Sentiment Label",
        "Sentiment Confidence",
        "Exit Reasons",
        "Days to Peak",
        "SHORTABLE_FLAG",
    ]:
        if k not in out or out[k] is None:
            out[k] = np.nan

    out["Current Price"] = out.get("Current Price", _safe_last(df["close"]))

    # ============================================================
    # 🔥 GUARANTEED OUTPUT FIELDS (FIX BLANK COLUMNS)
    # ============================================================

    try:
        # -------------------------------
        # 1. Mean Reversion
        # -------------------------------
        if "Mean_Reversion" not in out or pd.isna(out.get("Mean_Reversion")):
            out["Mean_Reversion"] = compute_mean_reversion(df)

        # ============================================================
        # 🔥 Best Risk Reward (ELITE VERSION — NEVER EMPTY)
        # ============================================================

        rr1 = _safe_float_or_nan(out.get("Risk/Reward T1"))
        rr2 = _safe_float_or_nan(out.get("Risk/Reward T2"))

        # ============================================================
        # 🔥 Best Risk Reward (FINAL FIX)
        # ============================================================

        risk_pct = _safe_float_or_nan(out.get("Risk_%"))
        reward_pct = _safe_float_or_nan(out.get("Reward_%"))

        if np.isfinite(risk_pct) and risk_pct > 0 and np.isfinite(reward_pct):
            rr = reward_pct / risk_pct
        else:
            entry = _safe_float_or_nan(out.get("Refined Buy Price"))
            atr = _safe_float_or_nan(out.get("ATR14"))

            if np.isfinite(entry) and np.isfinite(atr) and atr > 0:
                stop = entry - 1.5 * atr
                target = entry + 2.5 * atr

                risk = entry - stop
                reward = target - entry

                rr = reward / risk if risk > 0 else np.nan
            else:
                rr = np.nan

        out["Best_Risk_Reward"] = float(rr) if np.isfinite(rr) else np.nan

        out["RR_Quality"] = (
            "EXCELLENT" if rr >= 3 else
            "GOOD" if rr >= 2 else
            "AVERAGE" if rr >= 1.5 else
            "POOR"
        )

        # -------------------------------
        # 3. Exit Reasons
        # -------------------------------
        out["Exit Reasons"] = generate_exit_reason({
            "RSI": out.get("RSI"),
            "MACD Cross": out.get("MACD Cross"),
            "Current Price": out.get("Current Price"),
            "VWAP": out.get("VWAP"),
        })

    except Exception as e:
        print(f"[{symbol}] ⚠ Failed to populate extra fields:", e)

    if "VWAP" not in out or pd.isna(out["VWAP"]):
        out.update(compute_vwap_and_reversion(df))

    # --- Guaranteed population for missing export columns ---
    try:
        # Mean_Reversion: deterministic fallback from compute.py
        if "Mean_Reversion" not in out or pd.isna(out["Mean_Reversion"]) or out["Mean_Reversion"] == "":
            out["Mean_Reversion"] = compute_mean_reversion(df)

        # Exit Reasons: deterministic fallback text from current row state
        tmp_row = {
            "RSI": out.get("RSI"),
            "MACD Cross": out.get("MACD Cross"),
            "Current Price": out.get("Current Price"),
            "VWAP": out.get("VWAP"),
        }
        out["Exit Reasons"] = generate_exit_reason(tmp_row)
    except Exception as e:
        print(f"[{symbol}] guaranteed export columns failed: {e}")

    try:
        out = backfill_output_columns(symbol, df, out, qqq_df=qqq_df)
    except Exception as e:
        print(f"[{symbol}] backfill_output_columns failed: {e}")

    print(f"[{symbol}] DEBUG CORE FEATURES:")
    print(f"  Current Price: {out.get('Current Price')}")
    print(f"  VWAP: {out.get('VWAP')}")
    print(f"  VWAP_20D: {out.get('VWAP_20D')}")
    print(f"  VWAP_60D: {out.get('VWAP_60D')}")
    print(f"  Refined Buy Price: {out.get('Refined Buy Price')}")
    print(f"  Market Stage: {out.get('Market Stage')}")
    print(f"  Sub-Stage: {out.get('Market Sub-Stage')}")
    print(f"  Substage Confidence: {out.get('Substage Confidence')}")
    print(f"  L2 Status: {out.get('L2 Status')}")
    print(f"  L2 Quality: {out.get('L2 Quality')}")

    return out


# ============================================================
# Build rows
# ============================================================

def read_symbols_from_excel(xlsx: str) -> List[str]:
    df = pd.read_excel(xlsx, sheet_name=0)
    syms = [str(s).strip().upper() for s in df.get("Symbol", []).tolist() if str(s).strip()]
    return list(dict.fromkeys(syms))


def build_master_rows(
        template: str | None = None,
        ttl_minutes: int = 240,
        force_refresh: bool = False,
        limit: int = 0,
):
    symbols = []
    try:
        from .config import symbols as config_symbols
        symbols.extend([str(s).strip().upper() for s in config_symbols if str(s).strip()])
    except Exception:
        pass

    if template:
        try:
            xls = pd.read_excel(template, sheet_name=0)
            if "Symbol" in xls.columns:
                symbols.extend([str(s).strip().upper() for s in xls["Symbol"].dropna().tolist() if str(s).strip()])
        except Exception:
            pass

    symbols = list(dict.fromkeys(symbols))
    if limit and limit > 0:
        symbols = symbols[:limit]

    qqq_df = None
    try:
        qqq_df, _, _ = fetch_daily("QQQ", years=10, ttl_minutes=ttl_minutes, force_refresh=False)
        if qqq_df is not None and not qqq_df.empty:
            qqq_df = normalize_ohlcv(qqq_df)
    except Exception:
        qqq_df = None

    out = []
    for symbol in symbols:
        df, source, err = fetch_daily(
            symbol,
            years=10,
            ttl_minutes=ttl_minutes,
            force_refresh=force_refresh,
        )
        safe_log(f"[FETCH] {symbol} -> source={source} err={err}")

        if df is None or getattr(df, "empty", True):
            print(f"[{symbol}] [ERROR] FETCH FAILED -> source={source} err={err}")
            row = {
                "Symbol": symbol,
                "DATA_SOURCE": "ERROR",
                "Error Detail": err or "no data",
            }
            out.append((row, pd.DataFrame()))
            continue

        df = normalize_ohlcv(df)
        row = compute_symbol_features(symbol=symbol, df=df, qqq_df=qqq_df)
        row["DATA_SOURCE"] = source
        out.append((row, df))

    return out


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols-from", choices=["config", "excel", "both"], default="both")
    ap.add_argument("--in", dest="in_xlsx", default="predictions_summary_out.xlsx")
    ap.add_argument("--out", dest="out_path", default="master_features.parquet")
    ap.add_argument("--ttl", type=int, default=240)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    symbols = []
    if args.symbols_from in ("config", "both"):
        try:
            from .config import symbols as CONFIG_SYMBOLS
            symbols += [str(s).strip().upper() for s in (CONFIG_SYMBOLS or []) if str(s).strip()]
        except Exception:
            pass

    if args.symbols_from in ("excel", "both") and os.path.exists(args.in_xlsx):
        symbols += read_symbols_from_excel(args.in_xlsx)

    symbols = list(dict.fromkeys(symbols))
    if args.limit and args.limit > 0:
        symbols = symbols[:args.limit]

    qqq_df, _, _ = fetch_daily("QQQ", years=args.years, ttl_minutes=args.ttl, force_refresh=args.refresh)
    qqq_df = normalize_ohlcv(qqq_df) if qqq_df is not None else None

    rows = []
    for i, sym in enumerate(symbols, 1):
        t0 = time.time()
        df, source, err = fetch_daily(sym, years=args.years, ttl_minutes=args.ttl, force_refresh=args.refresh)

        if df is None or len(df) < 60:
            rows.append({"Symbol": sym, "DATA_SOURCE": source, "ERROR": err or "fetch_failed"})
            continue

        df = normalize_ohlcv(df)
        out = compute_symbol_features(sym, df, qqq_df=qqq_df)
        out["DATA_SOURCE"] = source
        rows.append(out)

        if i % 25 == 0:
            dt = time.time() - t0
            print(f"{i}/{len(symbols)} {sym} done ({dt:.2f}s)")

    mdf = pd.DataFrame(rows)

    if args.out_path.lower().endswith(".parquet"):
        mdf.to_parquet(args.out_path, index=False)
    else:
        mdf.to_csv(args.out_path, index=False)

    print(f"Saved master features: {args.out_path} rows={len(mdf)} cols={len(mdf.columns)}")


if __name__ == "__main__":
    main()
