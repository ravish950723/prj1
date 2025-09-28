#!/usr/bin/env python3
"""
Time-to-Threshold + Institutional Flow & Zone Analyzer (IBKR / ib_insync)

Enhancements in this version:
- **Flow episodes** (multi-day accumulation/distribution windows)
- **Zone detection** limited to recent history (configurable lookback)
- **"Flow now" classification** → Investing / Pulling Out / Neutral
- **Watchlists** for next Accumulation/Distribution with configurable
  confidence and ETA filters (printed and written to CSV)

For each symbol in config_954.py, this script:
  1) Prints average trading days to first ±{5,10,15,20,25}% move
  2) Infers institutional accumulation/distribution via CMF/ADL/OBV/MFV/vol z-score
  3) Detects consolidation zones (Accumulation/Distribution/Neutral)
  4) Detects recent flow windows (episodes)
  5) Predicts next zone with confidence & ETA
  6) Builds near-term watchlists based on your thresholds

⚠️ "Institutional" flow here is *proxied* by price/volume behavior.

Usage
-----
python time_to_move.py \
  --years 5 --rth 0 --what ADJUSTED_LAST \
  --zones_lookback 250 --watch_conf 0.6 --eta_max 15

Outputs
-------
- Per-symbol console line with time-to-threshold, 90d flow counts, flow-now,
  last zone, next-zone prediction, and latest flow windows
- Final table with episode dates and zone info
- CSV: time_to_move_results.csv
- CSV: watchlist_next_accumulation.csv, watchlist_next_distribution.csv
"""

from __future__ import annotations

import os
import sys
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

try:
    from ib_insync import IB, Stock, util
except Exception as e:
    print("Please install ib_insync first: pip install ib_insync", file=sys.stderr)
    raise

# --- Load symbols & IB connection params from the user's config ---
try:
    from config import IB_HOST, IB_PORT, IB_CLIENT_ID, symbols  # type: ignore
except Exception as e:
    print("Could not import config_954.py from current directory / PYTHONPATH.", file=sys.stderr)
    print("Make sure the file exists and includes IB_HOST, IB_PORT, IB_CLIENT_ID, and symbols.", file=sys.stderr)
    raise


@dataclass
class Params:
    years: int = 5
    rth: bool = False
    what: str = "ADJUSTED_LAST"  # TRADES, MIDPOINT, BID, ASK, ADJUSTED_LAST
    min_rows: int = 260          # require at least ~1Y of data to compute stats
    zones_lookback: int = 250    # days of history to scan for consolidation zones
    watch_conf: float = 0.6      # min confidence for watchlists
    eta_max: int = 15            # max eta_hi (days) to include in watchlists


THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25]

# =========================
# Indicator computations
# =========================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add CMF/ADL/OBV, BB, ATR, z-scores & helper columns."""
    d = df.copy().sort_index()
    for col in ("open","high","low","close","volume"):
        d[col] = pd.to_numeric(d[col], errors="coerce")

    # Typical price & Money Flow components
    tp = (d["high"] + d["low"] + d["close"]) / 3.0
    hl_range = (d["high"] - d["low"]).replace(0, np.nan)
    mfm = ((d["close"] - d["low"]) - (d["high"] - d["close"])) / hl_range  # Money Flow Multiplier
    mfm = mfm.fillna(0.0)
    mfv = mfm * d["volume"]  # Money Flow Volume

    # ADL & CMF (20)
    d["ADL"] = mfv.cumsum()
    d["CMF20"] = mfv.rolling(20).sum() / d["volume"].rolling(20).sum()

    # OBV
    delta = d["close"].diff()
    obv = np.where(delta > 0, d["volume"], np.where(delta < 0, -d["volume"], 0))
    d["OBV"] = pd.Series(obv, index=d.index).cumsum()

    # Bollinger Bands (20, 2)
    ma = d["close"].rolling(20).mean()
    std = d["close"].rolling(20).std()
    d["BB_mid"] = ma
    d["BB_up"] = ma + 2*std
    d["BB_lo"] = ma - 2*std

    # ATR(14)
    tr1 = (d["high"] - d["low"]).abs()
    tr2 = (d["high"] - d["close"].shift(1)).abs()
    tr3 = (d["low"]  - d["close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    d["ATR14"] = tr.rolling(14).mean()
    d["ATR_pct"] = d["ATR14"] / d["close"]

    # Volume baselines & z-score
    d["vol_ma10"] = d["volume"].rolling(10).mean()
    d["vol_ma20"] = d["volume"].rolling(20).mean()
    d["vol_ma50"] = d["volume"].rolling(50).mean()
    vol_std50 = d["volume"].rolling(50).std()
    d["vol_z"] = (d["volume"] - d["vol_ma50"]) / vol_std50

    # Consolidation range over window
    w = 15
    d["range_pct_w"] = (d["high"].rolling(w).max() - d["low"].rolling(w).min()) / d["close"]

    # Helper flags
    d["up_candle"] = d["close"] > d["open"]
    d["down_candle"] = d["close"] < d["open"]

    # Expose MFV for later windows
    d["MFV"] = mfv

    return d


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Return per-day slope over rolling window via simple linear fit."""
    y = series.values.astype(float)
    idx = np.arange(len(series))
    out = np.full(len(series), np.nan)
    if window < 2:
        return pd.Series(out, index=series.index)
    for i in range(window-1, len(series)):
        xw = idx[i-window+1:i+1]
        yw = y[i-window+1:i+1]
        xw = xw - xw.mean()  # center
        denom = (xw**2).sum()
        if denom == 0:
            continue
        slope = (xw*yw).sum() / denom
        out[i] = slope
    return pd.Series(out, index=series.index)


# =========================
# Accumulation/Distribution flags & zones
# =========================

def flag_accumulation_distribution(d: pd.DataFrame) -> pd.DataFrame:
    """Mark daily accumulation/distribution days using CMF/ADL/volume heuristics."""
    df = d.copy()
    cmf = df["CMF20"].fillna(0)
    adl_slope10 = rolling_slope(df["ADL"], 10)
    vol_spike = df["vol_z"] > 1.0

    # Accumulation: positive CMF, rising ADL, up candle with meaningful volume
    acc = (
        (cmf > 0) & (adl_slope10 > 0) & (df["up_candle"]) &
        ((df["volume"] > 1.2*df["vol_ma20"]) | vol_spike)
    )

    # Distribution: negative CMF, falling ADL, down candle with meaningful volume
    dist = (
        (cmf < 0) & (adl_slope10 < 0) & (df["down_candle"]) &
        ((df["volume"] > 1.2*df["vol_ma20"]) | vol_spike)
    )

    df["acc_day"] = acc.fillna(False)
    df["dist_day"] = dist.fillna(False)
    return df


def detect_zones(d: pd.DataFrame,
                 range_thresh: float = 0.08,
                 min_len: int = 5,
                 lookback: int = 250) -> List[Dict[str, Any]]:
    """Detect consolidation zones (recent window only) and classify.

    - A zone exists where 15-day highest-high vs lowest-low range is ≤ range_thresh
      and ATR_pct is modest ("quiet").
    - Classification uses CMF/ADL slope sign over the zone.
    - Only scans the most recent `lookback` trading days to avoid ancient zones.
    """
    df = d.tail(lookback).copy()
    in_zone = (df["range_pct_w"] <= range_thresh) & (df["ATR_pct"] < 0.04)

    zones: List[Dict[str, Any]] = []
    i = 0
    n = len(df)
    while i < n:
        if not bool(in_zone.iloc[i]):
            i += 1
            continue
        start = i
        while i < n and bool(in_zone.iloc[i]):
            i += 1
        end = i - 1
        if end - start + 1 < min_len:
            continue
        window = df.iloc[start:end+1]
        z_low = window["low"].min()
        z_high = window["high"].max()
        cmf_mean = window["CMF20"].mean()
        adl_slope = (window["ADL"].iloc[-1] - window["ADL"].iloc[0]) / max(1, end-start)

        if cmf_mean > 0 and adl_slope > 0:
            z_type = "Accumulation"
        elif cmf_mean < 0 and adl_slope < 0:
            z_type = "Distribution"
        else:
            z_type = "Neutral"

        zones.append({
            "start": window.index[0],
            "end": window.index[-1],
            "low": float(z_low),
            "high": float(z_high),
            "type": z_type,
            "cmf_mean": float(cmf_mean),
            "adl_slope": float(adl_slope),
        })
    return zones


def detect_flow_windows(d: pd.DataFrame,
                        lookback: int = 180,
                        window: int = 5,
                        min_len: int = 3) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Identify recent *multi-day* accumulation/distribution windows.

    Heuristics:
    - Acc window: CMF>0, ADL rising, MFV_5d sum > 0 and above its 50d mean
    - Dist window: CMF<0, ADL falling, MFV_5d sum < 0 and below its 50d mean

    Returns (last_acc_window, last_dist_window) dicts or None.
    """
    df = d.tail(lookback).copy()
    mfv_5 = df["MFV"].rolling(window).sum()
    mfv_5_ma50 = mfv_5.rolling(50).mean()
    cmf = df["CMF20"].fillna(0)
    adl_slope10 = rolling_slope(df["ADL"], 10)

    acc_mask = (cmf > 0) & (adl_slope10 > 0) & (mfv_5 > 0) & (mfv_5 > mfv_5_ma50)
    dist_mask = (cmf < 0) & (adl_slope10 < 0) & (mfv_5 < 0) & (mfv_5 < mfv_5_ma50)

    def _windows_from_mask(mask: pd.Series) -> List[Tuple[int,int]]:
        out: List[Tuple[int,int]] = []
        i = 0
        vals = mask.fillna(False).values
        while i < len(vals):
            if not vals[i]:
                i += 1
                continue
            s = i
            while i < len(vals) and vals[i]:
                i += 1
            e = i-1
            if e - s + 1 >= min_len:
                out.append((s,e))
        return out

    acc_windows = _windows_from_mask(acc_mask)
    dist_windows = _windows_from_mask(dist_mask)

    def _mk(window_idx: Tuple[int,int]) -> Dict[str, Any]:
        s, e = window_idx
        w = df.iloc[s:e+1]
        return {
            "start": w.index[0],
            "end": w.index[-1],
            "days": int(e - s + 1),
            "mfv_sum": float(w["MFV"].sum()),
            "cmf_mean": float(w["CMF20"].mean()),
            "adl_slope": float((w["ADL"].iloc[-1] - w["ADL"].iloc[0]) / max(1, e-s)),
        }

    last_acc = _mk(acc_windows[-1]) if acc_windows else None
    last_dist = _mk(dist_windows[-1]) if dist_windows else None
    return last_acc, last_dist


def classify_flow_now(d: pd.DataFrame) -> Tuple[str, float, float]:
    """Classify current flow as Investing / Pulling Out / Neutral.

    Uses latest CMF sign, ADL 10-slope, and 10-day MFV sum.
    Returns (label, cmf_last, mfv10_sum).
    """
    cmf_last = float(d["CMF20"].iloc[-1]) if not d["CMF20"].isna().iloc[-1] else 0.0
    adl_slope10_last = float(rolling_slope(d["ADL"], 10).iloc[-1])
    mfv10 = float(d["MFV"].tail(10).sum())

    if cmf_last > 0 and adl_slope10_last > 0 and mfv10 > 0:
        return ("Investing", cmf_last, mfv10)
    if cmf_last < 0 and adl_slope10_last < 0 and mfv10 < 0:
        return ("Pulling Out", cmf_last, mfv10)
    return ("Neutral", cmf_last, mfv10)


def predict_next_zone(d: pd.DataFrame) -> Tuple[str, float, Tuple[int, int]]:
    """Heuristic prediction of the next likely Accumulation or Distribution zone.

    Returns (label, confidence, eta_days_range). label in {"Accumulation","Distribution","None"}.
    """
    df = d.copy()
    cmf = df["CMF20"].fillna(0)
    cmf_delta5 = cmf - cmf.shift(5)
    adl_slope10 = rolling_slope(df["ADL"], 10)

    # Near support/resistance via BB and 50-day extremes
    near_support = (df["close"] <= df["BB_mid"]) & (df["close"] <= df["BB_lo"] * 1.05)
    near_resist  = (df["close"] >= df["BB_mid"]) & (df["close"] >= df["BB_up"] * 0.95)
    low50 = df["close"].rolling(50).min()
    high50 = df["close"].rolling(50).max()
    near_50L = df["close"] <= (low50 * 1.05)
    near_50H = df["close"] >= (high50 * 0.95)

    # Volume conditions
    vol_dryup = df["vol_ma10"] < (df["vol_ma50"] * 0.9)
    vol_spike = df["vol_z"] > 1.0

    # Consolidation tightening
    tight = (df["range_pct_w"] < 0.06) & (df["ATR_pct"] < 0.03)

    acc_signals = [
        (cmf.iloc[-1] > 0),
        (cmf_delta5.iloc[-1] > 0),
        (adl_slope10.iloc[-1] > 0),
        bool(near_support.iloc[-1] or near_50L.iloc[-1]),
        bool(tight.iloc[-1] and vol_dryup.iloc[-1])
    ]
    dist_signals = [
        (cmf.iloc[-1] < 0),
        (cmf_delta5.iloc[-1] < 0),
        (adl_slope10.iloc[-1] < 0),
        bool(near_resist.iloc[-1] or near_50H.iloc[-1]),
        bool(tight.iloc[-1] and vol_spike.iloc[-1])
    ]

    acc_score = sum(int(x) for x in acc_signals) / len(acc_signals)
    dist_score = sum(int(x) for x in dist_signals) / len(dist_signals)

    if acc_score > dist_score and acc_score >= 0.4:
        label = "Accumulation"
        conf = acc_score
    elif dist_score > acc_score and dist_score >= 0.4:
        label = "Distribution"
        conf = dist_score
    else:
        return ("None", 0.0, (0, 0))

    # Rough ETA window from tightness + score
    if tight.iloc[-1]:
        eta = (3, 10) if conf >= 0.6 else (7, 20)
    else:
        eta = (10, 30)

    return (label, float(conf), eta)


# =========================
# Time-to-threshold logic
# =========================

def time_to_thresholds(closes: pd.Series, thresholds: List[float]) -> Tuple[Dict[float, List[int]], Dict[float, List[int]]]:
    n = len(closes)
    down_hits = {p: [] for p in thresholds}
    up_hits   = {p: [] for p in thresholds}
    c = closes.values
    for i in range(n - 1):
        base = c[i]
        down_targets = {p: base * (1.0 - p) for p in thresholds}
        up_targets   = {p: base * (1.0 + p) for p in thresholds}
        remaining_down = set(thresholds)
        remaining_up   = set(thresholds)
        for j in range(i + 1, n):
            price = c[j]
            if remaining_down:
                to_remove = [p for p in remaining_down if price <= down_targets[p]]
                for p in to_remove:
                    down_hits[p].append(j - i)
                    remaining_down.remove(p)
            if remaining_up:
                to_remove = [p for p in remaining_up if price >= up_targets[p]]
                for p in to_remove:
                    up_hits[p].append(j - i)
                    remaining_up.remove(p)
            if not remaining_down and not remaining_up:
                break
    return down_hits, up_hits


def summarize_hits(hit_dict: Dict[float, List[int]]) -> Dict[str, float]:
    out = {}
    for p in THRESHOLDS:
        key = f"{int(p*100)}%"
        vals = hit_dict.get(p, [])
        if len(vals) == 0:
            out[key] = float('nan')
        else:
            out[key] = float(sum(vals)) / float(len(vals))
    return out


# =========================
# IB connection & history
# =========================

def connect_ib() -> IB:
    ib = IB()
    print(f"Connecting to IB at {IB_HOST}:{IB_PORT}, clientId={IB_CLIENT_ID} ...")
    ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, readonly=True, timeout=10)
    if not ib.isConnected():
        raise RuntimeError("Failed to connect to IB.")
    return ib


def fetch_history(ib: IB, symbol: str, params: Params) -> pd.DataFrame:
    contract = Stock(symbol, "SMART", "USD")
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=f"{params.years} Y",
        barSizeSetting="1 day",
        whatToShow=params.what,
        useRTH=params.rth,
        formatDate=1,
        keepUpToDate=False,
        chartOptions=[],
    )
    if not bars:
        raise RuntimeError(f"No historical data returned for {symbol}.")
    df = util.df(bars)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df[["date","open","high","low","close","volume"]].dropna()
    df = df.set_index("date").sort_index()
    return df


# =========================
# Main
# =========================

def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute time-to-move + infer institutional flow, flow windows, zones, next-zone, and watchlists (IBKR).")
    parser.add_argument("--years", type=int, default=5, help="Years of daily history to fetch (default: 5).")
    parser.add_argument("--rth", type=int, choices=[0,1], default=0, help="Use Regular Trading Hours only (0/1, default: 0).")
    parser.add_argument("--what", type=str, default="ADJUSTED_LAST", help="whatToShow for IB historical data (default: ADJUSTED_LAST).")
    parser.add_argument("--zones_lookback", type=int, default=250, help="Days of history to scan for consolidation zones (default: 250).")
    parser.add_argument("--watch_conf", type=float, default=0.6, help="Min confidence to include a symbol in watchlists (default: 0.6).")
    parser.add_argument("--eta_max", type=int, default=15, help="Max ETA-high (days) to include in watchlists (default: 15).")
    args = parser.parse_args(argv)

    params = Params(years=args.years, rth=bool(args.rth), what=args.what,
                    zones_lookback=args.zones_lookback, watch_conf=args.watch_conf, eta_max=args.eta_max)

    ib = connect_ib()
    rows = []

    try:
        for symbol in symbols:
            try:
                df0 = fetch_history(ib, symbol, params)
                if len(df0) < params.min_rows:
                    print(f"[{symbol}] Not enough rows ({len(df0)}) to compute stable stats; skipping.")
                    continue

                d = add_indicators(df0)
                d = flag_accumulation_distribution(d)
                zones = detect_zones(d, lookback=params.zones_lookback)
                last_zone = zones[-1] if len(zones) else None
                last_acc_win, last_dist_win = detect_flow_windows(d)
                next_label, next_conf, eta = predict_next_zone(d)

                # "Flow now" + net MFV 30d
                flow_label, cmf_last, mfv10 = classify_flow_now(d)
                net_mfv_30 = float(d["MFV"].tail(30).sum())

                # Time-to-threshold
                down_hits, up_hits = time_to_thresholds(d['close'], THRESHOLDS)
                down_summary = summarize_hits(down_hits)
                up_summary   = summarize_hits(up_hits)

                # Recent flow summary
                last90 = d.tail(90)
                acc_days_90 = int(last90["acc_day"].sum())
                dist_days_90 = int(last90["dist_day"].sum())

                row = {
                    "symbol": symbol,
                    # Down (falls by X%)
                    "fall_5%":  down_summary["5%"],
                    "fall_10%": down_summary["10%"],
                    "fall_15%": down_summary["15%"],
                    "fall_20%": down_summary["20%"],
                    "fall_25%": down_summary["25%"],
                    # Up (moves up by X%)
                    "up_5%":    up_summary["5%"],
                    "up_10%":   up_summary["10%"],
                    "up_15%":   up_summary["15%"],
                    "up_20%":   up_summary["20%"],
                    "up_25%":   up_summary["25%"],
                    # Institutional flow proxies
                    "acc_days_90": acc_days_90,
                    "dist_days_90": dist_days_90,
                    # Last zone summary (recent-only)
                    "last_zone": (last_zone["type"] if last_zone else "None"),
                    "last_zone_start": (pd.to_datetime(last_zone["start"]).date() if last_zone else None),
                    "last_zone_end": (pd.to_datetime(last_zone["end"]).date() if last_zone else None),
                    # Flow windows (episodes)
                    "last_acc_start": (pd.to_datetime(last_acc_win["start"]).date() if last_acc_win else None),
                    "last_acc_end": (pd.to_datetime(last_acc_win["end"]).date() if last_acc_win else None),
                    "last_acc_days": (last_acc_win["days"] if last_acc_win else None),
                    "last_dist_start": (pd.to_datetime(last_dist_win["start"]).date() if last_dist_win else None),
                    "last_dist_end": (pd.to_datetime(last_dist_win["end"]).date() if last_dist_win else None),
                    "last_dist_days": (last_dist_win["days"] if last_dist_win else None),
                    # Prediction
                    "next_zone": next_label,
                    "next_conf": round(next_conf, 2),
                    "eta_lo": eta[0],
                    "eta_hi": eta[1],
                    # Flow now & net MFV
                    "flow_now": flow_label,
                    "cmf_last": round(cmf_last, 4),
                    "net_mfv_30": round(net_mfv_30, 2),
                    # Sample sizes
                    "n_fall_5%":  len(down_hits[0.05]),
                    "n_fall_10%": len(down_hits[0.10]),
                    "n_fall_15%": len(down_hits[0.15]),
                    "n_fall_20%": len(down_hits[0.20]),
                    "n_fall_25%": len(down_hits[0.25]),
                    "n_up_5%":    len(up_hits[0.05]),
                    "n_up_10%":   len(up_hits[0.10]),
                    "n_up_15%":   len(up_hits[0.15]),
                    "n_up_20%":   len(up_hits[0.20]),
                    "n_up_25%":   len(up_hits[0.25]),
                }
                rows.append(row)

                # Console summary line
                nz = f" last_zone={row['last_zone']}" if row['last_zone'] else ""
                pred = f" | next={row['next_zone']} (conf={row['next_conf']}, eta={row['eta_lo']}-{row['eta_hi']}d)" if row['next_zone'] != "None" else ""
                accw = (
                    f" | accWin={row['last_acc_start']}->{row['last_acc_end']}({row['last_acc_days']}d)" if row['last_acc_start'] else ""
                )
                distw = (
                    f" | distWin={row['last_dist_start']}->{row['last_dist_end']}({row['last_dist_days']}d)" if row['last_dist_start'] else ""
                )
                flow = f" | flow_now={row['flow_now']}"
                print(
                    f"[{symbol}] fall(avg days): 5%={row['fall_5%']:.1f}, 10%={row['fall_10%']:.1f}, 15%={row['fall_15%']:.1f}, 20%={row['fall_20%']:.1f}, 25%={row['fall_25%']:.1f} | "
                    f"up(avg days): 5%={row['up_5%']:.1f}, 10%={row['up_10%']:.1f}, 15%={row['up_15%']:.1f}, 20%={row['up_20%']:.1f}, 25%={row['up_25%']:.1f} | "
                    f"acc90={acc_days_90}, dist90={dist_days_90}{nz}{pred}{accw}{distw}{flow}"
                )

            except Exception as e:
                print(f"[{symbol}] Error: {e}", file=sys.stderr)
                continue

    finally:
        if ib.isConnected():
            ib.disconnect()

        if not rows:
            print("No results computed.")
            return 0

    # Final table
    df_out = pd.DataFrame(rows)
    ordered_cols = [
        "symbol",
        "fall_5%", "fall_10%", "fall_15%", "fall_20%", "fall_25%",
        "up_5%",   "up_10%",   "up_15%",   "up_20%",   "up_25%",
        "acc_days_90", "dist_days_90",
        "last_zone", "last_zone_start", "last_zone_end",
        "last_acc_start", "last_acc_end", "last_acc_days",
        "last_dist_start", "last_dist_end", "last_dist_days",
        "next_zone", "next_conf", "eta_lo", "eta_hi",
        "flow_now", "cmf_last", "net_mfv_30",
        "n_fall_5%", "n_fall_10%", "n_fall_15%", "n_fall_20%", "n_fall_25%",
        "n_up_5%",   "n_up_10%",   "n_up_15%",   "n_up_20%",   "n_up_25%",
    ]
    df_out = df_out[ordered_cols]

    pd.set_option('display.width', 260)
    pd.set_option('display.max_columns', None)
    print("=== Average Trading Days + Institutional Flow Windows, Zones & Flow-Now ===")
    print(df_out.to_string(index=False, float_format=lambda x: f"{x:.1f}" if isinstance(x, (int, float)) and not math.isnan(x) else str(x)))

    out_dir = os.path.dirname(__file__) or "."
    out_csv = os.path.join(out_dir, "time_to_move_results.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"Saved results to: {out_csv}")

    # Build watchlists
    acc_watch = df_out[(df_out["next_zone"] == "Accumulation") & (df_out["next_conf"] >= params.watch_conf) & (df_out["eta_hi"] <= params.eta_max)] \
                        .sort_values(["eta_hi", "next_conf"], ascending=[True, False])
    dist_watch = df_out[(df_out["next_zone"] == "Distribution") & (df_out["next_conf"] >= params.watch_conf) & (df_out["eta_hi"] <= params.eta_max)] \
                        .sort_values(["eta_hi", "next_conf"], ascending=[True, False])

    cols_watch = ["symbol", "next_conf", "eta_lo", "eta_hi", "flow_now", "acc_days_90", "dist_days_90", "last_zone"]

    if not acc_watch.empty:
        print("=== Watchlist: Next Accumulation (conf >= %.2f, eta_hi <= %dd) ===" % (params.watch_conf, params.eta_max))
        print(acc_watch[cols_watch].to_string(index=False))
        acc_csv = os.path.join(out_dir, "watchlist_next_accumulation.csv")
        acc_watch[cols_watch].to_csv(acc_csv, index=False)
        print(f"Saved: {acc_csv}")

    if not dist_watch.empty:
        print("=== Watchlist: Next Distribution (conf >= %.2f, eta_hi <= %dd) ===" % (params.watch_conf, params.eta_max))
        print(dist_watch[cols_watch].to_string(index=False))
        dist_csv = os.path.join(out_dir, "watchlist_next_distribution.csv")
        dist_watch[cols_watch].to_csv(dist_csv, index=False)
        print(f"Saved: {dist_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
