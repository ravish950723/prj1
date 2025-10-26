from datetime import date, datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from utils import (
    log_signal,
    log_confidence_score,
    get_sector_info,
    _passes_rr_filter,
)

# ---------------------------------------------------------------------------
# Simple in-memory cooldown registry
_signal_last_fired: Dict[Tuple[str, str], pd.Timestamp] = {}


def _cooldown(symbol: str, condition: str, ts: pd.Timestamp, days: int = 3) -> bool:
    """Return True if the (symbol, condition) fired within the last `days`."""
    try:
        t = pd.Timestamp(ts)
        last = _signal_last_fired.get((symbol, condition))
        if last is None:
            return False
        return (t - last).days < days
    except Exception:
        return False


def _mark_fired(symbol: str, condition: str, ts: pd.Timestamp) -> None:
    _signal_last_fired[(symbol, condition)] = pd.Timestamp(ts)


# ---------------------------------------------------------------------------
# Confidence scoring (lightweight heuristic; keep numerically stable)
def _score_confidence(
    df: pd.DataFrame,
    latest: pd.Series,
    signal_type: str,
    condition: str,
    sector_type: str,
) -> float:
    """
    Return 0..100 confidence score.
    Ingredients: ADX, extension vs 200-DEMA, MACD context, ATR%, sector bias.
    """
    try:
        price = float(latest["close"])
        adx = float(df["adx"].iloc[-1]) if "adx" in df.columns else 25.0
        ema200 = float(df["200_dema"].iloc[-1]) if "200_dema" in df.columns else price
        macd = float(df["macd"].iloc[-1]) if "macd" in df.columns else 0.0
        macd_sig = float(df["macd_signal"].iloc[-1]) if "macd_signal" in df.columns else 0.0
        atr = float(df["atr"].iloc[-1]) if "atr" in df.columns else 0.0
        atr_pct = atr / max(price, 1e-9)

        ext = price / max(ema200, 1e-9)  # extension factor
        macd_term = 1.0 if macd > macd_sig else 0.0
        adx_term = np.clip((adx - 15.0) / 20.0, 0.0, 1.0)
        ext_penalty = np.clip((ext - 1.15) / 0.15, 0.0, 1.0)
        vol_pen = np.clip((atr_pct - 0.05) / 0.10, 0.0, 1.0)

        sector_bias = 0.05 if sector_type == "etf" else 0.0

        base = 50.0 + 20.0 * adx_term + 15.0 * macd_term - 20.0 * ext_penalty - 15.0 * vol_pen
        base *= (1.0 + sector_bias)
        return float(np.clip(base, 0.0, 100.0))
    except Exception:
        return 50.0


# ---------------------------------------------------------------------------
# Breakout helper
def _breakout_context(df: pd.DataFrame, prior_20d_high: float) -> Tuple[float, float, float, float]:
    """
    Return (dist_pct, volr20, atr14, bb_upper) for the latest row.
    """
    latest = df.iloc[-1]
    vol_med20 = df["volume"].rolling(20).median().iloc[-1] if "volume" in df.columns else 0.0
    volr20 = (float(latest["volume"]) / float(vol_med20)) if vol_med20 else 0.0
    dist_pct = (float(latest["close"]) / max(1e-9, float(prior_20d_high)) - 1.0) * 100.0
    atr14 = float(df["atr"].iloc[-1]) if "atr" in df.columns else 0.0
    bb_upper = float(df["bb_upper"].iloc[-1]) if "bb_upper" in df.columns else float("inf")
    return float(dist_pct), float(volr20), float(atr14), float(bb_upper)


# ---------------------------------------------------------------------------
# SELL signals
def check_sell_signals(df: pd.DataFrame) -> List[str]:
    """
    Evaluate SELL signals on the latest bar and log any high-quality events.
    Returns list of action labels that fired.
    """
    actions: List[str] = []
    if df.empty:
        return actions

    latest = df.iloc[-1]
    symbol = latest.get("symbol", "UNKNOWN")
    date_str = str(date.today())

    # Example conditions (extend as needed)
    try:
        adx = float(df["adx"].iloc[-1]) if "adx" in df.columns else 25.0
        atr = float(df["atr"].iloc[-1]) if "atr" in df.columns else 0.0
        price = float(latest["close"])
        atr_pct = atr / max(price, 1e-9)
        sector_type, _, _, _ = get_sector_info(symbol)

        # Example: Overbought vs 200-DEMA by > 27%
        ema200 = float(df["200_dema"].iloc[-1]) if "200_dema" in df.columns else price
        overextended = price > 1.27 * ema200
        if overextended:
            actions.append("Overbought (>27% above 200 DEMA)")

        # Example: MACD bearish below zero
        macd = float(df["macd"].iloc[-1]) if "macd" in df.columns else 0.0
        macd_sig = float(df["macd_signal"].iloc[-1]) if "macd_signal" in df.columns else 0.0
        if (macd < macd_sig) and (macd < 0):
            actions.append("MACD Bearish below zero")

        # Log each action with confidence and position sizing
        for action in actions:
            conf = _score_confidence(df, latest, "SELL", action, sector_type)
            size_pct = max(0.5, min(5.0, conf / 20.0))  # 0.5..5.0% notional hint

            log_signal(
                symbol,
                date_str,
                action,
                round(conf, 1),
                signal_type="SELL",
                condition=action,
                ATRpct=round(atr_pct * 100, 2),
                SizePct=round(size_pct, 2),
                **{
                    "Pattern detected": latest.get("pattern") or latest.get("pattern_name") or "-",
                    "Breakout": "No",
                },
            )
    except Exception as e:
        print(f"[WARN] check_sell_signals error: {e}")

    if not actions:
        print("[INFO] No strong sell signals. Hold or monitor closely.")

    return actions


# ---------------------------------------------------------------------------
# BUY / WATCH signals (includes hardened breakout + retest)
def check_entry_signals(df: pd.DataFrame) -> None:
    if df.empty:
        print("[INFO] Empty dataframe for entry signals.")
        return

    latest = df.iloc[-1]
    symbol = latest.get("symbol", "UNKNOWN")
    today = str(date.today())
    sector_type, rsi_base, vol_threshold, weight = get_sector_info(symbol)

    # --- Precompute helpers ---
    adx = float(df["adx"].iloc[-1]) if "adx" in df.columns else 25.0
    rsi_now = float(latest["rsi"]) if "rsi" in latest else 50.0
    macd_hist = (df["macd"] - df["macd_signal"]) if {"macd", "macd_signal"} <= set(df.columns) else pd.Series([0.0])
    macd_hist_uptick = (macd_hist.iloc[-1] > macd_hist.shift(1).iloc[-1]) if len(macd_hist) >= 2 else False

    macd_up = latest.get("macd", 0.0) > latest.get("macd_signal", 0.0)
    macd_ok = bool(macd_up or macd_hist_uptick)

    above_50 = latest["close"] > latest["50_dema"]
    above_200 = latest["close"] > latest["200_dema"]
    above_mas = above_50 and above_200

    # Volume normalization (z-score)
    vol = df["volume"].astype(float)
    vol_mean = vol.rolling(20).mean()
    vol_std = vol.rolling(20).std().replace(0, np.nan)
    vol_z = ((vol - vol_mean) / vol_std).iloc[-1] if vol_std.iloc[-1] == vol_std.iloc[-1] else 0.0  # safe NaN check

    price = float(latest["close"])
    atr = float(df["atr"].iloc[-1]) if "atr" in df.columns else 0.0
    atr_pct = atr / max(price, 1e-9)

    # Dynamic volume threshold with ETF relief
    base_min_z = 1.4 if sector_type != "etf" else 1.15
    if atr_pct <= 0.02:
        base_min_z += 0.15
    min_z = base_min_z
    volume_spike = vol_z >= min_z

    # allow "good enough" volume with supportive trend & vol
    vol_ok = volume_spike or (atr_pct >= 0.06 and adx >= 21 and vol_z >= -0.25)

    # ADX gates
    adx_gate_buy = 20 if sector_type == "etf" else 22
    adx_momo_gate = 21 if sector_type == "etf" else 23

    # Dynamic RSI threshold
    dyn_base = rsi_base  # passed in via sector map
    dyn_adj = 0.0
    if macd_hist_uptick:
        dyn_adj += 1.0
    if sector_type == "etf":
        dyn_adj += 1.0
    dynamic_rsi_threshold = float(dyn_base + dyn_adj)

    # Momentum helpers
    rsi_prev2 = df["rsi"].shift(2).iloc[-1] if len(df) >= 3 else rsi_now
    rsi_rising = rsi_now > float(rsi_prev2)
    not_overextended = price <= 1.27 * latest["200_dema"]

    # 2-bar near reclaim (yesterday)
    y_close = df["close"].shift(1).iloc[-1]
    y_50 = df["50_dema"].shift(1).iloc[-1]
    y_200 = df["200_dema"].shift(1).iloc[-1]
    y_macd = df["macd"].shift(1).iloc[-1]
    y_macd_sig = df["macd_signal"].shift(1).iloc[-1]
    y_adx = df["adx"].shift(1).iloc[-1] if "adx" in df.columns else adx
    y_hist = macd_hist.shift(1).iloc[-1] if len(macd_hist) >= 2 else 0.0
    yy_hist = macd_hist.shift(2).iloc[-1] if len(macd_hist) >= 3 else 0.0
    y_hist_uptick = y_hist > yy_hist

    y_near_reclaim = (
        (y_close > 0.985 * y_50)
        and (y_close > y_200)
        and ((y_macd > y_macd_sig) or y_hist_uptick)
        and (y_adx >= (adx_gate_buy - 1))
    )

    # BUY paths
    primary_buy = (
        (rsi_now < dynamic_rsi_threshold) and macd_ok and above_mas and vol_ok and not_overextended
    )

    pattern = latest.get("pattern") or latest.get("pattern_name")
    near_50_now = latest["close"] > 0.985 * latest["50_dema"]
    pattern_ok = (
        (pattern in ["Double Bottom", "Triple Bottom", "Cup and Handle", "Ascending Triangle"])
        and (above_200)
        and ((above_50) or (near_50_now and macd_ok and vol_ok))
        and (adx >= (adx_gate_buy - (1 if sector_type == "etf" else 0)))
        and not_overextended
    )

    dip_buy = (
        above_200
        and above_50
        and (df["close"].shift(1).iloc[-1] < df["50_dema"].shift(1).iloc[-1])
        and macd_ok
        and vol_ok
        and not_overextended
    )

    # Hardened breakout
    prior_20d_high = df["high"].rolling(20).max().shift(1).iloc[-1]
    dist_pct, volr20, atr14, bb_upper = _breakout_context(df, prior_20d_high)

    max_breakout_pct = 4.0
    not_too_far = dist_pct <= max_breakout_pct

    need_event_vol = atr_pct < 0.06
    event_vol_ok = (volr20 >= 1.35) if need_event_vol else (volr20 >= 1.15)

    not_above_band = price <= (bb_upper + 0.25 * atr14)

    breakout_buy = (
        (price > prior_20d_high)
        and macd_ok
        and (adx >= adx_gate_buy)
        and vol_ok
        and event_vol_ok
        and not_overextended
        and not_too_far
        and not_above_band
    )

    # Retest ("throwback") entry
    y_prior20 = df["high"].rolling(20).max().shift(2).iloc[-1]
    recent_broke = df["close"].shift(1).iloc[-1] > y_prior20
    pulled_back = abs(price - prior_20d_high) <= (0.8 * atr14 if atr14 else 0.0)
    volume_eased = True
    if len(df) >= 2:
        volume_eased = df["volume"].iloc[-1] <= df["volume"].shift(1).iloc[-1] * 0.75

    retest_buy = recent_broke and pulled_back and volume_eased and macd_ok and (adx >= adx_gate_buy) and above_mas

    # Failed breakout cooldown
    failed_breakout = (df["close"].shift(1).iloc[-1] > prior_20d_high) and (price < prior_20d_high)
    if failed_breakout:
        _mark_fired(symbol, "FAILED_BREAKOUT_COOLDOWN", latest.name)
        if _cooldown(symbol, "FAILED_BREAKOUT_COOLDOWN", latest.name, days=3):
            breakout_buy = False
            retest_buy = False
            print("[FILTER] Failed breakout → cooldown active.")

    # Context logging
    try:
        print(
            f"[CTX] breakout prior20={prior_20d_high:.2f} close={price:.2f} "
            f"dist={dist_pct:.2f}% volx20={volr20:.2f} ADX={adx:.1f} "
            f"MACD_ok={macd_ok} vol_ok={bool(vol_ok)} evt_vol_ok={event_vol_ok}"
        )
    except Exception:
        pass

    # Momentum & confirm-reclaim
    momentum_buy = (
        above_mas and macd_ok and (adx >= adx_momo_gate) and rsi_rising and (38 <= rsi_now <= 68) and vol_ok and not_overextended
    )

    confirm_reclaim_buy = y_near_reclaim and above_200 and above_50 and macd_ok and vol_ok and not_overextended

    # --- Fire BUY if any path passes ---
    if primary_buy or pattern_ok or dip_buy or breakout_buy or momentum_buy or confirm_reclaim_buy or retest_buy:
        reason = (
            "Breakout: 20D high + MACD/Hist uptick + ADX + vol ok" if breakout_buy
            else "Retest: throwback hold + MACD/ADX + vol ok" if retest_buy
            else "Momentum" if momentum_buy
            else "Pattern"
            if pattern_ok
            else "DipReclaim" if dip_buy
            else "ConfirmReclaim" if confirm_reclaim_buy
            else "Primary"
        )

        if not _passes_rr_filter(df, latest):
            print("[FILTER] R:R gate blocked the BUY.")
            return

        if _cooldown(symbol, reason, latest.name, days=3):
            print("[FILTER] Duplicate BUY suppressed (cooldown).")
            return

        _mark_fired(symbol, reason, latest.name)

        # Confidence
        conf = _score_confidence(df, latest, "BUY", reason, sector_type)

        # Pattern & breakout flags for logging
        pattern_label = pattern or latest.get("pattern") or "-"
        if isinstance(pattern_label, (list, tuple)):
            pattern_label = "/".join(str(p) for p in pattern_label if p)
        breakout_flag = bool(breakout_buy or latest.get("breakout", False))

        log_signal(
            symbol,
            today,
            reason,
            round(conf, 1),
            signal_type="BUY",
            condition=(
                "Momentum"
                if momentum_buy
                else "Breakout"
                if breakout_buy
                else "DipReclaim"
                if dip_buy
                else "Pattern"
                if pattern_ok
                else "ConfirmReclaim"
                if confirm_reclaim_buy
                else "Continuation"
            ),
            ATRpct=round(atr_pct * 100, 2),
            SizePct=round(max(0.5, min(5.0, conf / 20.0)), 2),
            **{
                "Pattern detected": pattern_label,
                "Breakout": "Yes" if breakout_flag else "No",
            },
        )

        log_confidence_score(conf, "BUY confidence")
        return

    # --- Near-miss WATCH ----------------------------------------------------
    passed = {
        "vol_ok": bool(vol_ok),
        "macd_ok": bool(macd_ok),
        "adx_ok": bool(adx >= (adx_gate_buy - 1)),
        "above_200": bool(above_200),
        "near_50": bool(latest["close"] > 0.985 * latest["50_dema"]),
    }
    score = sum(passed.values())
    if score >= 3 and passed["above_200"] and passed["near_50"]:
        watch_reason = "Watch: Near 50-DEMA with MACD/ADX/Vol OK"
        if not _cooldown(symbol, watch_reason, latest.name, days=2):
            conf = 45.0
            pattern_label = pattern or "-"
            log_signal(
                symbol,
                today,
                watch_reason,
                round(conf, 1),
                signal_type="WATCH",
                condition="Near50",
                ATRpct=round(atr_pct * 100, 2),
                SizePct=0.0,
                **{
                    "Pattern detected": pattern_label,
                    "Breakout": "No",
                },
            )
            log_confidence_score(conf, "WATCH confidence")


# ---------------------------------------------------------------------------
# Compatibility helpers expected by sell.py

# Try to import a fetcher for weekly trend computation
try:
    from data_fetcher import fetch_historical_data_fresh as _fetch_hist
except Exception:
    try:
        from data_fetcher import fetch_historical_data as _fetch_hist
    except Exception:
        _fetch_hist = None

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()

def check_weekly_trend(symbol: str) -> bool:
    """
    Heuristic weekly trend detector:
    - Pull daily data (3Y) if available
    - Resample to W-FRI close
    - Compute 10W/30W EMA (approx via weekly closes)
    - Uptrend if close > ema30 and ema10 > ema30 (with 2-week confirmation)
    Returns True (Bullish) or False (Bearish/Neutral).
    """
    try:
        if _fetch_hist is None:
            return True  # fail-open to avoid breaking pipeline
        df = _fetch_hist(symbol, duration="3 Y", bar_size="1 day")
        if df is None or df.empty:
            return True
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
            else:
                df.index = pd.to_datetime(df.index)
        w = df.sort_index().resample("W-FRI").last()
        w = w.dropna(subset=["close"])
        w["ema10w"] = _ema(w["close"], span=10)
        w["ema30w"] = _ema(w["close"], span=30)
        if len(w) < 35:
            return True
        c, e10, e30 = w["close"].iloc[-1], w["ema10w"].iloc[-1], w["ema30w"].iloc[-1]
        c1, e10_1, e30_1 = w["close"].iloc[-2], w["ema10w"].iloc[-2], w["ema30w"].iloc[-2]
        bullish = (c > e30 and e10 > e30) and (c1 > e30_1 and e10_1 > e30_1)
        return bool(bullish)
    except Exception as e:
        print(f"[WARN] check_weekly_trend({symbol}) fallback due to error: {e}")
        return True

def apply_trailing_stop(df: pd.DataFrame, atr_mult: float = 3.0) -> float:
    """
    Simple ATR-based trailing stop:
    stop = last_close - atr_mult * ATR(14)
    Returns a float; if ATR missing, returns last_close * 0.9 as a conservative fallback.
    """
    try:
        last_close = float(df["close"].iloc[-1])
        atr = float(df["atr"].iloc[-1]) if "atr" in df.columns else None
        if atr is None or not np.isfinite(atr):
            return round(last_close * 0.9, 2)
        return round(last_close - atr_mult * atr, 2)
    except Exception:
        return float("nan")

def check_exit_signals(df: pd.DataFrame) -> None:
    """
    Placeholder for explicit EXIT signals (e.g., stop hits, bearish crossovers).
    Kept lightweight to preserve pipeline compatibility.
    """
    try:
        # Example: print & no-op; real rules can be added here.
        pass
    except Exception as e:
        print(f"[WARN] check_exit_signals error: {e}")
