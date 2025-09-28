"""
Complete analysis script with Excel export:
- Connects to IBKR (ib_insync)
- Fetches ~3Y daily bars
- Prints % up/down vs last 52w/26w/14w highs and the high prices/dates
- Computes average daily volatility (std of daily returns) for last 52w and 26w
- Also computes volatility from the last-N-week high → today
- Converts volatility to a trailing stop % using a multiplier k (default 2.0)
- Counts non-overlapping drawdown events over last 52 weeks
- **Dumps results to Excel** (Summary, Stops, Segments, Drawdowns sheets)

Requires a local config.py with:
  symbols = ["AAPL", "MSFT", ...]
  IB_HOST = "127.0.0.1"
  IB_PORT = 7497
  IB_CLIENT_ID = 103

Run:
  python price_deviation_analysis_full.py
"""
from __future__ import annotations
from datetime import datetime, timedelta
from typing import Optional, Tuple, List

import pandas as pd
import numpy as np
from ib_insync import IB, Stock, util

# Pull symbols + IB settings from config.py
from config import symbols, IB_HOST, IB_PORT, IB_CLIENT_ID

# ==========================
# Excel collectors (in-memory tables)
# ==========================
SUMMARY_ROWS: List[dict] = []   # one row per symbol (current, rolling highs + dates + %)
SEGMENT_ROWS: List[dict] = []   # non-overlapping (0–14, 14–26, 26–52)
DRAWDOWN_ROWS: List[dict] = []  # non-overlapping drawdown counts in last 52w
STOPS_ROWS: List[dict] = []     # returns-vol, ATR%, chosen; from-high→today vols

# =============================================================
# Connection helper
# =============================================================

def connect_ib() -> IB:
    """Create and return a connected IB instance."""
    ib = IB()
    ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)
    return ib

# =============================================================
# Data helpers
# =============================================================

def fetch_stock_data(ib: IB, symbol: str) -> pd.DataFrame:
    """
    Fetch ~3 years of daily bars from IBKR and return a DataFrame with 'date' ascending.
    Columns include: date, open, high, low, close, volume.
    """
    contract = Stock(symbol, 'SMART', 'USD')
    ib.qualifyContracts(contract)
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr='3 Y',
        barSizeSetting='1 day',
        whatToShow='TRADES',
        useRTH=True,
        formatDate=1,
    )
    df = util.df(bars)
    if df.empty:
        return df
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').reset_index(drop=True)


def last_n_weeks(df: pd.DataFrame, weeks: int) -> pd.DataFrame:
    if df.empty or 'date' not in df.columns:
        return df
    end = df['date'].max()
    start = end - timedelta(weeks=weeks)
    return df[df['date'] >= start].copy()


def last_52_weeks(df: pd.DataFrame) -> pd.DataFrame:
    return last_n_weeks(df, 52)

# =============================================================
# Drawdown counting (optional but useful context)
# =============================================================

def _count_drawdowns_non_overlapping(df: pd.DataFrame, threshold_pct: float) -> int:
    """
    Count non-overlapping drawdowns where close falls by >= threshold_pct from the most recent peak
    before a new high is made (last 52 weeks only).
    """
    if df.empty or 'close' not in df.columns:
        return 0

    d = last_52_weeks(df)
    if d.empty or len(d) < 2:
        return 0

    closes = d['close'].astype(float).values
    peak = closes[0]
    counted_for_this_peak = False
    events = 0

    for price in closes[1:]:
        if price > peak:
            peak = price
            counted_for_this_peak = False
        else:
            drawdown_pct = (peak - price) / peak * 100.0
            if (drawdown_pct >= threshold_pct) and (not counted_for_this_peak):
                events += 1
                counted_for_this_peak = True
    return events


def count_falls_5pct(df):  return _count_drawdowns_non_overlapping(df, 5.0)

def count_falls_10pct(df): return _count_drawdowns_non_overlapping(df, 10.0)

def count_falls_15pct(df): return _count_drawdowns_non_overlapping(df, 15.0)

def count_falls_20pct(df): return _count_drawdowns_non_overlapping(df, 20.0)

def count_falls_25pct(df): return _count_drawdowns_non_overlapping(df, 25.0)

def count_falls_35pct(df): return _count_drawdowns_non_overlapping(df, 35.0)

def count_falls_40pct(df): return _count_drawdowns_non_overlapping(df, 40.0)

# =============================================================
# Period highs & deviations (rolling windows)
# =============================================================

def _high_and_pct_from_high(df: pd.DataFrame, weeks: int) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (period_high_price, pct_diff_from_high) over the last `weeks`.
    pct_diff = (current_close - period_high)/period_high * 100
      > 0 => above that high (rare intraday), < 0 => below that high.
    """
    if df.empty or not {'date','high','close'}.issubset(df.columns):
        return None, None
    window = last_n_weeks(df, weeks)
    if window.empty:
        return None, None
    period_high = float(window['high'].max())
    if period_high == 0:
        return None, None
    current_price = float(df['close'].iloc[-1])
    pct_from_high = (current_price - period_high) / period_high * 100.0
    return period_high, pct_from_high


def pct_from_52w_high(df: pd.DataFrame) -> Optional[float]:
    _h, pct = _high_and_pct_from_high(df, 52)
    return pct


def pct_from_26w_high(df: pd.DataFrame) -> Optional[float]:
    _h, pct = _high_and_pct_from_high(df, 26)
    return pct


def pct_from_14w_high(df: pd.DataFrame) -> Optional[float]:
    _h, pct = _high_and_pct_from_high(df, 14)
    return pct

# =============================================================
# Volatility & trailing-stop functions
# =============================================================

def avg_volatility_pct(df: pd.DataFrame, weeks: int) -> Optional[float]:
    """
    Average *daily* volatility (%) over the last `weeks`, measured as the
    standard deviation of daily close-to-close returns within that window.
    """
    if df.empty or 'close' not in df.columns:
        return None
    window = last_n_weeks(df, weeks)
    if window.empty or len(window) < 2:
        return None
    returns = window['close'].astype(float).pct_change().dropna()
    if returns.empty:
        return None
    # daily volatility in percent
    return float(returns.std() * 100.0)


def trailing_stop_pct_from_vol(vol_pct: Optional[float], k: float = 2.0) -> Optional[float]:
    """
    Convert a volatility estimate (daily %) into a trailing stop % using multiplier k.
    Example: if vol_pct=2.0 and k=2 -> trailing stop = 4.0% of price.
    """
    if vol_pct is None:
        return None
    return float(k * vol_pct)

# =============================================================
# Volatility (ATR%) optional helper
# =============================================================

def atr_percent_over_window(df: pd.DataFrame, weeks: int, atr_len: int = 14) -> Optional[float]:
    """Compute average ATR% over the last `weeks` (ATR_len default 14).
    ATR% is averaged as (ATR / close) * 100 across the window.
    """
    if df.empty or not {'high','low','close','date'}.issubset(df.columns):
        return None
    w = last_n_weeks(df, weeks)
    if w.empty or len(w) < atr_len + 1:
        return None
    h = w['high'].astype(float).values
    l = w['low'].astype(float).values
    c = w['close'].astype(float).values

    # True Range series
    prev_close = np.roll(c, 1)
    prev_close[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_close), np.abs(l - prev_close)))

    # Wilder's ATR via EMA(alpha=1/atr_len)
    alpha = 1.0 / float(atr_len)
    atr = []
    ema = tr[0]
    for x in tr:
        ema = alpha * x + (1 - alpha) * ema
        atr.append(ema)
    atr = np.array(atr)

    atr_pct_series = (atr / c) * 100.0
    return float(pd.Series(atr_pct_series).iloc[-atr_len:].mean())

# =============================================================
# Printing helpers (rolling highs + stops)
# =============================================================

def print_pct_from_52w_high(df: pd.DataFrame) -> None:
    high, pct = _high_and_pct_from_high(df, 52)
    print("🔺 52-week high comparison:")
    if high is None:
        print("   • Last 52-week high price: N/A")
        print("   • % vs last 52-week high: N/A")
        return
    print(f"   • Last 52-week high price: {high:.2f}")
    print(f"   • % vs last 52-week high: {pct:+.2f}%")


def print_pct_from_26w_high(df: pd.DataFrame) -> None:
    high, pct = _high_and_pct_from_high(df, 26)
    print("🔺 26-week high comparison:")
    if high is None:
        print("   • Last 26-week high price: N/A")
        print("   • % vs last 26-week high: N/A")
        return
    print(f"   • Last 26-week high price: {high:.2f}")
    print(f"   • % vs last 26-week high: {pct:+.2f}%")


def print_pct_from_14w_high(df: pd.DataFrame) -> None:
    high, pct = _high_and_pct_from_high(df, 14)
    print("🔺 14-week high comparison:")
    if high is None:
        print("   • Last 14-week high price: N/A")
        print("   • % vs last 14-week high: N/A")
        return
    print(f"   • Last 14-week high price: {high:.2f}")
    print(f"   • % vs last 14-week high: {pct:+.2f}%")


def _print_stop_details(window_label: str, current_price: float, vol_ret_pct: Optional[float], k: float, also_atr_pct: Optional[float] = None) -> Optional[float]:
    """Helper to print return-volatility stop, optional ATR%-based stop, and return chosen stop% (tighter of available)."""
    print(f"📊 {window_label} average volatility & trailing stop:")
    chosen = None

    if vol_ret_pct is None:
        print("   • Avg daily volatility (returns): N/A")
    else:
        ts = trailing_stop_pct_from_vol(vol_ret_pct, k)
        stop_price = current_price * (1 - ts/100.0)
        print(f"   • Avg daily volatility (returns): {vol_ret_pct:.2f}% → Stop ({k:.1f}×): {ts:.2f}%  ⇒ Stop px: {stop_price:.2f}")
        chosen = ts

    if also_atr_pct is not None:
        ts_atr = trailing_stop_pct_from_vol(also_atr_pct, k)
        stop_price_atr = current_price * (1 - ts_atr/100.0)
        print(f"   • Avg ATR%% (price-based): {also_atr_pct:.2f}% → Stop ({k:.1f}×): {ts_atr:.2f}% ⇒ Stop px: {stop_price_atr:.2f}")
        if chosen is None or ts_atr < chosen:
            chosen = ts_atr

    if chosen is not None:
        print(f"   • Chosen trailing stop: {chosen:.2f}% (tighter of available)")
    else:
        print("   • Chosen trailing stop: N/A")
    return chosen


def print_volatility_and_trailing_stop_52w(df: pd.DataFrame, k: float = 2.0) -> None:
    current_price = float(df['close'].iloc[-1]) if not df.empty else float('nan')
    vol = avg_volatility_pct(df, 52)
    atrp = atr_percent_over_window(df, 52, atr_len=14)
    _print_stop_details("52-week", current_price, vol, k, also_atr_pct=atrp)


def print_volatility_and_trailing_stop_26w(df: pd.DataFrame, k: float = 2.0) -> None:
    current_price = float(df['close'].iloc[-1]) if not df.empty else float('nan')
    vol = avg_volatility_pct(df, 26)
    atrp = atr_percent_over_window(df, 26, atr_len=14)
    _print_stop_details("26-week", current_price, vol, k, also_atr_pct=atrp)

# =============================================================
# Segmented (non-overlapping) highs to avoid nested equality
# =============================================================

def high_and_pct_from_window(df: pd.DataFrame, start_weeks: int, end_weeks: int) -> Tuple[Optional[float], Optional[float]]:
    """
    Non-overlapping window between (end_weeks ago, start_weeks ago].
    Example: (14, 26) = the 12-week period from 26w ago up to 14w ago (excludes most recent 14w).
    Returns (period_high_price, pct_vs_that_high).
    """
    req = {'date','high','close'}
    if df.empty or not req.issubset(df.columns) or end_weeks <= start_weeks:
        return None, None
    latest = df['date'].max()
    start_date = latest - timedelta(weeks=end_weeks)
    end_date   = latest - timedelta(weeks=start_weeks)
    w = df[(df['date'] >= start_date) & (df['date'] < end_date)]
    if w.empty:
        return None, None
    period_high = float(w['high'].max())
    if period_high == 0:
        return None, None
    current_price = float(df['close'].iloc[-1])
    pct = (current_price - period_high) / period_high * 100.0
    return period_high, pct


def print_segmented_highs(df: pd.DataFrame) -> None:
    print("🪟 Segmented highs (non-overlapping windows):")
    segments = [
        ("0–14w (most recent)", 0, 14),
        ("14–26w (prior)", 14, 26),
        ("26–52w (older)", 26, 52),
    ]
    for label, start_w, end_w in segments:
        high, pct = high_and_pct_from_window(df, start_w, end_w)
        if high is None:
            print(f"   • {label}: High=N/A, % vs that high=N/A")
        else:
            print(f"   • {label}: High={high:.2f}, % vs that high: {pct:+.2f}%")

# =============================================================
# New: "from today to last N-week high" helpers
# =============================================================

def _last_nw_high_info(df: pd.DataFrame, weeks: int):
    """
    Find the LAST N-week rolling high (max of 'high' within the last `weeks`).
    Returns (high_price, high_date (date), pct_vs_high_from_today, start_index_in_df).
    pct_vs_high_from_today = (today_close - high_price)/high_price * 100.
    """
    req = {'date','high','close'}
    if df.empty or not req.issubset(df.columns):
        return None, None, None, None
    win = last_n_weeks(df, weeks)
    if win.empty:
        return None, None, None, None
    idx = win['high'].idxmax()
    if pd.isna(idx):
        return None, None, None, None
    high_price = float(df.loc[idx, 'high'])
    high_date  = pd.to_datetime(df.loc[idx, 'date']).date()
    today_close = float(df['close'].iloc[-1])
    pct = (today_close - high_price) / high_price * 100.0
    return high_price, high_date, pct, int(idx)


def _avg_vol_from_high_to_today(df: pd.DataFrame, start_idx: int) -> Optional[float]:
    """
    Average daily volatility (%) measured as std dev of daily close-to-close returns
    from the high's row (inclusive) to today.
    """
    if start_idx is None or df.empty or 'close' not in df.columns:
        return None
    segment = df.iloc[start_idx:]
    if segment.shape[0] < 3:
        return None
    returns = segment['close'].astype(float).pct_change().dropna()
    if returns.empty:
        return None
    return float(returns.std() * 100.0)


def print_today_to_last_52w_high(df: pd.DataFrame) -> None:
    hp, hd, pct, idx = _last_nw_high_info(df, 52)
    print("📏 Today → last 52-week high:")
    if hp is None:
        print("   • Last 52w high: N/A  | Date: N/A  | % from high: N/A")
        return
    print(f"   • Last 52w high: {hp:.2f}  | Date: {hd}  | % from high (today): {pct:+.2f}%")


def print_today_to_last_26w_high(df: pd.DataFrame) -> None:
    hp, hd, pct, idx = _last_nw_high_info(df, 26)
    print("📏 Today → last 26-week high:")
    if hp is None:
        print("   • Last 26w high: N/A  | Date: N/A  | % from high: N/A")
        return
    print(f"   • Last 26w high: {hp:.2f}  | Date: {hd}  | % from high (today): {pct:+.2f}%")


def print_today_to_last_14w_high(df: pd.DataFrame) -> None:
    hp, hd, pct, idx = _last_nw_high_info(df, 14)
    print("📏 Today → last 14-week high:")
    if hp is None:
        print("   • Last 14w high: N/A  | Date: N/A  | % from high: N/A")
        return
    print(f"   • Last 14w high: {hp:.2f}  | Date: {hd}  | % from high (today): {pct:+.2f}%")


def print_vol_stop_from_high_to_today_52w(df: pd.DataFrame, k: float = 2.0) -> None:
    hp, hd, pct, idx = _last_nw_high_info(df, 52)
    print("📊 Volatility from last 52w high → today & trailing stop:")
    if hp is None:
        print("   • Volatility: N/A  | Trailing stop: N/A")
        return
    vol = _avg_vol_from_high_to_today(df, idx)
    if vol is None:
        print(f"   • High date: {hd}  | Avg daily vol: N/A  | Trailing stop: N/A")
        return
    ts = trailing_stop_pct_from_vol(vol, k)
    today_close = float(df['close'].iloc[-1])
    stop_px = today_close * (1 - ts/100.0)
    print(f"   • High date: {hd}  | Avg daily vol: {vol:.2f}%  | Stop ({k:.1f}×): {ts:.2f}% ⇒ Stop px: {stop_px:.2f}")


def print_vol_stop_from_high_to_today_26w(df: pd.DataFrame, k: float = 2.0) -> None:
    hp, hd, pct, idx = _last_nw_high_info(df, 26)
    print("📊 Volatility from last 26w high → today & trailing stop:")
    if hp is None:
        print("   • Volatility: N/A  | Trailing stop: N/A")
        return
    vol = _avg_vol_from_high_to_today(df, idx)
    if vol is None:
        print(f"   • High date: {hd}  | Avg daily vol: N/A  | Trailing stop: N/A")
        return
    ts = trailing_stop_pct_from_vol(vol, k)
    today_close = float(df['close'].iloc[-1])
    stop_px = today_close * (1 - ts/100.0)
    print(f"   • High date: {hd}  | Avg daily vol: {vol:.2f}%  | Stop ({k:.1f}×): {ts:.2f}% ⇒ Stop px: {stop_px:.2f}")

# =============================================================
# Analysis driver
# =============================================================

def analyze_stock(ib: IB, symbol: str, k_multiplier: float = 2.0) -> None:
    print(f"📈 Analyzing {symbol}")

    # Collectors per symbol
    summary = {"symbol": symbol}

    try:
        df = fetch_stock_data(ib, symbol)
    except Exception as e:
        print(f"   ⚠️ Failed to fetch data: {e}")
        return

    if df.empty:
        print("   ⚠️ No data returned from IB.")
        return

    current_price = float(df['close'].iloc[-1])
    print(f"   • Current close: {current_price:.2f}")
    summary.update({"current_close": current_price})

    # --- Highs vs current (rolling windows) with dates/percent ---
    h52, d52, pct52, idx52 = _last_nw_high_info(df, 52)
    h26, d26, pct26, idx26 = _last_nw_high_info(df, 26)
    h14, d14, pct14, idx14 = _last_nw_high_info(df, 14)

    print_today_to_last_52w_high(df)
    print_today_to_last_26w_high(df)
    print_today_to_last_14w_high(df)

    summary.update({
        "last_52w_high": h52, "last_52w_high_date": d52, "pct_from_52w_high": pct52,
        "last_26w_high": h26, "last_26w_high_date": d26, "pct_from_26w_high": pct26,
        "last_14w_high": h14, "last_14w_high_date": d14, "pct_from_14w_high": pct14,
    })

    # --- Non-overlapping segmented windows for clarity ---
    print_segmented_highs(df)
    seg_labels = [("0-14w", 0, 14), ("14-26w", 14, 26), ("26-52w", 26, 52)]
    for label, s, e in seg_labels:
        sh, sp = high_and_pct_from_window(df, s, e)
        SEGMENT_ROWS.append({"symbol": symbol, "segment": label, "segment_high": sh, "pct_vs_segment_high": sp})

    # --- Volatility & trailing stops (returns + ATR%) ---
    vol52_ret = avg_volatility_pct(df, 52)
    ts52_ret  = trailing_stop_pct_from_vol(vol52_ret, k=k_multiplier) if vol52_ret is not None else None
    atrp52    = atr_percent_over_window(df, 52, atr_len=14)
    ts52_atr  = trailing_stop_pct_from_vol(atrp52, k=k_multiplier) if atrp52 is not None else None
    chosen52  = min([x for x in [ts52_ret, ts52_atr] if x is not None]) if any(v is not None for v in [ts52_ret, ts52_atr]) else None
    stop_px_52_ret = current_price * (1 - ts52_ret/100.0) if ts52_ret is not None else None
    stop_px_52_atr = current_price * (1 - ts52_atr/100.0) if ts52_atr is not None else None

    vol26_ret = avg_volatility_pct(df, 26)
    ts26_ret  = trailing_stop_pct_from_vol(vol26_ret, k=k_multiplier) if vol26_ret is not None else None
    atrp26    = atr_percent_over_window(df, 26, atr_len=14)
    ts26_atr  = trailing_stop_pct_from_vol(atrp26, k=k_multiplier) if atrp26 is not None else None
    chosen26  = min([x for x in [ts26_ret, ts26_atr] if x is not None]) if any(v is not None for v in [ts26_ret, ts26_atr]) else None
    stop_px_26_ret = current_price * (1 - ts26_ret/100.0) if ts26_ret is not None else None
    stop_px_26_atr = current_price * (1 - ts26_atr/100.0) if ts26_atr is not None else None

    # Print detailed sections
    print_volatility_and_trailing_stop_52w(df, k=k_multiplier)
    print_volatility_and_trailing_stop_26w(df, k=k_multiplier)

    # --- Vol from last-high -> today views ---
    vol_52_from_high = _avg_vol_from_high_to_today(df, idx52) if idx52 is not None else None
    ts_52_from_high  = trailing_stop_pct_from_vol(vol_52_from_high, k=k_multiplier) if vol_52_from_high is not None else None
    vol_26_from_high = _avg_vol_from_high_to_today(df, idx26) if idx26 is not None else None
    ts_26_from_high  = trailing_stop_pct_from_vol(vol_26_from_high, k=k_multiplier) if vol_26_from_high is not None else None

    print_vol_stop_from_high_to_today_52w(df, k=k_multiplier)
    print_vol_stop_from_high_to_today_26w(df, k=k_multiplier)

    # Capture stops into sheet rows
    STOPS_ROWS.append({
        "symbol": symbol,
        "vol52_ret%": vol52_ret, "ts52_ret%": ts52_ret, "atrp52%": atrp52, "ts52_atr%": ts52_atr, "chosen_ts52%": chosen52,
        "stop_px_52_ret": stop_px_52_ret, "stop_px_52_atr": stop_px_52_atr,
        "vol26_ret%": vol26_ret, "ts26_ret%": ts26_ret, "atrp26%": atrp26, "ts26_atr%": ts26_atr, "chosen_ts26%": chosen26,
        "stop_px_26_ret": stop_px_26_ret, "stop_px_26_atr": stop_px_26_atr,
        "vol_from_last_52w_high%": vol_52_from_high, "ts_from_last_52w_high%": ts_52_from_high,
        "vol_from_last_26w_high%": vol_26_from_high, "ts_from_last_26w_high%": ts_26_from_high,
    })

    # --- Optional: last-52w non-overlapping drawdowns ---
    dd_5  = count_falls_5pct(df)
    dd_10 = count_falls_10pct(df)
    dd_15 = count_falls_15pct(df)
    dd_20 = count_falls_20pct(df)
    dd_25 = count_falls_25pct(df)
    dd_35 = count_falls_35pct(df)
    dd_40 = count_falls_40pct(df)

    print("   • Non-overlapping drawdowns in last 52w:")
    print(f"       ≥5%:  {dd_5}  |  ≥10%: {dd_10}  |  ≥15%: {dd_15}  |  ≥20%: {dd_20}  |  ≥25%: {dd_25}  |  ≥35%: {dd_35}  |  ≥40%: {dd_40}")

    DRAWDOWN_ROWS.append({
        "symbol": symbol, "ge_5": dd_5, "ge_10": dd_10, "ge_15": dd_15, "ge_20": dd_20, "ge_25": dd_25, "ge_35": dd_35, "ge_40": dd_40
    })

    # Store final summary row
    SUMMARY_ROWS.append(summary)

# =============================================================
# Excel writer
# =============================================================

def write_to_excel(path: str) -> None:
    # Build DataFrames once for sizing/formatting
    df_sum = pd.DataFrame(SUMMARY_ROWS) if SUMMARY_ROWS else None
    df_stops = pd.DataFrame(STOPS_ROWS) if STOPS_ROWS else None
    df_seg = pd.DataFrame(SEGMENT_ROWS) if SEGMENT_ROWS else None
    df_dd = pd.DataFrame(DRAWDOWN_ROWS) if DRAWDOWN_ROWS else None

    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        # ---- Write sheets (raw)
        if df_sum is not None:
            df_sum.to_excel(writer, sheet_name="Summary", index=False)
        if df_stops is not None:
            df_stops.to_excel(writer, sheet_name="Stops", index=False)
        if df_seg is not None:
            df_seg.to_excel(writer, sheet_name="Segments", index=False)
        if df_dd is not None:
            df_dd.to_excel(writer, sheet_name="Drawdowns", index=False)

        # ---- Formatting helpers
        wb = writer.book
        header_fmt = wb.add_format({"bold": True, "bg_color": "#F2F2F2", "border": 1})
        money = wb.add_format({"num_format": "#,##0.00", "align": "center"})
        num   = wb.add_format({"num_format": "0.00", "align": "center"})
        intc  = wb.add_format({"num_format": "0", "align": "center"})
        pct_no100 = wb.add_format({"num_format": "0.00", "align": "center"})  # for percentage *points*

        def autosize(ws, df):
            # Header row with formatting
            for c, col in enumerate(df.columns):
                ws.write(0, c, col, header_fmt)
            # Autosize columns based on sample of rows
            for c, col in enumerate(df.columns):
                series = df[col].astype(str).head(1000)
                max_len = max(len(str(col)), *(len(x) for x in series))
                ws.set_column(c, c, min(max(10, max_len + 2), 42))
            ws.freeze_panes(1, 1)

        def apply_three_color(ws, first_row, last_row, col_idx):
            ws.conditional_format(first_row, col_idx, last_row, col_idx, {
                "type": "3_color_scale",
                "min_color": "#63BE7B",  # green (better)
                "mid_color": "#FFEB84",  # yellow
                "max_color": "#F8696B",  # red (worse)
            })

        # ---- Summary sheet formatting
        if df_sum is not None:
            ws = writer.sheets["Summary"]
            autosize(ws, df_sum)
            money_cols = {"current_close", "last_52w_high", "last_26w_high", "last_14w_high"}
            pct_cols   = {"pct_from_52w_high", "pct_from_26w_high", "pct_from_14w_high"}
            for c, name in enumerate(df_sum.columns):
                if name in money_cols:
                    ws.set_column(c, c, None, money)
                elif name in pct_cols:
                    ws.set_column(c, c, None, pct_no100)
            # Heatmap across the % columns
            r0, r1 = 1, (len(df_sum))
            for name in pct_cols:
                if name in df_sum.columns:
                    ci = df_sum.columns.get_loc(name)
                    apply_three_color(ws, r0, r1, ci)

        # ---- Stops sheet formatting
        if df_stops is not None:
            ws = writer.sheets["Stops"]
            autosize(ws, df_stops)
            money_cols = [c for c in df_stops.columns if c.startswith("stop_px_")]
            pct_cols   = [c for c in df_stops.columns if c.endswith("%")]
            for c, name in enumerate(df_stops.columns):
                if name in money_cols:
                    ws.set_column(c, c, None, money)
                elif name in pct_cols:
                    ws.set_column(c, c, None, pct_no100)

        # ---- Segments sheet formatting
        if df_seg is not None:
            ws = writer.sheets["Segments"]
            autosize(ws, df_seg)
            for c, name in enumerate(df_seg.columns):
                if name == "segment_high":
                    ws.set_column(c, c, None, money)
                elif name == "pct_vs_segment_high":
                    ws.set_column(c, c, None, pct_no100)

        # ---- Drawdowns sheet formatting
        if df_dd is not None:
            ws = writer.sheets["Drawdowns"]
            autosize(ws, df_dd)
            for c in range(len(df_dd.columns)):
                ws.set_column(c, c, None, intc)

# =============================================================
# Main
# =============================================================

def main() -> None:
    ib = connect_ib()
    try:
        for sym in symbols:
            analyze_stock(ib, sym, k_multiplier=2.0)  # tune k as needed (1.5–3.0 typical)
    finally:
        ib.disconnect()

    # After processing all symbols, write Excel
    fname = f"stock_analysis_{datetime.now().strftime('%Y-%m-%d_%H%M')}.xlsx"
    write_to_excel(fname)
    print(f"💾 Excel written: {fname}")


if __name__ == "__main__":
    main()
