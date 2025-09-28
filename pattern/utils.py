import pandas as pd
import numpy as np
from typing import Iterable, Optional, Tuple
from fetching import fetch_data_cached

# === Data access (delegates to fetching cache/IB) ===

def fetch_ib_data(
    symbol: str,
    duration: str = "3 Y",
    bar_size: str = "1 day",
    refresh: bool = False,
) -> pd.DataFrame:
    return fetch_data_cached(symbol, duration=duration, bar_size=bar_size, refresh=refresh)


# === Quarterly Supply/Demand Zones ===

def _resample_to_quarterly(df: pd.DataFrame, freq: str = "QE-DEC") -> pd.DataFrame:
    """
    Resample daily OHLCV to quarterly OHLCV.
    `freq` is the quarter-end anchor (e.g., "QE-DEC" for calendar quarters,
    or a fiscal anchor like "QE-MAR").
    Requires columns: ['date','open','high','low','close'] (volume optional).
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").set_index("date")
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in d.columns:
        agg["volume"] = "sum"
    q = d.resample(freq).agg(agg).dropna(subset=["open","high","low","close"])
    q.reset_index(inplace=True)
    return q


def _pivot_flags(df_q: pd.DataFrame, left: int = 1, right: int = 1) -> pd.DataFrame:
    """
    Mark simple 3-bar fractal pivots on quarterly data.
    pivot_high: high greater than previous 'left' highs and greater or equal next 'right' highs.
    pivot_low:  low lower than previous 'left' lows and lower or equal next 'right' lows.
    """
    q = df_q.copy().reset_index(drop=True)
    highs = q["high"].values
    lows = q["low"].values
    n = len(q)
    ph = np.zeros(n, dtype=bool)
    pl = np.zeros(n, dtype=bool)

    for i in range(n):
        l0 = max(0, i - left)
        r0 = min(n, i + right + 1)
        if i - left < 0 or i + right >= n:
            continue
        if highs[i] > np.max(highs[l0:i]) and highs[i] >= np.max(highs[i+1:r0]):
            ph[i] = True
        if lows[i] < np.min(lows[l0:i]) and lows[i] <= np.min(lows[i+1:r0]):
            pl[i] = True

    q["pivot_high"] = ph
    q["pivot_low"] = pl
    return q


def compute_quarterly_zones(df_daily: pd.DataFrame, *, freq: str = "QE-DEC", near_pct: float = 0.03) -> dict:
    """
    Compute nearest Quarterly Demand and Supply zones (proximal/distal) relative to last close.

    Demand zone (support): use base quarter at a pivot low:
        proximal = max(open, close) of that quarter
        distal   = low  of that quarter

    Supply zone (resistance): use base quarter at a pivot high:
        proximal = min(open, close) of that quarter
        distal   = high of that quarter

    Extras returned for each zone (if found):
        - width: |proximal - distal|
        - delta: last_close - proximal  (signed distance)
        - distance: abs(delta)
        - distance_pct: abs(delta) / proximal * 100
        - near: abs(delta) / proximal <= near_pct

    Set `freq` to change quarter anchor (e.g., "QE-MAR").
    Set `near_pct` to change the near-zone threshold (default 3%).
    """
    if df_daily is None or df_daily.empty:
        return {"demand": None, "supply": None}

    required = {"date","open","high","low","close"}
    missing = required - set(df_daily.columns)
    if missing:
        raise ValueError(f"Quarterly zone calc requires columns {sorted(required)}; missing {sorted(missing)}")

    last_close = float(df_daily["close"].iloc[-1])
    q = _resample_to_quarterly(df_daily, freq=freq)
    q = _pivot_flags(q, left=1, right=1)

    def enrich(row, proximal, distal):
        delta = last_close - proximal
        distance = abs(delta)
        width = abs(proximal - distal)
        near = (distance / proximal) <= near_pct if proximal else False
        return {
            "date": pd.to_datetime(row["date"]).date().isoformat(),
            "proximal": float(proximal),
            "distal": float(distal),
            "width": float(width),
            "delta": float(delta),
            "distance": float(distance),
            "distance_pct": float(distance / proximal * 100) if proximal else None,
            "near": bool(near),
        }

    # Nearest demand (pivot_low with proximal below current price)
    demand_candidates = q[q["pivot_low"]].copy()
    demand_candidates["proximal"] = demand_candidates[["open","close"]].max(axis=1)
    demand_candidates["distal"] = demand_candidates["low"]
    demand_candidates = demand_candidates[demand_candidates["proximal"] < last_close]
    demand = None
    if not demand_candidates.empty:
        demand_candidates["distance"] = last_close - demand_candidates["proximal"]
        row = demand_candidates.sort_values("distance").iloc[0]
        demand = enrich(row, row["proximal"], row["distal"])

    # Nearest supply (pivot_high with proximal above current price)
    supply_candidates = q[q["pivot_high"]].copy()
    supply_candidates["proximal"] = supply_candidates[["open","close"]].min(axis=1)
    supply_candidates["distal"] = supply_candidates["high"]
    supply_candidates = supply_candidates[supply_candidates["proximal"] > last_close]
    supply = None
    if not supply_candidates.empty:
        supply_candidates["distance"] = supply_candidates["proximal"] - last_close
        row = supply_candidates.sort_values("distance").iloc[0]
        supply = enrich(row, row["proximal"], row["distal"])

    return {"demand": demand, "supply": supply}


# === Indicators & Signals ===

def is_stock_undervalued(df: pd.DataFrame, threshold: float = 0.9) -> bool:
    current_price = float(df["close"].iloc[-1])
    long_term_avg = float(df["close"].mean())
    return current_price < threshold * long_term_avg


def add_macd_signal(df: pd.DataFrame) -> pd.DataFrame:
    exp12 = df["close"].ewm(span=12, adjust=False).mean()
    exp26 = df["close"].ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["MACD"] = macd
    df["MACD_Signal"] = signal
    df["MACD_Bullish"] = (df["MACD"] > df["MACD_Signal"]) & (df["MACD"].diff() > 0)
    return df


def add_rsi_signal(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI_Oversold_Reversal"] = (df["RSI"] < 30) & (df["close"].diff() > 0)
    return df


def add_volume_surge(df: pd.DataFrame, window: int = 20, multiplier: float = 2.0) -> pd.DataFrame:
    df["Avg_Volume"] = df["volume"].rolling(window).mean()
    df["Volume_Surge"] = df["volume"] > (multiplier * df["Avg_Volume"])
    return df


def summarize_signals(df: pd.DataFrame) -> dict:
    macd_signal = bool(df["MACD_Bullish"].iloc[-1])
    rsi_signal = bool(df["RSI_Oversold_Reversal"].iloc[-1])
    volume_surge = bool(df["Volume_Surge"].iloc[-1])
    return {
        "MACD_Bullish": macd_signal,
        "RSI_Oversold_Reversal": rsi_signal,
        "Volume_Surge": volume_surge,
    }


def detect_ema_crossover(df: pd.DataFrame) -> pd.DataFrame:
    df["EMA_9"] = df["close"].ewm(span=9).mean()
    df["EMA_21"] = df["close"].ewm(span=21).mean()
    crossover = (df["EMA_9"] > df["EMA_21"]) & (df["EMA_9"].shift(1) <= df["EMA_21"].shift(1))
    return df[crossover]


def detect_bollinger_squeeze(df: pd.DataFrame, window: int = 20) -> bool:
    df["MA"] = df["close"].rolling(window).mean()
    df["STD"] = df["close"].rolling(window).std()
    df["BB_Width"] = (2 * df["STD"]) / df["MA"]
    last_width = df["BB_Width"].iloc[-1]
    threshold = df["BB_Width"].rolling(window).quantile(0.2).iloc[-1]
    return bool(last_width < threshold)


def stock_is_volatile(df: pd.DataFrame, threshold: float = 0.05) -> bool:
    return bool(df["close"].pct_change().rolling(20).std().iloc[-1] > threshold)


def avg_dollar_volume(df: pd.DataFrame, window: int = 20) -> float:
    """Average dollar volume (close * volume) over rolling window."""
    dv = (df["close"] * df["volume"]).rolling(window).mean()
    return float(dv.iloc[-1])


def relative_strength_vs_proxy(
    df_target: pd.DataFrame,
    df_proxy: pd.DataFrame,
    windows: Tuple[int, int] = (20, 60),
) -> bool:
    """True if target outperforms proxy over all given windows."""
    for w in windows:
        if len(df_target) <= w or len(df_proxy) <= w:
            return False
        rt = df_target["close"].iloc[-1] / df_target["close"].iloc[-1 - w] - 1.0
        rp = df_proxy["close"].iloc[-1] / df_proxy["close"].iloc[-1 - w] - 1.0
        if rt <= rp:
            return False
    return True


def analyze_drop_rebound_patterns(
    df: pd.DataFrame,
    drop_thresholds: Iterable[int] = (5, 10),
    rebound_threshold: float = 5,
    rebound_window: int = 10,
    volatility_threshold: float = 0.05,
    cooldown: int = 0,
) -> pd.DataFrame:
    """Compute how often a drop of X% from a recent peak rebounds by Y% within N days.

    Fixes: guard for NaNs early in the series and uses an inclusive rebound window slice [min_idx : min_idx+window].
    Adds a non-overlap `cooldown` (days to skip after a qualifying drop event).
    """
    df = df[["date", "close"]].copy()
    df["return"] = df["close"].pct_change() * 100

    peak_window = 60 if stock_is_volatile(df, volatility_threshold) else 30
    df["rolling_peak"] = df["close"].rolling(window=peak_window, min_periods=peak_window).max()

    results = []
    n = len(df)

    for drop_pct in drop_thresholds:
        drop_rebound_count = 0
        total_drop_events = 0
        i = 0
        last_index_allowed = n - rebound_window - 1
        while i < last_index_allowed:
            price_now = df["rolling_peak"].iloc[i]
            if pd.isna(price_now):
                i += 1
                continue

            window_slice = df["close"].iloc[i + 1 : i + 1 + rebound_window]
            if window_slice.empty:
                i += 1
                continue

            min_idx = window_slice.idxmin()
            low_price = df["close"].loc[min_idx]
            drop_percent = ((price_now - low_price) / price_now) * 100

            if drop_percent >= drop_pct:
                total_drop_events += 1
                rebound_prices = df["close"].iloc[min_idx : min_idx + rebound_window]
                if not rebound_prices.empty:
                    rebound = ((rebound_prices.max() - low_price) / low_price) * 100
                    if rebound >= rebound_threshold:
                        drop_rebound_count += 1
                # apply non-overlap cooldown
                i = int(min_idx) + cooldown
            else:
                i += 1

        success_rate = (drop_rebound_count / total_drop_events) * 100 if total_drop_events > 0 else 0.0
        results.append(
            {
                "drop_threshold": int(drop_pct),
                "rebound_threshold": float(rebound_threshold),
                "window_days": int(rebound_window),
                "total_events": int(total_drop_events),
                "successful_rebounds": int(drop_rebound_count),
                "success_rate_%": round(success_rate, 2),
            }
        )

    return pd.DataFrame(results)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # --- EMAs ---
    df["EMA_20"] = df["close"].ewm(span=20).mean()
    df["EMA_50"] = df["close"].ewm(span=50).mean()
    df["EMA_200"] = df["close"].ewm(span=200).mean()

    # --- MACD & signal ---
    df["EMA12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # --- MACD bullish crossover flag ---
    df["MACD_Crossover"] = (df["MACD"].shift(1) < df["Signal"].shift(1)) & (df["MACD"] > df["Signal"])
    return df


def export_results_to_csv(results, filename: str = "results.csv") -> None:
    df = pd.DataFrame(results)
    if not df.empty and "symbol" in df.columns:
        cols = ["symbol"] + [c for c in df.columns if c != "symbol"]
        df = df[cols]
    df.to_csv(filename, index=False)