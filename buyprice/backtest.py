# backtest.py

import numpy as np
import pandas as pd


def evaluate_backtest_accuracy(symbol, df, buy_price, gain_thresh=0.04, use_close=False):
    """
    Simple 90-day forward backtest:

    - Looks at last 90 bars in df
    - Computes (price - buy_price) / buy_price
    - hit = any day with gain_pct > gain_thresh
    - max_gain = max gain_pct * 100
    - days_to_peak = index (0..89) where max_gain occurs

    Returns (hit: bool, max_gain_pct: float, days_to_peak: int).
    On any error or unusable data → (False, 0.0, -1).
    """
    try:
        # Basic sanity checks
        if df is None or df.empty:
            raise ValueError("empty df")
        if buy_price is None or not np.isfinite(float(buy_price)):
            raise ValueError(f"bad buy_price={buy_price}")

        df_tail = df.tail(90).copy().reset_index(drop=True)
        if df_tail.empty:
            raise ValueError("no rows in tail(90)")

        price_col = "close" if use_close else "high"
        if price_col not in df_tail.columns:
            raise KeyError(f"missing column {price_col}")

        # Coerce to numeric, drop NaNs
        ref = pd.to_numeric(df_tail[price_col], errors="coerce")
        df_tail["gain_pct"] = (ref - float(buy_price)) / float(buy_price)
        df_tail = df_tail.dropna(subset=["gain_pct"])
        if df_tail.empty:
            raise ValueError("gain_pct all NaN after cleaning")

        gain_series = pd.to_numeric(df_tail["gain_pct"], errors="coerce")
        gain_series = gain_series.replace([np.inf, -np.inf], np.nan).dropna()
        if gain_series.empty:
            raise ValueError("gain_series all NaN/inf")

        hit = bool((gain_series > float(gain_thresh)).any())

        max_gain_raw = gain_series.max()
        if not np.isfinite(max_gain_raw):
            max_gain = 0.0
            days_to_peak = -1
        else:
            max_gain = float(max_gain_raw * 100.0)
            days_to_peak = int(gain_series.idxmax())

        return hit, max_gain, days_to_peak

    except Exception as e:
        print(f"[{symbol}] Backtest error: {e}")
        return False, 0.0, -1
