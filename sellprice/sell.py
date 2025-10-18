from ib_connection import connect_ib, disconnect_ib
from data_fetcher import fetch_historical_data
from patterns import detect_trendline_break, detect_chart_pattern
from analyze_trades import run_post_analysis
from indicators import calculate_indicators, compute_market_stage
from utils import export_signals_to_excel



import pandas as pd
import numpy as np

from utils import log_signal
from signals import (
    check_weekly_trend, check_sell_signals, check_exit_signals,
    check_entry_signals, apply_trailing_stop
)

from selling_price import (
    get_selling_price_6w, get_selling_price_8w,
    get_selling_price_12w, get_selling_price_18w,
    get_selling_price_30w,
)


from config import symbols


def process_symbol(symbol):
    print(f"\n=== Processing Symbol: {symbol} ===")
    weekly_trend = check_weekly_trend(symbol)
    df = fetch_historical_data(symbol)

    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        else:
            # last resort: try to parse the existing index
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                raise ValueError("Expected a DatetimeIndex or a 'date' column from data_fetcher()")

    df = df.sort_index()

    df = calculate_indicators(df)
    # ✅ compute stage before you try to read it
    df = compute_market_stage(df)

    # (order of the next two doesn’t matter)
    df = detect_trendline_break(df)
    df = detect_chart_pattern(df)

    # Weekly SELL targets from recent candle structure
    sp6 = get_selling_price_6w(df)
    sp8 = get_selling_price_8w(df)
    sp12 = get_selling_price_12w(df)
    sp18 = get_selling_price_18w(df)
    sp30 = get_selling_price_30w(df)

    last_close = float(df['close'].iloc[-1])
    atr_value = float(df['atr'].iloc[-1]) if 'atr' in df.columns else np.nan
    atr_pct = round(100.0 * (atr_value / last_close), 2) if np.isfinite(atr_value) else None

    print(
        f"[{symbol}] Weekly SELL targets → "
        f"6w:{_r(sp6)}  8w:{_r(sp8)}  12w:{_r(sp12)}  18w:{_r(sp18)}  30w:{_r(sp30)}"
    )

    # log the stage (with a light guard so it never crashes)
    stage = str(df['market_stage'].iloc[-1]) if 'market_stage' in df.columns else "Neutral/Transition"
    stage_conf = float(df['stage_conf'].iloc[-1]) if 'stage_conf' in df.columns else 0.0
    date_str = df.index[-1].strftime("%Y-%m-%d")
    log_signal(symbol, date_str, f"Stage: {stage}", stage_conf, signal_type="WATCH", condition="STAGE")

    # Persist targets for auditability
    last_close = float(df['close'].iloc[-1])

    def pct_to_close(tgt):
        if tgt is None or last_close <= 0:
            return None
        return round(100.0 * (float(tgt) / last_close - 1.0), 2)

    p6 = pct_to_close(sp6)
    p8 = pct_to_close(sp8)
    p12 = pct_to_close(sp12)
    p18 = pct_to_close(sp18)
    p30 = pct_to_close(sp30)

    # Log SELL targets with ATRpct included
    log_signal(symbol, date_str, f"SELL_TGT_6w:{sp6}", p6 if p6 is not None else 0.0, signal_type="TARGET",
               condition="SELL_TGT_6w", ATRpct=atr_pct)
    log_signal(symbol, date_str, f"SELL_TGT_8w:{sp8}", p8 if p8 is not None else 0.0, signal_type="TARGET",
               condition="SELL_TGT_8w", ATRpct=atr_pct)
    log_signal(symbol, date_str, f"SELL_TGT_12w:{sp12}", p12 if p12 is not None else 0.0, signal_type="TARGET",
               condition="SELL_TGT_12w", ATRpct=atr_pct)
    log_signal(symbol, date_str, f"SELL_TGT_18w:{sp18}", p18 if p18 is not None else 0.0, signal_type="TARGET",
               condition="SELL_TGT_18w", ATRpct=atr_pct)
    log_signal(symbol, date_str, f"SELL_TGT_30w:{sp30}", p30 if p30 is not None else 0.0, signal_type="TARGET",
               condition="SELL_TGT_30w", ATRpct=atr_pct)

    # df = detect_trendline_break(df)
    # df = detect_chart_pattern(df)
    # Log the stage so it shows up in summaries
    stage = str(df['market_stage'].iloc[-1])
    stage_conf = float(df['stage_conf'].iloc[-1])
    date_str = df.index[-1].strftime("%Y-%m-%d")
    log_signal(symbol, date_str, f"Stage: {stage}", stage_conf, signal_type="WATCH", condition="STAGE")



    print(f"\n--- {symbol} Signals ---")
    signals = check_sell_signals(df)
    if not weekly_trend:
        print("[WARNING] Weekly trend is bearish. Use caution with long positions.")
    for s in signals:
        print(f"[SIGNAL] {s}")
    print(f"Trailing Stop: {apply_trailing_stop(df):.2f}")

    check_exit_signals(df)
    check_entry_signals(df)


def _r(x):
    return None if x is None else round(float(x), 2)

if __name__ == "__main__":
    # Removed IB connection to avoid accidental order placements
    connect_ib()
    for sym in set(symbols):
        process_symbol(sym)
    disconnect_ib()
    print(" Running post-analysis summary...")
    run_post_analysis()
    export_signals_to_excel("signals_raw.xlsx")