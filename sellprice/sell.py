from ib_connection import connect_ib, disconnect_ib
from data_fetcher import fetch_historical_data
from indicators import calculate_indicators, compute_market_stage
from patterns import detect_trendline_break, detect_chart_pattern
from analyze_trades import run_post_analysis
from indicators import calculate_indicators, compute_market_stage

from utils import log_signal
from signals import (
    check_weekly_trend, check_sell_signals, check_exit_signals,
    check_entry_signals, apply_trailing_stop
)

from config import symbols


def process_symbol(symbol):
    print(f"\n=== Processing Symbol: {symbol} ===")
    weekly_trend = check_weekly_trend(symbol)
    df = fetch_historical_data(symbol)
    df = calculate_indicators(df)
    # ✅ compute stage before you try to read it
    df = compute_market_stage(df)

    # (order of the next two doesn’t matter)
    df = detect_trendline_break(df)
    df = detect_chart_pattern(df)

    # log the stage (with a light guard so it never crashes)
    stage = str(df['market_stage'].iloc[-1]) if 'market_stage' in df.columns else "Neutral/Transition"
    stage_conf = float(df['stage_conf'].iloc[-1]) if 'stage_conf' in df.columns else 0.0
    date_str = df.index[-1].strftime("%Y-%m-%d")
    log_signal(symbol, date_str, f"Stage: {stage}", stage_conf, signal_type="WATCH", condition="STAGE")

    df = detect_trendline_break(df)
    df = detect_chart_pattern(df)
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


if __name__ == "__main__":
    # Removed IB connection to avoid accidental order placements
    connect_ib()
    for sym in set(symbols):
        process_symbol(sym)
    disconnect_ib()
    print(" Running post-analysis summary...")
    run_post_analysis()
