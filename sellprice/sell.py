from ib_connection import connect_ib, disconnect_ib
from data_fetcher import fetch_historical_data
from indicators import calculate_indicators
from patterns import detect_trendline_break, detect_chart_pattern
from analyze_trades import run_post_analysis
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
    df = detect_trendline_break(df)
    df = detect_chart_pattern(df)
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
