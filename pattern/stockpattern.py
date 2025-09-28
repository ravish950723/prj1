import argparse
import os
import pandas as pd
import numpy as np
from datetime import timedelta

from utils import (
    fetch_ib_data,
    add_macd_signal,
    add_rsi_signal,
    add_volume_surge,
    detect_ema_crossover,
    detect_bollinger_squeeze,
    analyze_drop_rebound_patterns,
    is_stock_undervalued,
    summarize_signals,
    export_results_to_csv,
    relative_strength_vs_proxy,
    avg_dollar_volume,
    compute_quarterly_zones,
)
from config import symbol_to_sector, symbols, sector_etfs


results_list = []


def get_label(score: int, passes_liquidity: bool) -> str:
    if score >= 7 and passes_liquidity:
        return "📈 STRONG BUY"
    elif score >= 7 and not passes_liquidity:
        return "🔎 WATCH (liquidity)"
    elif score >= 5:
        return "🔎 WATCH"
    else:
        return "⚠️ WEAK"


def analyze_stock(
    symbol: str,
    proxy_df: pd.DataFrame,
    refresh: bool,
    min_events: int,
    cooldown: int,
    min_dollar_vol: float,
) -> None:
    print(f"📊 Analyzing: {symbol}")
    try:
        df = fetch_ib_data(symbol, refresh=refresh)
        print(f"📥 Fetched {len(df)} rows of data for {symbol}")

        df = add_macd_signal(df)
        print("🧮 MACD Signal added")
        df = add_rsi_signal(df)
        print("🧮 RSI Signal added")
        df = add_volume_surge(df)
        print("🧮 Volume Surge calculated")

        trend_df_5_10 = analyze_drop_rebound_patterns(df, drop_thresholds=[5, 10], cooldown=cooldown)
        trend_df_15_20 = analyze_drop_rebound_patterns(df, drop_thresholds=[15, 20], cooldown=cooldown)
        trend_df_25_30 = analyze_drop_rebound_patterns(df, drop_thresholds=[25, 30], cooldown=cooldown)

        print("📉 Drop-Rebound Analysis (5%-10%):")
        print(trend_df_5_10)
        print("📉 Drop-Rebound Analysis (15%-20%):")
        print(trend_df_15_20)
        print("📉 Drop-Rebound Analysis (25%-30%):")
        print(trend_df_25_30)

        undervalued = is_stock_undervalued(df)
        ema_cross_df = detect_ema_crossover(df)
        ema_bullish = not ema_cross_df.empty
        bb_squeeze = detect_bollinger_squeeze(df)

        rs_vs_proxy = relative_strength_vs_proxy(df, proxy_df) if not df.empty else False
        adv = avg_dollar_volume(df)
        passes_liquidity = adv >= float(min_dollar_vol)

        print(f"🔍 Undervalued (Price < 90% of 3Y avg): {'✅ YES' if undervalued else '❌ NO'}")
        print(f"🔁 EMA Bullish Crossover Signal Detected: {'✅ YES' if ema_bullish else '❌ NO'}")
        print(f"📉 Bollinger Band Squeeze Detected: {'✅ YES' if bb_squeeze else '❌ NO'}")
        print(f"💪 RS vs proxy (20/60d): {'✅ YES' if rs_vs_proxy else '❌ NO'}  | 💵 Avg $Vol(20): {adv:,.0f} {'✅' if passes_liquidity else '❌'} (min {min_dollar_vol:,.0f})")

        signal_summary = summarize_signals(df)

        # Pull success rates and event counts
        sr5 = float(trend_df_5_10.iloc[0]["success_rate_%"]) if not trend_df_5_10.empty else 0.0
        sr10 = float(trend_df_5_10.iloc[1]["success_rate_%"]) if len(trend_df_5_10) > 1 else 0.0
        ev10 = int(trend_df_5_10.iloc[1]["total_events"]) if len(trend_df_5_10) > 1 else 0

        sr15 = float(trend_df_15_20.iloc[0]["success_rate_%"]) if not trend_df_15_20.empty else 0.0
        sr20 = float(trend_df_15_20.iloc[1]["success_rate_%"]) if len(trend_df_15_20) > 1 else 0.0
        ev20 = int(trend_df_15_20.iloc[1]["total_events"]) if len(trend_df_15_20) > 1 else 0

        # Index-safe lookups for 25% and 30%
        sr25 = float(trend_df_25_30.loc[trend_df_25_30["drop_threshold"] == 25, "success_rate_%"].iloc[0]) if not trend_df_25_30.empty and (trend_df_25_30["drop_threshold"] == 25).any() else 0.0
        sr30 = float(trend_df_25_30.loc[trend_df_25_30["drop_threshold"] == 30, "success_rate_%"].iloc[0]) if not trend_df_25_30.empty and (trend_df_25_30["drop_threshold"] == 30).any() else 0.0
        ev25 = int(trend_df_25_30.loc[trend_df_25_30["drop_threshold"] == 25, "total_events"].iloc[0]) if not trend_df_25_30.empty and (trend_df_25_30["drop_threshold"] == 25).any() else 0
        ev30 = int(trend_df_25_30.loc[trend_df_25_30["drop_threshold"] == 30, "total_events"].iloc[0]) if not trend_df_25_30.empty and (trend_df_25_30["drop_threshold"] == 30).any() else 0

        signal_summary.update(
            {
                "symbol": symbol,
                "sector": symbol_to_sector.get(symbol, "Unknown"),
                "undervalued": bool(undervalued),
                "ema_bullish": bool(ema_bullish),
                "bb_squeeze": bool(bb_squeeze),
                "rs_vs_proxy": bool(rs_vs_proxy),
                "avg_dollar_vol": float(adv),
                "passes_liquidity": bool(passes_liquidity),
                "drop_rebound_5%_success_rate": sr5,
                "drop_rebound_10%_success_rate": sr10,
                "events_10": int(ev10),
                "drop_rebound_15%_success_rate": sr15,
                "drop_rebound_20%_success_rate": sr20,
                "events_20": int(ev20),
                "drop_rebound_25%_success_rate": sr25,
                "drop_rebound_30%_success_rate": sr30,
                "events_25": int(ev25),
                "events_30": int(ev30),
            }
        )

        try:
            q_freq = os.getenv("FISCAL_Q_FREQ", "QE-DEC")  # e.g., "QE-MAR" for Mar/Jun/Sep/Dec fiscal year
            near_pct = float(os.getenv("ZONE_NEAR_PCT", "0.03"))  # 3% default

            zones = compute_quarterly_zones(df, freq=q_freq, near_pct=near_pct)
            dz = zones.get("demand") or {}
            sz = zones.get("supply") or {}
            signal_summary.update({
                # demand
                "q_demand_date": dz.get("date"),
                "q_demand_proximal": dz.get("proximal"),
                "q_demand_distal": dz.get("distal"),
                "q_demand_width": dz.get("width"),
                "q_demand_delta": dz.get("delta"),
                "q_demand_distance": dz.get("distance"),
                "q_demand_distance_pct": dz.get("distance_pct"),
                "q_demand_near": dz.get("near"),
                # supply
                "q_supply_date": sz.get("date"),
                "q_supply_proximal": sz.get("proximal"),
                "q_supply_distal": sz.get("distal"),
                "q_supply_width": sz.get("width"),
                "q_supply_delta": sz.get("delta"),
                "q_supply_distance": sz.get("distance"),
                "q_supply_distance_pct": sz.get("distance_pct"),
                "q_supply_near": sz.get("near"),
            })
        except Exception as _e:
            # Keep analysis running even if zone calc fails
            signal_summary.update({
                "q_demand_date": None,
                "q_demand_proximal": None,
                "q_demand_distal": None,
                "q_demand_width": None,
                "q_demand_delta": None,
                "q_demand_distance": None,
                "q_demand_distance_pct": None,
                "q_demand_near": None,
                "q_supply_date": None,
                "q_supply_proximal": None,
                "q_supply_distal": None,
                "q_supply_width": None,
                "q_supply_delta": None,
                "q_supply_distance": None,
                "q_supply_distance_pct": None,
                "q_supply_near": None,
            })

        # Optional console preview
        if dz or sz:
            print(
                f"🧭 Quarterly Zones — Demand: {dz or 'None'} | Supply: {sz or 'None'} | "
                f"Near flags ⇒ D:{dz.get('near') if dz else None} S:{sz.get('near') if sz else None}"
            )

        # Scoring (conservative):
        score = sum(
            [
                signal_summary["MACD_Bullish"],
                signal_summary["RSI_Oversold_Reversal"],
                signal_summary["Volume_Surge"],
                signal_summary["undervalued"],
                signal_summary["ema_bullish"],
                signal_summary["bb_squeeze"],
                signal_summary["rs_vs_proxy"],
            ]
        )
        # Award success-rate points only if enough events
        if ev10 >= min_events and sr10 > 70:
            score += 1
        if ev20 >= min_events and sr20 > 70:
            score += 1

        signal_summary["score"] = int(score)
        signal_summary["label"] = get_label(signal_summary["score"], passes_liquidity)

        print("📈 Signal Summary:")
        for k, v in signal_summary.items():
            print(f"   {k}: {v}")

        results_list.append(signal_summary)
        export_results_to_csv(results_list, "results.csv")
        print("✅ Results exported to results.csv")

    except Exception as e:
        print(f"❌ Error analyzing {symbol}: {e}")


def analyze_all(tickers, refresh: bool, min_events: int, cooldown: int, min_dollar_vol: float):
    # Use sector proxy ETF once (URA as nuclear proxy)
    proxy_df = fetch_ib_data("URA", refresh=refresh)
    for sym in tickers:
        analyze_stock(sym, proxy_df, refresh, min_events, cooldown, min_dollar_vol)
    print("🏁 Analysis complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nuclear scanner")
    parser.add_argument("--refresh", action="store_true", help="Bypass on-disk cache and refetch from IB")
    parser.add_argument("--min-events", type=int, default=50, help="Minimum drop events to trust SR points")
    parser.add_argument("--cooldown", type=int, default=5, help="Days to skip after counting a drop event")
    parser.add_argument(
        "--min-dollar-vol", type=float, default=3_000_000,
        help="Minimum 20D avg dollar volume ($) to treat as liquid"
    )
    args = parser.parse_args()

    analyze_all(
        symbols,
        refresh=args.refresh,
        min_events=args.min_events,
        cooldown=args.cooldown,
        min_dollar_vol=args.min_dollar_vol,
    )