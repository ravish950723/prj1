from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd



from .backtest import evaluate_backtest_accuracy
from .compute import compute_indicators
from .darvas import darvas_box_signal
from .exit_signals import compute_exit_signals
from .institutional_investor import score_institutional_investor
from .macro_features import enrich_with_macro_features
from .rank_long_short import (
    RankConfig,
    build_long_plan,
    build_short_plan,
    compute_common_indicators_rls,
    detect_long_setup,
    detect_short_setup,
    explain_spike_drop,
    score_long,
    score_short,
    verdicts,
)
from .symbol_analysis import analyze_symbol
from .upward import (
    compute_signal_score,
    compute_upward_trend,
    detect_bullish_engulfing,
    detect_hammer,
    detect_mean_reversion_buy,
    detect_smc_accumulation_breakout,
)

from .quant import compute_quant_features
from .utils import safe_float, safe_str, yn


def _infer_market_substage(df: pd.DataFrame) -> str:
    """
    Fallback only. Primary source should be compute.py -> df['market_substage'].
    """
    try:
        if "market_substage" in df.columns and len(df) > 0:
            val = str(df["market_substage"].iloc[-1]).strip()
            if val:
                return val

        close = pd.to_numeric(df["close"], errors="coerce")
        ema20 = pd.to_numeric(df.get("EMA_20", pd.Series(np.nan, index=df.index)), errors="coerce")
        ema50 = pd.to_numeric(df.get("EMA_50", pd.Series(np.nan, index=df.index)), errors="coerce")
        ema200 = pd.to_numeric(df.get("EMA_200", pd.Series(np.nan, index=df.index)), errors="coerce")
        adx = pd.to_numeric(df.get("ADX_14", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
        vol_surge = pd.to_numeric(df.get("VOL_SURGE_RATIO", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)

        c = float(close.iloc[-1])
        e20 = float(ema20.iloc[-1])
        e50 = float(ema50.iloc[-1])
        e200 = float(ema200.iloc[-1])
        a = float(adx.iloc[-1])
        v = float(vol_surge.iloc[-1])

        if not all(np.isfinite(x) for x in [c, e20, e50, e200]):
            return "BASE_FORMATION"

        if e20 > e50 > e200 and c >= e20:
            if a >= 30 and v >= 1.8:
                return "HIGH_VOLUME_BREAKOUT (CTA_BREAKOUT)"
            return "EMA_STACK_FORMATION (BULL_STACKED)"

        if e20 < e50 < e200 and c <= e20:
            if a >= 30 and v >= 1.5:
                return "MOMENTUM_SELLOFF"
            return "EMA_STACK_FORMATION (BEAR_STACKED)"

        return "BASE_FORMATION"
    except Exception:
        return "BASE_FORMATION"




def _weekly_upper_wick_high_volume(df: pd.DataFrame) -> tuple[str, float]:
    try:
        w = df.copy()
        w['date'] = pd.to_datetime(w['date'], errors='coerce')
        w = w.dropna(subset=['date']).set_index('date').resample('W-FRI').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        if len(w) < 13:
            return 'N', 0.0
        last = w.iloc[-1]
        rng = float(last['high'] - last['low'])
        if not np.isfinite(rng) or rng <= 0:
            return 'N', 0.0
        upper_wick = float(last['high'] - max(last['open'], last['close']))
        wick_ratio = upper_wick / rng
        prev_high = float(w['high'].iloc[-13:-1].max())
        pct_from_prev = ((float(last['high']) - prev_high) / prev_high * 100.0) if prev_high > 0 else 0.0
        avg_vol = float(w['volume'].iloc[-13:-1].mean())
        flag = wick_ratio >= 0.55 and float(last['volume']) >= 1.2 * avg_vol
        return yn(flag), round(pct_from_prev, 2)
    except Exception:
        return 'N', 0.0


def compute_feature_row(symbol: str, df: pd.DataFrame, source: str, qqq_df: Optional[pd.DataFrame], cfg: RankConfig) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        'Symbol': symbol,
        'DATA_SOURCE': source,
        'ASOF_DATE': datetime.utcnow().strftime('%Y-%m-%d'),
    }
    if df is None or df.empty:
        return out

    df = compute_indicators(df.copy(), symbol=symbol)
    df = enrich_with_macro_features(df)
    df = darvas_box_signal(df)
    df = compute_upward_trend(df)
    last = df.iloc[-1]

    out["HIGHER_HIGH_HIGHER_LOW"] = yn(bool(last.get("HIGHER_HIGH_HIGHER_LOW", 0)))
    out["LOWER_HIGH_LOWER_LOW"] = yn(bool(last.get("LOWER_HIGH_LOWER_LOW", 0)))
    out["EMA_STACKED_BULLISH"] = yn(bool(last.get("EMA_STACKED_BULLISH", 0)))
    out["EMA_STACKED_BEARISH"] = yn(bool(last.get("EMA_STACKED_BEARISH", 0)))
    out["RANGE_BOUND"] = yn(bool(last.get("RANGE_BOUND", 0)))
    out["TIGHT_RANGE_CONSOLIDATION"] = yn(bool(last.get("TIGHT_RANGE_CONSOLIDATION", 0)))
    out["SQUEEZE"] = yn(bool(last.get("SQUEEZE", 0)))
    out["RESISTANCE_BREAKOUT"] = yn(bool(last.get("RESISTANCE_BREAKOUT", 0)))
    out["SUPPORT_BREAKDOWN"] = yn(bool(last.get("SUPPORT_BREAKDOWN", 0)))
    out["HIGH_VOLUME_BREAKOUT"] = yn(bool(last.get("HIGH_VOLUME_BREAKOUT", 0)))
    out["FALSE_BREAKOUT"] = yn(bool(last.get("FALSE_BREAKOUT", 0)))
    out["FALSE_BREAKDOWN"] = yn(bool(last.get("FALSE_BREAKDOWN", 0)))
    out["HAMMER"] = yn(bool(last.get("HAMMER", 0)))
    out["BULLISH_ENGULFING"] = yn(bool(last.get("BULLISH_ENGULFING", 0)))
    out["BEARISH_ENGULFING"] = yn(bool(last.get("BEARISH_ENGULFING", 0)))
    out["DOJI"] = yn(bool(last.get("DOJI", 0)))
    out["VOLUME_SPIKE"] = yn(bool(last.get("VOLUME_SPIKE", 0)))
    out["VOLUME_DRY_UP"] = yn(bool(last.get("VOLUME_DRY_UP", 0)))
    out["LOW_VOLUME_PULLBACK"] = yn(bool(last.get("LOW_VOLUME_PULLBACK", 0)))
    out["RSI_DIVERGENCE_BULLISH"] = yn(bool(last.get("RSI_DIVERGENCE_BULLISH", 0)))
    out["RSI_DIVERGENCE_BEARISH"] = yn(bool(last.get("RSI_DIVERGENCE_BEARISH", 0)))
    out["MACD_BULLISH_CROSSOVER"] = yn(bool(last.get("MACD_BULLISH_CROSSOVER", 0)))
    out["MACD_BEARISH_CROSSOVER"] = yn(bool(last.get("MACD_BEARISH_CROSSOVER", 0)))
    out["VWAP_RECLAIM"] = yn(bool(last.get("VWAP_RECLAIM", 0)))
    out["VWAP_REJECTION"] = yn(bool(last.get("VWAP_REJECTION", 0)))
    out["GAP_UP"] = yn(bool(last.get("GAP_UP", 0)))
    out["GAP_DOWN"] = yn(bool(last.get("GAP_DOWN", 0)))
    out["GAP_FILL"] = yn(bool(last.get("GAP_FILL", 0)))
    out["SUPPORT_RESPECT"] = yn(bool(last.get("SUPPORT_RESPECT", 0)))
    out["RESISTANCE_REJECTION"] = yn(bool(last.get("RESISTANCE_REJECTION", 0)))
    out["RANGE_EXPANSION"] = yn(bool(last.get("RANGE_EXPANSION", 0)))

    pattern_cols = [
        col for col in df.columns
        if col.isupper() and df[col].iloc[-1] == 1
    ]

    priority = [
        "HIGH_VOLUME_BREAKOUT",
        "ACCUMULATION_BREAKOUT",
        "VWAP_RECLAIM",
        "BULL_FLAG",
        "ASCENDING_TRIANGLE",
        "BULLISH_ENGULFING",
        "HAMMER",
    ]

    sorted_hits = sorted(pattern_cols, key=lambda x: (x not in priority, x))

    out["Pattern Detected"] = ", ".join(sorted_hits[:5]) if sorted_hits else "NONE"

    close = safe_float(last.get('close'))
    out['Current Price'] = close
    out['VWAP Support'] = safe_float(last.get('vwap_support'))
    out['ADX'] = safe_float(last.get('ADX_14'))
    out['RSI'] = safe_float(last.get('RSI_14'))
    out['EMA21 Slope'] = safe_float(last.get('EMA_21_slope'))
    out['MACD_SIGNAL'] = safe_float(last.get('MACD_signal'))
    out['MACD_HIST'] = safe_float(last.get('MACD_hist'))
    out['ATR14'] = safe_float(last.get('ATR_14'))
    out['VWAP'] = safe_float(last.get('vwap_support'))
    out['Trend'] = safe_str(last.get('HTF_Trend', 'Neutral'), 'Neutral')
    out['Sym Vol Regime'] = safe_float(last.get('sym_vol_regime'))
    out['VIX Vol Regime'] = safe_float(last.get('VIX_vol_regime'))
    out['Volume Pressure'] = round(safe_float(last.get('volume_weight'), 0.0), 4)
    out['EMA Uptrend'] = yn(bool(last.get('EMA_uptrend', False)))
    out['MACD Cross'] = yn(bool(last.get('MACD_Crossover', last.get('MACD_crossover', False))))
    out['RSI State'] = 'OVERBOUGHT' if out['RSI'] >= 70 else 'OVERSOLD' if out['RSI'] <= 30 else 'BULLISH' if out['RSI'] >= 55 else 'BEARISH' if out['RSI'] <= 45 else 'NEUTRAL'
    out['ADX Strength'] = 'STRONG' if out['ADX'] >= 30 else 'MODERATE' if out['ADX'] >= 20 else 'WEAK'
    out['At BB Lower'] = yn(close <= safe_float(last.get('BB_lower'), 0.0) if safe_float(last.get('BB_lower'), 0.0) > 0 else False)
    out['OBV Trend'] = 'UP' if safe_float(last.get('OBV')) >= safe_float(df['OBV'].iloc[-5] if len(df) >= 5 else 0.0) else 'DOWN'



    common = compute_common_indicators_rls(df.copy(), qqq_df=qqq_df)
    out.update(common)
    mapping = {
        'Refined Buy Price': 'PRICE',
        'DMA20': 'DMA20', 'DMA50': 'DMA50', 'DMA100': 'DMA100', 'DMA150': 'DMA150', 'DMA200': 'DMA200',
        'PCT_FROM_DMA20': 'PCT_FROM_DMA20', 'PCT_FROM_DMA50': 'PCT_FROM_DMA50', 'PCT_FROM_DMA200': 'PCT_FROM_DMA200',
        'VOL_TODAY': 'VOL_TODAY', 'AVG_VOL_20D': 'AVG_VOL_20D', 'VOL_SURGE_RATIO': 'VOL_SURGE_RATIO',
        'VOL_TREND_5D': 'VOL_TREND_5D', 'GAP_PCT': 'GAP_PCT', 'REL_STRENGTH_20D_VS_QQQ': 'REL_STRENGTH_20D_VS_QQQ',
        'BID_ASK_SPREAD_PCT': 'BID_ASK_SPREAD_PCT', 'LIQUIDITY_SCORE': 'LIQUIDITY_SCORE',
        'BORROW_FEE_PCT': 'BORROW_FEE_PCT', 'VWAP_DISTANCE_PCT': 'VWAP_DISTANCE_PCT', 'ATR14_PCT': 'ATR14_PCT',
        'DISTRIBUTION_DAYS_20D': 'DISTRIBUTION_DAYS_20D', 'ACCUMULATION_DAYS_20D': 'ACCUMULATION_DAYS_20D',
        'LONG_SCORE': 'LONG_SCORE', 'LONG_SETUP_TAG': 'LONG_SETUP_TAG', 'LONG_VERDICT': 'LONG_VERDICT',
        'SHORT_SCORE': 'SHORT_SCORE', 'SHORT_SETUP_TAG': 'SHORT_SETUP_TAG', 'SHORT_VERDICT': 'SHORT_VERDICT',
        'LONG_ENTRY_ZONE_LOW': 'LONG_ENTRY_ZONE_LOW', 'LONG_ENTRY_ZONE_HIGH': 'LONG_ENTRY_ZONE_HIGH',
        'LONG_INVALIDATION': 'LONG_INVALIDATION', 'LONG_TARGET_1': 'LONG_TARGET_1', 'LONG_TARGET_2': 'LONG_TARGET_2', 'LONG_RR_RATIO': 'LONG_RR_RATIO',
        'SHORT_ENTRY_ZONE_LOW': 'SHORT_ENTRY_ZONE_LOW', 'SHORT_ENTRY_ZONE_HIGH': 'SHORT_ENTRY_ZONE_HIGH',
        'SHORT_INVALIDATION': 'SHORT_INVALIDATION', 'SHORT_TARGET_1': 'SHORT_TARGET_1', 'SHORT_TARGET_2': 'SHORT_TARGET_2', 'SHORT_RR_RATIO': 'SHORT_RR_RATIO',
        'SHORT_FEASIBILITY': 'SHORT_FEASIBILITY', 'SPIKE_DRIVER': 'SPIKE_DRIVER', 'DROP_DRIVER': 'DROP_DRIVER', 'DMA_STACK': 'DMA_STACK',
    }
    for tgt, src in mapping.items():
        if src in common:
            out[tgt] = common.get(src)

    long_setup = detect_long_setup(common, df, cfg)
    short_setup = detect_short_setup(common, df, cfg)
    long_score = float(score_long(common, cfg)) if long_setup else 0.0
    short_score = float(score_short(common, cfg)) if short_setup else 0.0
    long_plan = build_long_plan(common, df, long_setup, cfg) if long_setup else {}
    short_plan = build_short_plan(common, df, short_setup, cfg) if short_setup else {}
    spike_driver, drop_driver = explain_spike_drop(common, long_setup, short_setup)
    long_verdict, short_verdict = verdicts(long_score, short_score, long_setup, short_setup, common, cfg)
    out.update({'LONG_SCORE': long_score, 'SHORT_SCORE': short_score, 'LONG_VERDICT': long_verdict, 'SHORT_VERDICT': short_verdict, 'SPIKE_DRIVER': spike_driver, 'DROP_DRIVER': drop_driver})
    out.update(long_plan)
    out.update(short_plan)

    try:
        analysis = analyze_symbol(symbol, df_raw=df.copy()) or {}
    except Exception:
        analysis = {}
    out.update({k: v for k, v in analysis.items() if v is not None})

    inst = score_institutional_investor(df.copy())
    out['Institutional Score'] = safe_float(inst, 0.5)

    signal_score_val = compute_signal_score(df.copy())
    if isinstance(signal_score_val, tuple):
        tech_score, signal_count, signal_label = signal_score_val
    else:
        tech_score, signal_count, signal_label = safe_float(signal_score_val), 0, 'HOLD'
    out['Tech Fallback Score'] = round(safe_float(tech_score), 4)
    out['Signal'] = signal_label
    out['Signal Score'] = round(safe_float(last.get('signal_score', tech_score)), 4)
    out['Volume Weight'] = round(safe_float(last.get('volume_weight'), 0.0), 4)
    out['Confidence Score'] = round(safe_float(last.get('confidence_score'), 0.0), 4)
    out['CONFIDENCE_SCORE'] = out['Confidence Score']
    out['Trend_Strength'] = round(safe_float(last.get('trend_strength'), 0.0), 4)
    out['Breakout'] = yn(bool(last.get('darvas_signal', False)))
    out['Undervalued'] = yn(close <= out.get('VWAP Support', 0.0) if out.get('VWAP Support', 0.0) else False)
    out['Price Reversal'] = yn(bool(detect_bullish_engulfing(df.copy()) or detect_hammer(df.copy())))
    out['SMC_Breakout'] = yn(bool(detect_smc_accumulation_breakout(df.copy())))
    out['Mean_Reversion'] = yn(bool(detect_mean_reversion_buy(df.copy())))
    out['Bullish_Engulfing'] = yn(bool(detect_bullish_engulfing(df.copy())))
    out['Hammer'] = yn(bool(detect_hammer(df.copy())))
    out['Rule-Based Buy'] = yn(signal_label == 'BUY')
    out['Darvas Signal'] = yn(bool(last.get('darvas_signal', 0) == 1))
    out['Darvas Breakout %'] = round(safe_float(last.get('darvas_breakout_pct')), 4)

    out['Market Stage'] = safe_str(last.get('market_stage'), 'Neutral/Transition')
    out['Market Sub-Stage'] = safe_str(last.get('market_substage'), _infer_market_substage(df))
    week_flag, pct_prev = _weekly_upper_wick_high_volume(df)
    out['Whether Weekly chart has got higher up wicks volume.'] = week_flag
    out['How much % high Weekly chart is from previous lower volume.'] = pct_prev
    out['Whether the current DMA is greater than 50 DMA.'] = yn(safe_float(out.get('DMA20')) > safe_float(out.get('DMA50')))
    out['Whether the current DMA is greater than 100 DMA.'] = yn(safe_float(out.get('DMA20')) > safe_float(out.get('DMA100')))
    out['Whether the current DMA is greater than 150 DMA.'] = yn(safe_float(out.get('DMA20')) > safe_float(out.get('DMA150')))
    out['Whether the current DMA is greater than 200.'] = yn(safe_float(out.get('DMA20')) > safe_float(out.get('DMA200')))

    out.update(compute_quant_features(df.copy(), qqq_df=qqq_df))

    # ✅ USE ALREADY-INJECTED VALUES (from pipeline)

    out['EPS Increase 2Q'] = yn(bool(out.get('eps_inc_2q', 0)))
    out['EPS Increase 3Q'] = yn(bool(out.get('eps_inc_3q', 0)))
    out['EPS Increase 4Q'] = yn(bool(out.get('eps_inc_4q', 0)))

    out['Sentiment Label'] = (
        'POSITIVE' if safe_float(out.get('news_sentiment_score')) > 0.15
        else 'NEGATIVE' if safe_float(out.get('news_sentiment_score')) < -0.15
        else 'NEUTRAL'
    )

    # exits
    exit_info = compute_exit_signals(df.copy(), entry_price=max(safe_float(out.get('Refined Buy Price')), close), atr_mult=2.0)
    out['Atr Trailing Stop'] = safe_float(exit_info.get('AtrTrailingStop'))
    out['Exit Now'] = yn(bool(exit_info.get('ExitNow', False)))
    out['Exit Reasons'] = safe_str(exit_info.get('ExitReasons'), '')

    # backtest proxy
    hit, gain, days = evaluate_backtest_accuracy(symbol, df.copy(), out.get('Refined Buy Price', close), gain_threshold=0.15, lookahead_days=90)
    out['90D Hit'] = yn(hit)
    out['90D Gain (%)'] = round(safe_float(gain), 4)
    out['Days to Peak'] = int(days) if days >= 0 else 0

    # alias cleanup / fill
    out['Final_Action'] = safe_str(out.get('Recommendation', 'WATCH'), 'WATCH')
    out['FINAL_ACTION'] = out['Final_Action']
    out['CONFIDENCE_SCORE'] = safe_float(out.get('Confidence Score', out.get('CONFIDENCE_SCORE', 0.0)))
    out['Recommendation'] = safe_str(out.get('Recommendation', out['Final_Action']), out['Final_Action'])
    out['Momentum Recommendation'] = safe_str(out.get('Momentum Recommendation', out['Recommendation']), out['Recommendation'])
    out['Decision Reason'] = safe_str(out.get('Decision Reason', 'feature-engineering'))
    out['Momentum Decision Reason'] = safe_str(out.get('Momentum Decision Reason', out['Decision Reason']))
    out['Pattern Detected'] = safe_str(out.get('Pattern Detected', 'MULTI_FACTOR'))
    out['DipReclaim'] = yn(safe_float(out.get('VWAP Support')) > 0 and close >= safe_float(out.get('VWAP Support')))
    return out
