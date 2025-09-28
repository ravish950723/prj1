from data_fetcher import fetch_historical_data
from final.sellprice.signals import _passes_rr_filter
from utils import log_signal, log_confidence_score, get_sector_info
from datetime import date


import numpy as np
import pandas as pd


_portfolio_short_budget = {"budget": 1.0, "used": 0.0}  # 100% of desired short book

_signal_last_fired = {}

def _cooldown(symbol, condition, ts, days=3):
    key = (symbol, condition)
    t = _signal_last_fired.get(key)
    if t is None:
        return False
    try:
        return (ts - t).days < days
    except Exception:
        return False

def _mark_fired(symbol, condition, ts):
    _signal_last_fired[(symbol, condition)] = ts

def _score_confidence(df, latest, signal_type, condition, sector_type):
    """
    0..100. Combines trend quality (ADX), extension vs 200-DEMA, MACD context,
    OBV divergence slope, volatility penalty, and a small sector bias.
    """
    try:
        price = float(latest['close'])
        adx = float(df['adx'].iloc[-1]) if 'adx' in df.columns else np.nan
        atr_pct = float(df['atr'].iloc[-1]) / max(price, 1e-9)
        macd = float(latest.get('macd', np.nan))
        macd_sig = float(latest.get('macd_signal', np.nan))
        p200 = float(latest.get('200_dema', np.nan))
        obv = df['obv'].iloc[-1] if 'obv' in df.columns else np.nan
        obv_prev = df['obv'].iloc[-5] if 'obv' in df.columns and len(df)>=5 else np.nan

        score = 50.0

        # Trend quality
        if np.isfinite(adx):
            score += min(20.0, max(-10.0, (adx - 20.0)))   # ADX 40 ⇒ +20

        # Volatility penalty (uncertainty)
        score -= min(20.0, max(0.0, (atr_pct - 0.03) * 300))

        # Extension vs 200-DEMA
        if np.isfinite(p200):
            dist200 = (price - p200) / max(p200, 1e-9)
            if signal_type in ("SELL", "SHORT"):
                score += min(25.0, max(0.0, dist200 * 100))
            else:
                score += min(15.0, max(0.0, (-dist200) * 100))

        # MACD context
        if np.isfinite(macd) and np.isfinite(macd_sig):
            if signal_type in ("SELL", "SHORT"):
                score += min(10.0, max(0.0, -(macd - macd_sig) * 50))
            else:
                score += min(10.0, max(0.0,  (macd - macd_sig) * 50))

        # OBV divergence strength
        if np.isfinite(obv) and np.isfinite(obv_prev) and condition and "OBV Divergence" in condition:
            obv_slope = (obv - obv_prev) / 5.0
            # reward negative slope on SELL/SHORT when price is rising (already implied by divergence)
            score += min(15.0, max(0.0, (-np.sign(obv_slope)) * abs(obv_slope) / (abs(obv_prev)+1e-9) * 10000))

        # Sector bias
        if sector_type == "etf":
            score -= 5.0
        elif sector_type == "finance":   # miners/brokers in your map
            score += 3.0

        return round(max(0.0, min(100.0, score)), 1)
    except Exception:
        return 0.0

allow_short_on_weak_adx = True  # Configurable flag

def check_weekly_trend(symbol):
    df_weekly = fetch_historical_data(symbol, bar_size='1 week', duration='2 Y')
    df_weekly['50_w_ema'] = df_weekly['close'].ewm(span=50, adjust=False).mean()
    df_weekly['200_w_ema'] = df_weekly['close'].ewm(span=200, adjust=False).mean()
    is_bullish = df_weekly['50_w_ema'].iloc[-1] > df_weekly['200_w_ema'].iloc[-1]
    print(f"[TREND] Weekly trend for {symbol}: {'Bullish' if is_bullish else 'Bearish'}")
    return is_bullish

def check_sell_signals(df):
    latest, prev = df.iloc[-1], df.iloc[-2]
    actions = []

    # Sector-aware thresholds for overbought vs 200-DEMA
    sector_type, *_ = get_sector_info(latest['symbol'])

    # RSI reversal with MACD bearish cross
    if (
        prev['rsi'] > 70 and latest['rsi'] < 70 and
        prev['macd'] > prev['macd_signal'] and latest['macd'] < latest['macd_signal']
    ):
        actions.append("RSI Reversal & MACD Bearish Crossover")

    # Overbought check: ETFs tend to be less explosive than miners -> lower threshold
    threshold = 25 if sector_type == 'etf' else 30
    if latest['%_above_dema'] > threshold and latest['rsi'] > 70:
        actions.append(f"Overbought (>{threshold}% above 200 DEMA)")

    # Trendline break requires volume confirmation
    if latest['trendline_break'] and latest['volume'] > df['volume'].rolling(20).mean().iloc[-1] * 1.5:
        actions.append("Trendline Break with Volume")

    # MACD below zero & below signal (structural weakness)
    if latest['macd'] < 0 and latest['macd'] < latest['macd_signal']:
        actions.append("MACD Bearish below zero")

    # OBV divergence (OBV lower while price rising)
    if (df['obv'].iloc[-1] < df['obv'].rolling(50).max().iloc[-1] * 0.95) and (latest['close'] > prev['close']):
        actions.append("OBV Divergence")

    # High-volume price drop
    if latest['close'] < prev['close'] and latest['volume'] > df['volume'].rolling(20).mean().iloc[-1] * 1.5:
        actions.append("High Volume Price Drop")

    # Close below 50-DEMA
    if latest['close'] < df['50_dema'].iloc[-1] and prev['close'] > df['50_dema'].iloc[-2]:
        actions.append("Close Below 50 DEMA")

    # ATR trailing stop breach
    atr = df['atr'].iloc[-1]
    trailing_stop = latest['close'] - 2 * atr
    if latest['close'] < trailing_stop:
        actions.append("ATR Trailing Stop Hit")

    return actions


def check_exit_signals(df):
    actions = check_sell_signals(df)

    # Optional: skip noisy roads when ADX is weak (unless explicitly allowed)
    adx = df['adx'].iloc[-1] if 'adx' in df.columns else 25
    if adx < 20 and not allow_short_on_weak_adx:
        print("[FILTER] Weak ADX – skipping sell signals due to market noise.")
        return []

    symbol = df.iloc[-1]['symbol']
    date_str = df.index[-1].strftime("%Y-%m-%d")
    latest = df.iloc[-1]
    sector_type, *_ = get_sector_info(symbol)

    for action in actions:
        # SHORT-able includes dynamic Overbought text
        is_shortable = (
            action in ["MACD Bearish below zero", "Close Below 50 DEMA", "High Volume Price Drop"]
        ) or action.startswith("Overbought (")
        signal_type = "SHORT" if is_shortable else "SELL"

        # Gate shorts by trend quality (avoid low-trend whipsaws)
        if signal_type == "SHORT" and adx < 18:
            continue

        # Confidence
        conf = _score_confidence(df, latest, signal_type, action, sector_type)

        # Confluence boosts
        if action.startswith("Overbought (") and "OBV Divergence" in actions:
            conf = min(100.0, conf + 7.0)
        if "MACD Bearish below zero" in actions and action != "MACD Bearish below zero":
            conf = min(100.0, conf + 3.0)
        if action == "Trendline Break with Volume" and "High Volume Price Drop" in actions:
            conf = min(100.0, conf + 5.0)

        # Cooldown
        if _cooldown(symbol, action, df.index[-1], days=3):
            print(f"[FILTER] Duplicate {signal_type} suppressed (cooldown).")
            continue

        _mark_fired(symbol, action, df.index[-1])

        # Position sizing (confidence + volatility)
        price = float(latest['close'])
        atr_pct = float(df['atr'].iloc[-1]) / max(price, 1e-9)
        size_pct = max(0.0, min(3.5, 1.0 * (conf/70.0) * (0.03 / max(atr_pct, 1e-6))))

        if signal_type == "SHORT":
            # Volatility-aware short budget
            window = df.tail(50)
            basket_atr_pct = (window['atr'] / window['close']).mean()
            _portfolio_short_budget["budget"] = 0.5 if basket_atr_pct < 0.03 else 0.3

            remaining = _portfolio_short_budget["budget"] - _portfolio_short_budget["used"]
            if remaining <= 0:
                print("[FILTER] Short budget exhausted; skipping additional shorts.")
                continue

            # Respect remaining budget (budget is fraction, size_pct is %)
            size_pct = min(size_pct, max(0.0, remaining * 100.0))
            if size_pct <= 0:
                print("[FILTER] Short position size is zero after budget cap; skipping.")
                continue

            _portfolio_short_budget["used"] += size_pct / 100.0

        log_signal(
            symbol, date_str, action, round(conf, 1),
            signal_type=signal_type, condition=action,
            ATRpct=round(atr_pct*100, 2), SizePct=round(size_pct, 2)
        )

    if not actions:
        print("[INFO] No strong sell signals. Hold or monitor closely.")

    return actions



def check_entry_signals(df):
    """
    BUY logic with clearer diagnostics and slightly easier ETF volume Z in calm regimes.
    Also fixes WHY-NOT text: shows explicit comparisons and ✓/× per gate.
    """
    import numpy as np
    import pandas as pd
    from datetime import date

    latest = df.iloc[-1]
    symbol = latest.get("symbol", "UNKNOWN")
    today = str(date.today())

    sector_type, rsi_base, vol_threshold, weight = get_sector_info(latest['symbol'])
    print(f"[INFO] Checking entry signals for {latest.name} ({latest['symbol']}) in sector '{sector_type}':")

    # --- Volume Z ---
    vol_ma20 = df['volume'].rolling(20).mean().iloc[-1]
    vol_sd20 = df['volume'].rolling(20).std().iloc[-1]
    vol_z = (latest['volume'] - vol_ma20) / (vol_sd20 + 1e-9)

    price = float(latest['close'])
    atr = float(df['atr'].iloc[-1])
    atr_pct = atr / max(price, 1e-9)

    # Base min_z by ATR% (calm → need more z; wild → allow lower z)
    base_min_z = float(np.interp(atr_pct, [0.015, 0.03, 0.06], [2.0, 1.5, 1.0]))

    # ETF relax (a touch more than before in calm regimes)
    if sector_type == "etf":
        # Give up to -0.35 relief when ATR% is very low; taper to -0.15 by 3%
        etf_relief = float(np.interp(atr_pct, [0.0, 0.015, 0.03], [0.35, 0.30, 0.15]))
        base_min_z = max(0.85, base_min_z - etf_relief)

    min_z = base_min_z
    volume_spike = vol_z >= min_z

    # Allow "good enough" volume when volatility & trend are already supportive
    adx = df['adx'].iloc[-1] if 'adx' in df.columns else 25
    vol_ok = volume_spike or (atr_pct >= 0.06 and adx >= 21 and vol_z >= -0.25)

    # ADX gates (ETF slightly easier)
    adx_gate_buy = 20 if sector_type == "etf" else 22
    adx_momo_gate = 21 if sector_type == "etf" else 23

    # --- Dynamic RSI ---
    rsi_now = float(latest['rsi'])
    rsi_200 = df['rsi'].rolling(200)
    rsi_20p = rsi_200.apply(lambda s: pd.Series(s).quantile(0.20), raw=True).iloc[-1]
    rsi_40p = rsi_200.apply(lambda s: pd.Series(s).quantile(0.40), raw=True).iloc[-1]
    dyn_base = min(rsi_40p, max(20, rsi_20p, rsi_base - 2))

    below_50 = latest['close'] < latest['50_dema']

    macd_hist = latest['macd'] - latest['macd_signal']
    macd_hist_prev = (df['macd'] - df['macd_signal']).shift(1).iloc[-1]
    macd_hist_uptick = macd_hist > macd_hist_prev

    dyn_adj = 0.0
    if below_50:
        dyn_adj += 2.0            # mean reversion cushion
    if macd_hist_uptick:
        dyn_adj += 1.0            # early turn cushion
    if sector_type == "etf":
        dyn_adj += 1.0            # ETF cushion

    dynamic_rsi_threshold = float(dyn_base + dyn_adj)

    # --- Trend / momentum helpers ---
    macd_up = latest['macd'] > latest['macd_signal']
    macd_ok = macd_up or macd_hist_uptick  # accept histogram uptick

    above_50 = latest['close'] > latest['50_dema']
    above_200 = latest['close'] > latest['200_dema']
    above_mas = above_50 and above_200

    rsi_prev2 = df['rsi'].shift(2).iloc[-1] if len(df) >= 3 else df['rsi'].iloc[-1]
    rsi_rising = rsi_now > float(rsi_prev2)

    not_overextended = latest['close'] <= 1.27 * latest['200_dema']

    # --- 2-bar near reclaim (slightly wider window; hist uptick acceptable) ---
    y_close = df['close'].shift(1).iloc[-1]
    y_50 = df['50_dema'].shift(1).iloc[-1]
    y_200 = df['200_dema'].shift(1).iloc[-1]
    y_macd = df['macd'].shift(1).iloc[-1]
    y_macd_sig = df['macd_signal'].shift(1).iloc[-1]
    y_adx = df['adx'].shift(1).iloc[-1] if 'adx' in df.columns else adx
    y_hist = (df['macd'] - df['macd_signal']).shift(1).iloc[-1]
    yy_hist = (df['macd'] - df['macd_signal']).shift(2).iloc[-1]
    y_hist_uptick = y_hist > yy_hist

    y_near_reclaim = (
        (y_close > 0.985 * y_50) and
        (y_close > y_200) and
        ((y_macd > y_macd_sig) or y_hist_uptick) and
        (y_adx >= (adx_gate_buy - 1))
    )

    # --- BUY paths ---
    primary_buy = (
        (rsi_now < dynamic_rsi_threshold) and macd_ok and above_mas and vol_ok and not_overextended
    )

    pattern = latest.get('pattern')
    near_50_now = latest['close'] > 0.985 * latest['50_dema']
    pattern_ok = (
        (pattern in ['Double Bottom', 'Triple Bottom', 'Cup and Handle', 'Ascending Triangle']) and
        (above_200) and
        ((above_50) or (near_50_now and macd_ok and vol_ok)) and
        (adx >= (adx_gate_buy - (1 if sector_type == "etf" else 0))) and
        not_overextended
    )

    dip_buy = (
        above_200 and
        above_50 and
        (df['close'].shift(1).iloc[-1] < df['50_dema'].shift(1).iloc[-1]) and
        macd_ok and vol_ok and not_overextended
    )

    prior_20d_high = df['high'].rolling(20).max().shift(1).iloc[-1]
    breakout_buy = (
        (latest['close'] > prior_20d_high) and macd_ok and (adx >= adx_gate_buy) and vol_ok and not_overextended
    )

    momentum_buy = (
        above_mas and macd_ok and (adx >= adx_momo_gate)
        and rsi_rising and (38 <= rsi_now <= 68)
        and vol_ok and not_overextended
    )

    confirm_reclaim_buy = (
        y_near_reclaim and above_200 and above_50 and macd_ok and vol_ok and not_overextended
    )

    # --- Fire BUY if any path passes ---
    if primary_buy or pattern_ok or dip_buy or breakout_buy or momentum_buy or confirm_reclaim_buy:
        reason = format_buy_reason(
            primary_buy, pattern_ok, dip_buy, breakout_buy, momentum_buy, confirm_reclaim_buy, pattern
        )

        if not _passes_rr_filter(df, latest):
            print("[FILTER] R:R gate blocked the BUY.")
            return

        if _cooldown(symbol, reason, latest.name, days=3):
            print("[FILTER] Duplicate BUY suppressed (cooldown).")
            return
        _mark_fired(symbol, reason, latest.name)

        conf = _score_confidence(df, latest, "BUY", reason, sector_type)
        pattern_conf = latest.get('pattern_confidence', np.nan)
        if np.isfinite(pattern_conf):
            conf = min(100.0, max(conf, pattern_conf * 0.8))
        conf = min(100.0, max(conf, 100.0 * weight))

        size_pct = max(0.0, min(3.5, 1.0 * (conf / 70.0) * (0.03 / max(atr_pct, 1e-6))))

        log_signal(
            symbol, today, reason, round(conf, 1),
            signal_type="BUY",
            condition=(
                "Momentum" if momentum_buy else
                "Breakout" if breakout_buy else
                "DipReclaim" if dip_buy else
                "Pattern" if pattern_ok else
                "ConfirmReclaim" if confirm_reclaim_buy else
                "Continuation"
            ),
            ATRpct=round(atr_pct * 100, 2),
            SizePct=round(size_pct, 2)
        )
        log_confidence_score(conf, "BUY confidence")
        return

    # --- Near-miss WATCH ---
    passed = {
        "vol_ok": bool(vol_ok),
        "macd_ok": bool(macd_ok),
        "adx_ok": bool(adx >= (adx_gate_buy - 1)),
        "above_200": bool(above_200),
        "near_50": bool(latest['close'] > 0.985 * latest['50_dema']),
    }
    score = sum(passed.values())
    if score >= 3 and passed["above_200"] and passed["near_50"]:
        watch_reason = "Watch: Near 50-DEMA with MACD/ADX/Vol OK"
        if not _cooldown(symbol, watch_reason, latest.name, days=2):
            conf = 45.0
            log_signal(
                symbol, today, watch_reason, conf,
                signal_type="WATCH", condition="NearReclaim",
                ATRpct=round(atr_pct * 100, 2), SizePct=0.0
            )

    # --- Clearer diagnostics ---
    primary_flags = dict(
        vol_ok=vol_ok, macd_ok=macd_ok, above_50=above_50, above_200=above_200,
        adx_ok=(adx >= adx_gate_buy), rsi_rising=rsi_rising, not_overext=not_overextended
    )

    print(
        "[DEBUG] BUY gates → "
        f"vol_z={vol_z:.2f} (min_z={min_z:.2f}), "
        f"dyn_rsi_th={dynamic_rsi_threshold:.1f}, "
        f"primary={primary_buy}, pattern={pattern_ok}, dip={dip_buy}, breakout={breakout_buy}, "
        f"momentum={momentum_buy}, confirm_reclaim={confirm_reclaim_buy}, atr%={atr_pct*100:.2f}"
    )

    # Human-readable WHY-NOT with explicit comparisons and pass/fail marks
    why_bits = [
        f"vol: {vol_z:.2f} {'≥' if vol_ok else '<'} {min_z:.2f}" + (" ✓" if vol_ok else " ×"),
        f"RSI vs dyn_th: {rsi_now:.1f} ?< {dynamic_rsi_threshold:.1f} " + ("✓" if rsi_now < dynamic_rsi_threshold else "×"),
        f"MACD>sig/Hist " + ("✓" if macd_ok else "×"),
        f">50 {'✓' if above_50 else '×'}, >200 {'✓' if above_200 else '×'}",
        f"ADX {adx:.1f} {'≥' if adx >= adx_gate_buy else '<'} {adx_gate_buy} " + ("✓" if adx >= adx_gate_buy else "×"),
        f"RSI_rising {'✓' if rsi_rising else '×'}",
        f"overextended {'×' if not not_overextended else '✓'}",
    ]
    print("[WHY-NOT] " + ", ".join(why_bits))
    print("[INFO] No strong buy signals.")


def format_buy_reason(primary_buy, pattern_ok, dip_buy, breakout_buy, momentum_buy, confirm_reclaim_buy, pattern):
    return (
        "Strong Buy: RSI<dyn + MACD/Hist uptick + >50/200 + vol ok"
        if primary_buy else
        f"{pattern} + Price>50/200 (+near reclaim) + vol ok"
        if pattern_ok else
        "Dip Buy: 50-DEMA reclaim + MACD/Hist uptick + vol ok"
        if dip_buy else
        "Breakout: 20D high + MACD/Hist uptick + ADX + vol ok"
        if breakout_buy else
        "Momentum: ADX strong + RSI rising + MACD/Hist uptick + >50/200 + vol ok"
        if momentum_buy else
        "Confirm-Reclaim: 2-bar near-50 then reclaim + MACD/Hist uptick + vol ok"
    )




def apply_trailing_stop(df):
    latest = df.iloc[-1]
    atr = df['atr'].iloc[-1]
    chandelier = latest['close'] - 3 * atr
    bb_mid = df['bb_mid'].iloc[-1] if 'bb_mid' in df.columns else chandelier
    ema_fading = (
        '50_dema' in df.columns and
        df['50_dema'].iloc[-1] < df['50_dema'].shift(1).iloc[-1]
    )
    stop = max(chandelier, bb_mid) if (ema_fading and latest['close'] < bb_mid) else chandelier
    print(f"[INFO] Trailing Stop (blend): {stop:.2f}")
    return stop
