import numpy as np
import pandas as pd
import numpy as np
import pandas as pd

def calculate_indicators(df):
    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']

    # Exponential Moving Averages
    df['200_dema'] = df['close'].ewm(span=200, adjust=False).mean()
    df['50_dema'] = df['close'].ewm(span=50, adjust=False).mean()

    # % Above 200 DEMA
    df['%_above_dema'] = 100 * (df['close'] - df['200_dema']) / df['200_dema']

    # Swing High / Low
    df['swing_high'] = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
    df['swing_low'] = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))

    # RSI (14)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # OBV (On Balance Volume)
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

    # ATR (14)
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    high_low = df['high'] - df['low']
    tr = np.maximum.reduce([high_close, low_close, high_low])
    df['atr'] = pd.Series(tr, index=df.index).rolling(14).mean()

    # EMA Trend
    df['ema_trend'] = df['50_dema'] > df['200_dema']

    # VWAP
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

    # ADX (simplified using ATR and rolling mean)
    df['adx'] = 100 * df['atr'] / df['close'].rolling(14).mean()

    # EMA Crossover Detection
    df['ema_crossover'] = (df['50_dema'] > df['200_dema']) & (df['50_dema'].shift(1) <= df['200_dema'].shift(1))

    # Candlestick patterns
    df['doji'] = (abs(df['open'] - df['close']) / (df['high'] - df['low'] + 1e-5)) < 0.1
    df['engulfing'] = (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1))

    return df



# === Market Stage detection (Accumulation / Mark-Up / Distribution / Mark-Down) ===


def compute_market_stage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Labels the current regime into one of:
    Accumulation, Mark-Up, Distribution, Mark-Down (else Neutral/Transition).
    Adds columns: 'market_stage' (str), 'stage_conf' (0-100).
    """
    if len(df) < 60:
        df['market_stage'] = "Neutral/Transition"
        df['stage_conf'] = 0.0
        return df

    # Helper features
    close = df['close']
    ema50 = df.get('50_dema', close.ewm(span=50, adjust=False).mean())
    ema200 = df.get('200_dema', close.ewm(span=200, adjust=False).mean())
    adx = df.get('adx', pd.Series(25.0, index=df.index))
    rsi = df.get('rsi', pd.Series(50.0, index=df.index))
    macd = df.get('macd', pd.Series(0.0, index=df.index))
    macd_sig = df.get('macd_signal', pd.Series(0.0, index=df.index))
    macd_hist = (macd - macd_sig)
    pct_above_200 = df.get('%_above_dema', 100.0 * (close - ema200) / (ema200.replace(0, np.nan)))

    # Volatility & expansion
    atr = df.get('atr')
    atr_pct = (atr / close).clip(lower=0, upper=1.0) if atr is not None else pd.Series(0.02, index=df.index)
    vol_expanding = atr_pct > atr_pct.rolling(20).mean()

    # Trend quality and direction
    above_50 = close > ema50
    above_200 = close > ema200
    ema50_slope = ema50.diff(5)
    ema200_slope = ema200.diff(10)
    higher_highs = (df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))
    lower_lows = (df['low'] < df['low'].shift(1)) & (df['low'].shift(1) < df['low'].shift(2))

    # OBV divergence proxy: OBV not confirming highs
    obv = df.get('obv', pd.Series(0.0, index=df.index))
    obv_rolling_max = obv.rolling(50).max()
    obv_div = (obv < 0.98 * obv_rolling_max) & (close >= close.rolling(50).max() * 0.995)

    # Gate values (tuned conservatively for ETFs vs single stocks if desired)
    adx_trend = adx > 20
    adx_weak = adx < 18

    # --- Scores (0..1) for each regime
    # Accumulation: below/near 200, weak ADX, RSI mid, vol contracting/flat, ema slopes ~flat/turning up
    score_acc = (
        (pct_above_200.between(-12, 5)).astype(int) * 0.35 +
        (adx_weak).astype(int) * 0.20 +
        (rsi.between(40, 56)).astype(int) * 0.15 +
        (ema200_slope >= -0.2 * ema200.abs().replace(0, 1e-9)).astype(int) * 0.15 +
        (~vol_expanding).astype(int) * 0.15
    )

    # Mark-Up: above 50 & 200, ema slopes rising, MACD>signal or hist rising, ADX trending
    score_up = (
        (above_50 & above_200).astype(int) * 0.35 +
        (ema50_slope > 0).astype(int) * 0.15 +
        (ema200_slope > 0).astype(int) * 0.15 +
        ((macd > macd_sig) | (macd_hist > macd_hist.shift(1))).astype(int) * 0.20 +
        (adx_trend).astype(int) * 0.15
    )

    # Distribution: extended above 200, RSI>65 then cooling, MACD cross/down hist, OBV divergence, ADX rolling over
    rsi_cooling = (rsi.shift(1) > 65) & (rsi < rsi.shift(1))
    adx_rolling_over = (adx < adx.shift(5))
    score_dist = (
        (pct_above_200 > 8).astype(int) * 0.30 +
        (rsi_cooling).astype(int) * 0.20 +
        ((macd < macd_sig) | (macd_hist < macd_hist.shift(1))).astype(int) * 0.20 +
        (obv_div).astype(int) * 0.20 +
        (adx_rolling_over).astype(int) * 0.10
    )

    # Mark-Down: below 50 & 200, ema slopes down, MACD<sig & <0, ADX trending, vol expanding, lower lows
    score_down = (
        ((~above_50) & (~above_200)).astype(int) * 0.35 +
        (ema50_slope < 0).astype(int) * 0.15 +
        (ema200_slope < 0).astype(int) * 0.15 +
        ((macd < macd_sig) & (macd < 0)).astype(int) * 0.20 +
        ((adx > 20) & vol_expanding).astype(int) * 0.15
    )

    # Pick label by max score; compute confidence
    stage_scores = pd.concat(
        [score_acc.rename('Accumulation'),
         score_up.rename('Mark-Up'),
         score_dist.rename('Distribution'),
         score_down.rename('Mark-Down')],
        axis=1
    ).fillna(0.0)

    stage = stage_scores.idxmax(axis=1)
    conf = (stage_scores.max(axis=1) * 100).clip(0, 100)

    # If none is decisive, mark Neutral/Transition
    weak = stage_scores.max(axis=1) < 0.45
    stage = stage.mask(weak, "Neutral/Transition")
    conf = conf.mask(weak, 100 * stage_scores.max(axis=1))

    df['market_stage'] = stage
    df['stage_conf'] = conf.round(1)
    return df
