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
