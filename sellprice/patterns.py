import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def detect_trendline_break(df):
    swing_lows = df[df['swing_low']]
    if len(swing_lows) >= 3:
        x_scaled = np.array([(i - swing_lows.index[0]).days for i in swing_lows.index]).reshape(-1, 1)
        y = swing_lows['low'].values
        model = make_pipeline(PolynomialFeatures(1), RANSACRegressor())
        model.fit(x_scaled, y)
        latest_x = np.array([[(df.index[-1] - swing_lows.index[0]).days]])
        predicted = model.predict(latest_x)[0]
        df['trendline_break'] = df['low'].iloc[-1] < predicted
    else:
        df['trendline_break'] = False
    return df


def detect_chart_pattern(df):
    print("[INFO] Detecting chart patterns...")
    bottoms = df[df['swing_low']]
    pattern = None

    if len(bottoms) >= 3:
        lows = bottoms['low'].tail(3).values
        if np.abs(lows[0] - lows[1]) < 0.02 * lows[0] and np.abs(lows[1] - lows[2]) < 0.02 * lows[1]:
            pattern = 'Triple Bottom'
        elif np.abs(lows[0] - lows[1]) < 0.02 * lows[0]:
            pattern = 'Double Bottom'
        elif len(bottoms) >= 5:
            last5 = bottoms['low'].tail(5).values
            if last5[1] < last5[0] and last5[3] < last5[2] and abs(last5[1] - last5[3]) < 0.03 * last5[1]:
                pattern = 'Inverse Head and Shoulders'

    highs = df[df['swing_high']]['high'].tail(5).values
    if len(highs) == 5 and np.all(np.diff(highs) > 0):
        pattern = pattern or 'Ascending Triangle'

    if len(df) > 30:
        recent_close = df['close'].tail(30).values
        if recent_close[0] > recent_close[15] and recent_close[15] < recent_close[-1] and recent_close[-1] > \
                recent_close[0]:
            pattern = pattern or 'Cup and Handle'

    df['pattern'] = pattern
    print(f"[INFO] Pattern detected: {df['pattern'].iloc[-1]}")
    return df
