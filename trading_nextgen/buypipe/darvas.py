import numpy as np

def darvas_box_signal(df, lookback=None):
    """
    Adds:
      - darvas_box_high  (prior box high)
      - darvas_low       (prior box low)
      - darvas_box_height
      - darvas_signal    (1 if close > prior high with vol confirm)
      - darvas_breakout_pct
    """
    # Dynamic lookback from ATR_14, else default
    if lookback is None:
        if 'ATR_14' in df.columns and not df['ATR_14'].isna().all():
            lookback = int(np.clip(df['ATR_14'].iloc[-1] * 2, 10, 40))
        else:
            lookback = 20

    # Prior box bounds (shift so today compares to *previous* box)
    prior_highs = df['high'].rolling(lookback, min_periods=lookback).max().shift(1)
    prior_lows  = df['low'] .rolling(lookback, min_periods=lookback).min().shift(1)

    df['darvas_box_high'] = prior_highs
    df['darvas_low'] = prior_lows
    df['darvas_box_height'] = (df['darvas_box_high'] - df['darvas_low']).abs()

    # Volume confirmation vs 20D average
    vol_avg20 = df['volume'].rolling(20, min_periods=1).mean()
    breakout_mask = (
        df['darvas_box_high'].notna()
        & (df['close'] > df['darvas_box_high'])
        & (df['volume'] >= 1.1 *vol_avg20)   # bump to 1.2*vol_avg20 if you want stricter
    )
    df['darvas_signal'] = breakout_mask.astype(int)

    ref_high = df['darvas_box_high']
    df['darvas_breakout_pct'] = np.where(
        breakout_mask & np.isfinite(ref_high) & (ref_high > 0),
        (df['close'] - ref_high) / ref_high * 100.0,
        0.0
    )
    return df
