def score_institutional_investor(df):
    if not {'volume', 'close', 'OBV'}.issubset(df.columns):
        return 0.5  # fallback

    try:
        # Relative volume spike vs trailing average
        volume_20 = df['volume'].rolling(20).mean()
        volume_50 = df['volume'].rolling(50).mean()
        volume_score = (volume_20 / volume_50).clip(0.5, 2.0)


        # OBV slope as proxy for accumulation
        obv_slope = df['OBV'].diff().rolling(5).mean()

        # Price trend confirmation
        price_change = df['close'].pct_change().rolling(5).mean()

        # Combine all
        institutional_score = (
            0.5 * volume_score +
            0.3 * obv_slope.rank(pct=True) +
            0.2 * price_change.rank(pct=True)
        ).iloc[-1]

        return round(float(institutional_score), 2)

    except Exception as e:
        print(f"⚠️ Error scoring institutional investor: {e}")
        return 0.5
