def evaluate_backtest_accuracy(symbol, df, buy_price, gain_thresh=0.04, use_close=False):
    try:
        df_tail = df.tail(90).reset_index(drop=True)
        ref = df_tail['close'] if use_close else df_tail['high']
        df_tail['gain_pct'] = (ref - buy_price) / buy_price
        hit = (df_tail['gain_pct'] > gain_thresh).any()
        max_gain = float(df_tail['gain_pct'].max() * 100.0)
        days_to_peak = int(df_tail['gain_pct'].idxmax())
        return hit, max_gain, days_to_peak
    except Exception as e:
        print(f"[{symbol}] Backtest error: {e}")
        return False, 0.0, -1
