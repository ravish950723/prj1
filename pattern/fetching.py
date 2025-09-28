import os
import pandas as pd
from ib_insync import IB, Stock, util
from config import CACHE_DIR, IB_HOST, IB_PORT, IB_CLIENT_ID

os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_name(symbol: str, duration: str, bar_size: str) -> str:
    duration_clean = duration.replace(" ", "")
    bar_size_clean = bar_size.replace(" ", "")
    return os.path.join(CACHE_DIR, f"{symbol}_{duration_clean}_{bar_size_clean}.csv")


def fetch_data_cached(
    symbol: str,
    duration: str = "3 Y",
    bar_size: str = "1 day",
    refresh: bool = False,
    what_to_show: str = "ADJUSTED_LAST",
    use_rth: bool = True,
) -> pd.DataFrame:
    """Fetch historical data with simple on-disk caching.

    Uses a short-lived IB connection per call to avoid persistent clientId collisions.
    To speed up repeated runs, rely on the cache (default) or set refresh=True.
    """
    cache_file = _cache_name(symbol, duration, bar_size)

    if not refresh and os.path.exists(cache_file):
        print(f"[{symbol}] 🔁 Using cached: {os.path.basename(cache_file)}")
        return pd.read_csv(cache_file, parse_dates=["date"])  # type: ignore

    print(f"[{symbol}] 🔄 Fetching {duration} @ {bar_size} ({what_to_show})…")
    ib = IB()
    try:
        ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, readonly=True)
        contract = Stock(symbol, "SMART", "USD")
        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=use_rth,
            formatDate=1,
        )
        df = util.df(bars)
        if df.empty:
            raise RuntimeError(f"No historical data returned for {symbol}")

        # Normalize & validate
        df.columns = [str(c).lower() for c in df.columns]
        required = ["date", "open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            # Sometimes volume can be missing depending on whatToShow; ensure presence
            for c in missing:
                if c == "volume":
                    df[c] = 0
                else:
                    raise ValueError(f"[{symbol}] Missing required column: {c}")

        df.to_csv(cache_file, index=False)
        return df
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass