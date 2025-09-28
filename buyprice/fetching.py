
import os
import pandas as pd
from ib_insync import Stock, IB
from config import CACHE_DIR, IB_CLIENT_ID

from config import CACHE_DIR, IB_CLIENT_ID
os.makedirs(CACHE_DIR, exist_ok=True)


# Exchanges we’ll try in order. ARCA is critical for ETFs like IRBO.
_EXCHANGE_TRIES = [
    dict(exchange="SMART", primaryExchange="ARCA"),
    dict(exchange="ARCA"),
    dict(exchange="SMART"),                 # plain SMART (some equities)
    dict(exchange="ISLAND", primaryExchange="NASDAQ"),  # alt route for some tech
]

def _resolve_contract(ib: IB, symbol: str):
    # Try multiple exchange combos until one qualifies
    for spec in _EXCHANGE_TRIES:
        c = Stock(symbol, currency="USD", **spec)
        qualified = ib.qualifyContracts(c)
        if qualified:
            return qualified[0]
    return None

def fetch_data_cached(symbol: str, duration: str, bar_size: str, refresh: bool = False) -> pd.DataFrame:
    cache_file = os.path.join(CACHE_DIR, f"{symbol}_{duration}_{bar_size}.csv")

    if not refresh and os.path.exists(cache_file):
        print(f"[{symbol}] 🔁 Using cached data for {duration} - {bar_size}")
        df = pd.read_csv(cache_file, parse_dates=["date"]).sort_values("date")
        return df

    print(f"[{symbol}] 🔄 Fetching fresh data for {duration} at bar size {bar_size}...")
    ib = IB()
    try:
        # Connect (adjust port if you use live TWS: 7496)
        ib.connect('127.0.0.1', 7497, clientId=IB_CLIENT_ID)
        # For historical calls this isn’t required, but harmless if you lack live permissions:
        ib.reqMarketDataType(3)  # delayed-frozen ok

        contract = _resolve_contract(ib, symbol)
        if not contract:
            print(f"⚠️ {symbol}: Could not qualify contract on IBKR (tried SMART/ARCA/ISLAND). Skipping.")
            return pd.DataFrame()

        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',            # now
            durationStr=duration,      # e.g. "3 Y"
            barSizeSetting=bar_size,   # e.g. "1 day"
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )

        if not bars:
            # Last-ditch retry: allow outside RTH (some ETFs have thin RTH history on certain days)
            bars = ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=False,
                formatDate=1
            )

        if not bars:
            print(f"⚠️ {symbol}: No historical bars returned (even with useRTH=False). Skipping.")
            return pd.DataFrame()

        df = pd.DataFrame(bars)
        df.columns = [c.lower() for c in df.columns]
        if 'date' not in df.columns:
            print(f"⚠️ {symbol}: Missing 'date' column in IBKR response. Skipping.")
            return pd.DataFrame()

        df['date'] = pd.to_datetime(df['date'])
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']].sort_values('date').reset_index(drop=True)
        df.to_csv(cache_file, index=False)
        return df

    except Exception as e:
        print(f"❌ Error fetching {symbol}: {e}")
        return pd.DataFrame()
    finally:
        ib.disconnect()







