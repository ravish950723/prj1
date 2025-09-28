from ib_insync import Stock, util
from ib_connection import get_ib


def fetch_historical_data(symbol, bar_size='1 day', duration='3 Y'):
    ib = get_ib()
    contract = Stock(symbol, 'SMART', 'USD')
    ib.qualifyContracts(contract)

    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow='TRADES',
        useRTH=True,
        formatDate=1
    )

    df = util.df(bars)
    df.set_index('date', inplace=True)
    df['symbol'] = symbol
    return df
