import os

# === IB / caching ===
CACHE_DIR = os.getenv("CACHE_DIR", "cache")
IB_HOST = os.getenv("IB_HOST", "127.0.0.1")
IB_PORT = int(os.getenv("IB_PORT", "7497"))
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "103"))  # change via env if needed

# === Labels / thresholds ===
STRONG_BUY_LABEL = {
    "forward_window": 30,
    "min_max_gain_pct": 15.0,
    "max_drawdown_pct": 10.0,
    "min_adx": 20.0,
    "min_volume_weight": 1.0,
}

UPWARD_SIGNAL_WEIGHTS = {
    "smc": 0.05,
    "mean_rev": 0.03,
    "bullish_engulfing": 0.03,
    "hammer": 0.03,
    "trend_strength": 0.01,
    "buy_threshold": 0.08,
    "min_signals": 2,
}


symbols = [
    # Bitcoin-related stocks & ETFs
    "STCE", "COIN", "MSTR", "RIOT", "MARA", "GBTC", "BITO",
    "WGMI", "BTBT", "BITB", "CIFR",
    # Added Bitcoin miners & ETFs
    "BITW", "HUT", "CLSK",  "CORZ",
    # Newly added ETF
    "CRPT"
]


symbol_to_sector = {
    # Bitcoin-related stocks
    "COIN": "Bitcoin",
    "MSTR": "Bitcoin",
    "RIOT": "Bitcoin",
    "MARA": "Bitcoin",
    "BTBT": "Bitcoin",
    "CIFR": "Bitcoin",
    "HUT": "Bitcoin",
    "CLSK": "Bitcoin",
    "HVBT": "Bitcoin",
    "CORZ": "Bitcoin",

    # Bitcoin ETFs
    "STCE": "BitcoinETF",
    "GBTC": "BitcoinETF",
    "BITO": "BitcoinETF",
    "WGMI": "BitcoinETF",
    "BITB": "BitcoinETF",
    "BITW": "BitcoinETF",
    "CRPT": "BitcoinETF"
}


sector_etfs = {
    "Bitcoin": "STCE",         # STCE as the proxy ETF
    "BitcoinETF": "STCE"
}

# NEW: normalize your custom sectors to the defaults above
sector_map = {
    'Bitcoin': 'finance',  # or 'commodity' if you prefer miner-like behavior
    'BitcoinETF': 'etf'
}