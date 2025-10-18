from ib_insync import Stock, MarketOrder
from ib_connection import get_ib
from data_fetcher import fetch_historical_data
from config import symbols, symbol_to_sector, sector_map


# Global signal log file
log_file = "trades_log.csv"
signal_storage = []  # In-memory storage to avoid duplicates in runtime


def get_sector_info(symbol):
    sector_defaults = {
        'tech':      {'rsi_base': 30, 'vol_thresh': 0.03},
        'commodity': {'rsi_base': 35, 'vol_thresh': 0.02},
        'etf':       {'rsi_base': 32, 'vol_thresh': 0.025},
        'finance':   {'rsi_base': 28, 'vol_thresh': 0.035},
        'energy':    {'rsi_base': 33, 'vol_thresh': 0.03},
        'default':   {'rsi_base': 30, 'vol_thresh': 0.03},
    }

    # 1) your configured label (e.g., "Bitcoin", "BitcoinETF")
    custom_label = symbol_to_sector.get(symbol, "default")
    sector_type = sector_map.get(custom_label, custom_label)

    # 2) normalize to a known bucket used by thresholds
    normalized = sector_map.get(custom_label, custom_label)

    # 3) pull thresholds from normalized bucket
    cfg = sector_defaults.get(normalized, sector_defaults['default'])
    rsi_base = cfg['rsi_base']
    vol_threshold = cfg['vol_thresh']

    # 4) compute weight (volatility-based), with graceful fallback
    weight = 0.85
    try:
        ib = get_ib()
        contract = Stock(symbol, 'SMART', 'USD')
        ib.qualifyContracts(contract)
        from indicators import calculate_indicators
        df = fetch_historical_data(symbol, bar_size='1 day', duration='3 Y')
        df = calculate_indicators(df)
        atr = df['atr'].iloc[-1]
        close = float(df['close'].iloc[-1])
        normalized_atr = atr / df['close'].iloc[-1]
        volatility_factor = max(0.6, 1 - normalized_atr)
        weight = round(min(1.0, volatility_factor), 2)
    except Exception as e:
        print(f"[WARNING] Fallback for {symbol}: {e}")

    # Keep backward-compatible tuple for signals.py
    # return normalized, rsi_base, vol_threshold, weight
    return sector_type, rsi_base, vol_threshold, weight


def _normalize_signal_type(t: str) -> str:
    if not t:
        return "WATCH"
    t = str(t).strip().upper()
    # map common variants → canonical
    aliases = {
        "SELL SIGN2": "SELL",
        "SELL SIgn2".upper(): "SELL",
        "SELLING": "SELL",
        "SHORT SELLING1": "SHORT",
        "SHORT-SELL": "SHORT",
        "SHORTING": "SHORT",
        "BUYING": "BUY",
    }
    return aliases.get(t, t if t in {"BUY", "SELL", "SHORT", "WATCH"} else "WATCH")


import csv
import os

# ... keep your other imports / globals ...

def log_signal(symbol, date_str, reason, confidence, signal_type="WATCH", condition=None, **extras):
    # Normalize
    signal_type = _normalize_signal_type(signal_type)

    # Compose entry (allow extras like ATRpct, SizePct)
    entry = {
        "Symbol": symbol,
        "Date": date_str,
        "Signal": reason,
        "Confidence": confidence,
        "Type": signal_type,
        "Condition": condition or "-",
    }
    entry.update(extras)

    # De-dupe: same Symbol/Date/Signal/Type/Condition
    global signal_storage
    if not any(
        e.get("Symbol") == entry["Symbol"] and
        e.get("Date") == entry["Date"] and
        e.get("Signal") == entry["Signal"] and
        e.get("Type") == entry["Type"] and
        (e.get("Condition") or "-") == entry["Condition"]
        for e in signal_storage
    ):
        signal_storage.append(entry)

        write_header = not os.path.exists(log_file)
        with open(log_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "Symbol", "Date", "Signal", "Confidence", "Type",
                    "Condition", "ATRpct", "SizePct"
                ]
            )
            if write_header:
                writer.writeheader()
            out = dict(entry)
            out["ATRpct"] = entry.get("ATRpct")
            out["SizePct"] = entry.get("SizePct")
            writer.writerow(out)
        print(f"[LOG] {symbol} {signal_type}: {reason} ({confidence}%)")



def print_signal_summary(min_confidence=0, signal_type=None):
    filtered = [
        sig for sig in signal_storage
        if sig["Confidence"] >= min_confidence and
           (signal_type is None or sig["Type"].upper() == signal_type.upper())
    ]
    if not filtered:
        print("📭 No signals to summarize.")
        return []        # <-- return an empty list instead of None

    print("📊 Signal Summary:")
    headers = ["Symbol", "Date", "Signal", "Confidence", "Type", "Condition"]
    table = [[
        sig["Symbol"],
        sig["Date"],
        sig["Signal"],
        f"{sig['Confidence']}%",
        sig["Type"],
        sig["Condition"] or "-"
    ] for sig in filtered]

    # print(tabulate(table, headers=headers, tablefmt="fancy_grid"))
    return filtered  # <-- return the data


def log_confidence_score(score, reason):
    print(f"[CONFIDENCE] {reason}: {score:.1f}%")


# final/sellprice/core.py

def _passes_rr_filter(df, latest_row, min_rr: float = 2.0):
    """
    Simple risk:reward gate.
    Expects the dataframe to have computed stop/target hints, or falls back to ATR-based stops.
    - Risk  = entry - stop
    - Reward = target - entry
    Returns True when Reward/Risk >= min_rr and inputs make sense.
    """
    try:
        price = float(latest_row["close"])

        # Try optional columns first; fall back to ATR-based stop/target if unavailable
        stop = float(latest_row.get("stop_price", price - 2.0 * float(df["atr"].iloc[-1])))
        target = float(latest_row.get("target_price", price + 4.0 * float(df["atr"].iloc[-1])))

        risk = price - stop
        reward = target - price

        if risk <= 0 or reward <= 0:
            return False
        return (reward / risk) >= min_rr
    except Exception:
        # If we can't compute a sensible R:R, fail safe (block)
        return False


# === Excel export helpers ===
def export_signals_to_excel(xlsx_path: str = "signals_raw.xlsx") -> str:
    """
    Dump the in-memory `signal_storage` (raw, unfiltered) to an Excel workbook.
    Returns the path written.
    """
    import pandas as pd
    if not signal_storage:
        # still write an empty file with headers so downstream automations don't break
        cols = ["Symbol","Date","Signal","Confidence","Type","Condition","ATRpct","SizePct"]
        pd.DataFrame(columns=cols).to_excel(xlsx_path, index=False)
        print(f"💾 Saved (empty): {xlsx_path}")
        return xlsx_path

    df_raw = pd.DataFrame(signal_storage)
    # Stable column order
    cols = ["Symbol","Date","Signal","Confidence","Type","Condition","ATRpct","SizePct"]
    for c in cols:
        if c not in df_raw.columns:
            df_raw[c] = None
    df_raw = df_raw[cols]

    # Excel output (requires openpyxl)
    try:
        df_raw.to_excel(xlsx_path, index=False)
        print(f"💾 Saved: {xlsx_path}")
    except Exception as e:
        print(f"⚠️ Could not write {xlsx_path}: {e}")
    return xlsx_path
