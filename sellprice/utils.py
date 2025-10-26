from ib_insync import Stock, MarketOrder
from ib_connection import get_ib
from data_fetcher import fetch_historical_data
from config import symbols, symbol_to_sector, sector_map
import csv
from pathlib import Path

# Global signal log file
log_file = "trades_log.csv"

def _ensure_log_header(path, new_fieldnames):
    """
    If the existing CSV header is missing columns, rewrite the file with the new header
    and preserve existing rows (filling missing columns with blanks).
    """
    import csv, os, io
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        old_fieldnames = reader.fieldnames or []
        if not set(new_fieldnames).issuperset(set(old_fieldnames)):
            # Don't remove any old columns
            return
        # If headers already match exactly (order can differ), do nothing
        if set(old_fieldnames) == set(new_fieldnames):
            return
        rows = list(reader)

    # Rewrite with new header and updated rows
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        for r in rows:
            for k in new_fieldnames:
                r.setdefault(k, None)
            writer.writerow(r)
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



def log_signal(symbol, date_str, message, confidence, signal_type="WATCH", condition="", ATRpct=None, extras=None, **kwargs):
    """
    Append a signal row into signals_raw.csv.
    - 'extras' is a dict of additional columns to write, e.g. {"Pattern detected": "...", "Breakout": "..."}.
    - Always includes 'Pattern detected' and 'Breakout' columns in the CSV header.
    """
    out = {
        "Symbol": symbol,
        "Date": date_str,
        "Signal": message,
        "Confidence": confidence,
        "Type": signal_type,
        "Condition": condition,
        "ATRpct": ATRpct,
        # Ensure these two columns always exist:
        "Pattern detected": None,
        "Breakout": None,
    }
    # Merge direct kwargs (e.g., SizePct) into the row
    if kwargs:
        for k, v in kwargs.items():
            out[k] = v

    # Merge extras (if any) into the row
    if extras and isinstance(extras, dict):
        for k, v in extras.items():
            out[k] = v

    # Establish field order. Put the standard fields first, then any other extras (stable order)
    base_fields = [
        "Symbol", "Date", "Signal", "Confidence", "Type", "Condition", "ATRpct",
        "Pattern detected", "Breakout"
    ]
    extra_fields = [k for k in out.keys() if k not in base_fields]
    fieldnames = base_fields + extra_fields
    # Evolve header if needed
    csv_path = Path("signals_raw.csv")
    _ensure_log_header(str(csv_path), fieldnames)
# or your existing path for the raw signals CSV if different
    file_exists = csv_path.exists()

    # If file exists, we won't rewrite the header line, but we can still write rows
    # with a superset of columns; pandas will pick them up on future reads.
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(out)

    # Also keep an in-memory copy for post-analysis
    try:
        signal_storage.append(out)
    except Exception as e:
        print(f"[WARN] Could not append to signal_storage: {e}")


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
        cols = ["Symbol","Date","Signal","Confidence","Type","Condition","ATRpct","SizePct","Pattern detected","Breakout"]
        pd.DataFrame(columns=cols).to_excel(xlsx_path, index=False)
        print(f"💾 Saved (empty): {xlsx_path}")
        return xlsx_path

    df_raw = pd.DataFrame(signal_storage)
    # Stable column order
    cols = ["Symbol","Date","Signal","Confidence","Type","Condition","ATRpct","SizePct","Pattern detected","Breakout"]
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
