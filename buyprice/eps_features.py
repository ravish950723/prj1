# eps_features.py
#
# Quarterly EPS fetch + growth flags (2, 3, 4 quarters)
# Uses ib_insync like the rest of the project. If fundamentals
# are unavailable or parsing fails, flags default to None.

from typing import Optional, Dict
import pandas as pd

try:
    from ib_insync import IB, Stock
except Exception as _e:  # ib_insync not installed or import failed
    IB = None
    Stock = None

from config import IB_HOST, IB_PORT, IB_CLIENT_ID


def fetch_quarterly_eps(symbol: str) -> Optional[pd.DataFrame]:
    """
    Fetch quarterly EPS for the given symbol from IBKR fundamentals.

    Returns a DataFrame with columns:
        ['reportDate', 'eps']  (latest first)
    or None on failure.

    Implementation notes:
    - Uses IBKR 'ReportsFinStatements' snapshot.
    - Tries to find a column containing EPS per share.
    """
    if IB is None or Stock is None:
        print("⚠️ ib_insync not available; EPS disabled.")
        return None

    ib = IB()
    try:
        # Use a different clientId than your main price-fetch flow to avoid clashes
        ib.connect(IB_HOST, IB_PORT, clientId=int(IB_CLIENT_ID) + 200, readonly=True, timeout=10.0)

        contract = Stock(symbol, "SMART", "USD")
        ib.qualifyContracts(contract)

        # Fundamental snapshot: financial statements
        data = ib.reqFundamentalData(contract, "ReportsFinStatements")
        if not data:
            print(f"⚠️ No fundamental data for {symbol}")
            return None

        # IBKR returns XML; pandas can parse many of these structures
        try:
            df = pd.read_xml(data)
        except Exception as e:
            print(f"⚠️ Failed to parse fundamentals XML for {symbol}: {e}")
            return None

        # Heuristic: look for a column containing EPS values
        col_candidates = [c for c in df.columns if "earn" in c.lower() and "share" in c.lower()]
        if not col_candidates:
            print(f"⚠️ No EPS column found in fundamentals for {symbol}")
            return None

        eps_col = col_candidates[0]

        # Find a date/report column
        date_col = None
        for c in df.columns:
            cl = c.lower()
            if "report" in cl and "date" in cl:
                date_col = c
                break
            if cl == "date":
                date_col = c

        if date_col is None:
            print(f"⚠️ No report date column found for {symbol}")
            return None

        eps_df = df[[date_col, eps_col]].dropna()
        eps_df.columns = ["reportDate", "eps"]
        eps_df["eps"] = pd.to_numeric(eps_df["eps"], errors="coerce")
        eps_df = eps_df.dropna(subset=["eps"])

        if eps_df.empty:
            return None

        # Sort latest first
        eps_df = eps_df.sort_values("reportDate", ascending=False).reset_index(drop=True)
        return eps_df

    except Exception as e:
        print(f"⚠️ EPS fetch error for {symbol}: {e}")
        return None
    finally:
        try:
            if ib.isConnected():
                ib.disconnect()
        except Exception:
            pass


def eps_growth_flags(eps_df: Optional[pd.DataFrame]) -> Dict[str, Optional[bool]]:
    """
    Given an EPS DataFrame (latest first), compute growth flags.

    Returns a dict with keys:
        - 'EPS Increase 2Q'
        - 'EPS Increase 3Q'
        - 'EPS Increase 4Q'

    Each is True/False, or None if not enough data.
    """
    result: Dict[str, Optional[bool]] = {
        "EPS Increase 2Q": None,
        "EPS Increase 3Q": None,
        "EPS Increase 4Q": None,
    }

    if eps_df is None or len(eps_df) < 2:
        return result

    eps = eps_df["eps"].tolist()

    # Need at least N quarters for an N-quarter streak
    if len(eps) >= 2:
        result["EPS Increase 2Q"] = bool(eps[0] > eps[1])

    if len(eps) >= 3:
        result["EPS Increase 3Q"] = bool(eps[0] > eps[1] > eps[2])

    if len(eps) >= 4:
        result["EPS Increase 4Q"] = bool(eps[0] > eps[1] > eps[2] > eps[3])

    return result
