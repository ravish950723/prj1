from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import numpy as np

try:
    from ib_insync import IB, Stock, ETF, Contract
except Exception:  # pragma: no cover
    IB = None
    Stock = None
    ETF = None
    Contract = None


@dataclass
class L2Snapshot:
    status: str
    symbol: str
    exchange: str
    rows_requested: int
    rows_received: int
    best_bid: float
    best_ask: float
    spread_abs: float
    spread_pct: float
    total_bid_size: float
    total_ask_size: float
    imbalance: float
    bid_vwap_5: float
    ask_vwap_5: float
    microprice: float
    slope_bid: float
    slope_ask: float
    depth_quality: str
    raw_book: Optional[List[Dict[str, Any]]] = None


def _safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _quality_from_rows(rows: int, spread_pct: float, total_size: float) -> str:
    if rows >= 8 and np.isfinite(spread_pct) and spread_pct < 0.08 and total_size > 2000:
        return "DEEP"
    if rows >= 5 and np.isfinite(spread_pct) and spread_pct < 0.15 and total_size > 500:
        return "GOOD"
    if rows >= 3:
        return "FAIR"
    return "THIN"


def _weighted_price(levels):
    px_sz = [(p, s) for p, s in levels if np.isfinite(p) and np.isfinite(s) and s > 0]
    if not px_sz:
        return np.nan
    num = sum(p * s for p, s in px_sz)
    den = sum(s for _, s in px_sz)
    return num / den if den > 0 else np.nan


def _slope(levels):
    px = [p for p, s in levels if np.isfinite(p)]
    if len(px) < 2:
        return np.nan
    return float(px[0] - px[-1])


def _make_stock_contract(symbol: str, exchange: str, currency: str = "USD"):
    # SMART often works for top data, but IBKR docs say depth requests must be direct-routed.
    # So prefer ARCA/ISLAND/BATS/NASDAQ/NYSE depending on instrument.
    if Stock is None:
        raise RuntimeError("ib_insync not installed")
    return Stock(symbol, exchange, currency)


def fetch_l2_snapshot(
    symbol: str,
    host: str = "127.0.0.1",
    port: int = 7496,
    client_id: int = 103,
    exchange: str = "ARCA",
    currency: str = "USD",
    num_rows: int = 10,
    timeout_sec: float = 2.5,
) -> Dict[str, Any]:
    """
    Real L2 via IBKR TWS/Gateway.
    Requires:
      - TWS or IB Gateway running
      - live username/session
      - market depth subscription for instrument/exchange
      - direct-routed exchange
    """
    if IB is None:
        return {
            "L2 Status": "IB_INSYNC_NOT_INSTALLED",
            "L2 Quality": np.nan,
            "L2 Rows Requested": num_rows,
            "L2 Rows Received": 0,
        }

    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id, readonly=True, timeout=timeout_sec)

        contract = _make_stock_contract(symbol=symbol, exchange=exchange, currency=currency)
        qualified = ib.qualifyContracts(contract)
        if not qualified:
            return {
                "L2 Status": "CONTRACT_NOT_QUALIFIED",
                "L2 Quality": np.nan,
                "L2 Rows Requested": num_rows,
                "L2 Rows Received": 0,
            }

        contract = qualified[0]

        # isSmartDepth=False keeps it direct-routed, aligned with IBKR docs.
        ticker = ib.reqMktDepth(contract, numRows=num_rows, isSmartDepth=False)

        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            ib.sleep(0.1)
            if len(ticker.domBids) > 0 or len(ticker.domAsks) > 0:
                break

        bids = []
        asks = []

        for row in list(ticker.domBids)[:num_rows]:
            bids.append((_safe_float(row.price), _safe_float(row.size, 0.0)))

        for row in list(ticker.domAsks)[:num_rows]:
            asks.append((_safe_float(row.price), _safe_float(row.size, 0.0)))

        ib.cancelMktDepth(contract)

        best_bid = bids[0][0] if bids else np.nan
        best_ask = asks[0][0] if asks else np.nan
        spread_abs = best_ask - best_bid if np.isfinite(best_bid) and np.isfinite(best_ask) else np.nan
        spread_pct = (spread_abs / ((best_ask + best_bid) / 2.0)) * 100.0 if np.isfinite(spread_abs) and (best_ask + best_bid) > 0 else np.nan

        total_bid_size = float(sum(s for _, s in bids))
        total_ask_size = float(sum(s for _, s in asks))
        imbalance = (
            (total_bid_size - total_ask_size) / (total_bid_size + total_ask_size)
            if (total_bid_size + total_ask_size) > 0 else np.nan
        )

        bid_vwap_5 = _weighted_price(bids[:5])
        ask_vwap_5 = _weighted_price(asks[:5])

        microprice = np.nan
        if np.isfinite(best_bid) and np.isfinite(best_ask) and bids and asks:
            bid_sz_1 = bids[0][1]
            ask_sz_1 = asks[0][1]
            denom = bid_sz_1 + ask_sz_1
            if denom > 0:
                microprice = (best_ask * bid_sz_1 + best_bid * ask_sz_1) / denom

        rows_received = max(len(bids), len(asks))
        quality = _quality_from_rows(rows_received, spread_pct, total_bid_size + total_ask_size)

        raw_book = []
        max_len = max(len(bids), len(asks))
        for i in range(max_len):
            raw_book.append(
                {
                    "level": i + 1,
                    "bid_px": bids[i][0] if i < len(bids) else np.nan,
                    "bid_sz": bids[i][1] if i < len(bids) else np.nan,
                    "ask_px": asks[i][0] if i < len(asks) else np.nan,
                    "ask_sz": asks[i][1] if i < len(asks) else np.nan,
                }
            )

        snap = L2Snapshot(
            status="OK" if rows_received > 0 else "NO_DEPTH",
            symbol=symbol,
            exchange=exchange,
            rows_requested=num_rows,
            rows_received=rows_received,
            best_bid=best_bid,
            best_ask=best_ask,
            spread_abs=spread_abs,
            spread_pct=spread_pct,
            total_bid_size=total_bid_size,
            total_ask_size=total_ask_size,
            imbalance=imbalance,
            bid_vwap_5=bid_vwap_5,
            ask_vwap_5=ask_vwap_5,
            microprice=microprice,
            slope_bid=_slope(bids),
            slope_ask=_slope(asks),
            depth_quality=quality,
            raw_book=raw_book,
        )

        return {
            "L2 Status": snap.status,
            "L2 Exchange": snap.exchange,
            "L2 Rows Requested": snap.rows_requested,
            "L2 Rows Received": snap.rows_received,
            "L2 Best Bid": snap.best_bid,
            "L2 Best Ask": snap.best_ask,
            "BID_ASK_SPREAD_PCT": snap.spread_pct,
            "L2 Spread Abs": snap.spread_abs,
            "L2 Total Bid Size": snap.total_bid_size,
            "L2 Total Ask Size": snap.total_ask_size,
            "L2 Imbalance": snap.imbalance,
            "L2 Bid VWAP 5": snap.bid_vwap_5,
            "L2 Ask VWAP 5": snap.ask_vwap_5,
            "L2 Microprice": snap.microprice,
            "L2 Bid Slope": snap.slope_bid,
            "L2 Ask Slope": snap.slope_ask,
            "L2 Quality": snap.depth_quality,
            "L2 Book": snap.raw_book,
        }

    except Exception as e:
        return {
            "L2 Status": f"ERROR: {type(e).__name__}",
            "L2 Quality": np.nan,
            "L2 Rows Requested": num_rows,
            "L2 Rows Received": 0,
            "L2 Error Detail": str(e),
        }
    finally:
        try:
            if ib.isConnected():
                ib.disconnect()
        except Exception:
            pass