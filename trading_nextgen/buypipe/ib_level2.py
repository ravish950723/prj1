from __future__ import annotations

import math
import os
import time
from .ib_safe import SafeIB
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class IBLevel2Config:
    host: str = os.getenv("IB_HOST", "127.0.0.1")
    port: int = int(os.getenv("IB_PORT", "7496"))
    client_id: int = int(os.getenv("IB_CLIENT_ID_L2", os.getenv("IB_CLIENT_ID", "103")))
    exchange: str = os.getenv("IB_L2_EXCHANGE", "SMART")
    currency: str = os.getenv("IB_CURRENCY", "USD")
    primary_exchange: str = os.getenv("IB_PRIMARY_EXCHANGE", "")
    sec_type_default: str = os.getenv("IB_SEC_TYPE_DEFAULT", "STK")
    depth_rows: int = int(os.getenv("IB_L2_DEPTH_ROWS", "10"))
    snapshot_wait_seconds: float = float(os.getenv("IB_L2_WAIT_SECONDS", "2.5"))
    connect_timeout_seconds: float = float(os.getenv("IB_CONNECT_TIMEOUT_SECONDS", "4.0"))
    enabled: bool = os.getenv("BUYPIPE_USE_LEVEL2", "1") != "0"


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _guess_sec_type(symbol: str) -> str:
    symbol = (symbol or "").upper().strip()
    etf_like = {
        "QQQ", "SPY", "IWM", "BUG", "CPER", "GLD", "SLV", "IBIT", "BITO", "QTUM", "ARKK", "ARKQ",
        "ARKX", "HACK", "WGMI", "PPA", "MISL", "DXYZ", "BUG", "CRPT"
    }
    return "ETF" if symbol in etf_like else "STK"


def _normalize_contract(ib: Any, symbol: str, cfg: IBLevel2Config) -> Any:
    from ib_insync import Contract, Stock  # type: ignore

    sec_type = _guess_sec_type(symbol)
    if sec_type in {"STK", "ETF"}:
        contract = Stock(symbol=symbol, exchange=cfg.exchange, currency=cfg.currency)
        if cfg.primary_exchange:
            contract.primaryExchange = cfg.primary_exchange
    else:
        contract = Contract(symbol=symbol, secType=cfg.sec_type_default, exchange=cfg.exchange, currency=cfg.currency)

    qualified = ib.qualifyContracts(contract)
    if qualified:
        return qualified[0]
    return contract


def _read_depth_rows(ticker: Any) -> Dict[str, list[dict[str, float]]]:
    bids: list[dict[str, float]] = []
    asks: list[dict[str, float]] = []

    for row in getattr(ticker, "domBids", []) or []:
        bids.append({
            "price": _safe_float(getattr(row, "price", None)),
            "size": _safe_float(getattr(row, "size", None), 0.0),
        })
    for row in getattr(ticker, "domAsks", []) or []:
        asks.append({
            "price": _safe_float(getattr(row, "price", None)),
            "size": _safe_float(getattr(row, "size", None), 0.0),
        })

    bids = [x for x in bids if math.isfinite(x["price"]) and x["price"] > 0]
    asks = [x for x in asks if math.isfinite(x["price"]) and x["price"] > 0]
    bids.sort(key=lambda x: x["price"], reverse=True)
    asks.sort(key=lambda x: x["price"])
    return {"bids": bids, "asks": asks}


def _compute_metrics(symbol: str, depth: Dict[str, list[dict[str, float]]]) -> Dict[str, Any]:
    bids = depth["bids"]
    asks = depth["asks"]

    best_bid = bids[0]["price"] if bids else float("nan")
    best_ask = asks[0]["price"] if asks else float("nan")
    best_bid_size = bids[0]["size"] if bids else 0.0
    best_ask_size = asks[0]["size"] if asks else 0.0

    mid = (best_bid + best_ask) / 2.0 if math.isfinite(best_bid) and math.isfinite(best_ask) else float("nan")
    spread = best_ask - best_bid if math.isfinite(best_bid) and math.isfinite(best_ask) else float("nan")
    spread_pct = (spread / mid) * 100.0 if math.isfinite(spread) and math.isfinite(mid) and mid > 0 else float("nan")

    bid_depth = sum(x["size"] for x in bids)
    ask_depth = sum(x["size"] for x in asks)
    top5_bid_depth = sum(x["size"] for x in bids[:5])
    top5_ask_depth = sum(x["size"] for x in asks[:5])

    total_depth = bid_depth + ask_depth
    top5_total = top5_bid_depth + top5_ask_depth

    depth_imbalance = ((bid_depth - ask_depth) / total_depth) if total_depth > 0 else 0.0
    top5_imbalance = ((top5_bid_depth - top5_ask_depth) / top5_total) if top5_total > 0 else 0.0

    microprice = (
        ((best_ask * best_bid_size) + (best_bid * best_ask_size)) / (best_bid_size + best_ask_size)
        if (best_bid_size + best_ask_size) > 0 and math.isfinite(best_bid) and math.isfinite(best_ask)
        else mid
    )

    weighted_mid = mid
    if bids and asks:
        bid_notional = sum(x["price"] * x["size"] for x in bids[:5])
        ask_notional = sum(x["price"] * x["size"] for x in asks[:5])
        weighted_mid = (bid_notional + ask_notional) / top5_total if top5_total > 0 else mid

    liquidity_score = float("nan")
    if math.isfinite(mid) and mid > 0:
        depth_component = _clip(math.log10(top5_total + 1.0) / 4.0, 0.0, 1.0)
        spread_component = 1.0 - _clip((spread_pct if math.isfinite(spread_pct) else 5.0) / 2.0, 0.0, 1.0)
        liquidity_score = _clip(0.65 * depth_component + 0.35 * spread_component, 0.0, 1.0)

    volume_pressure = _clip(top5_imbalance, -1.0, 1.0)
    short_feasibility = "HIGH" if ask_depth > 0 and (spread_pct if math.isfinite(spread_pct) else 99.0) < 0.25 else (
        "MEDIUM" if ask_depth > 0 and (spread_pct if math.isfinite(spread_pct) else 99.0) < 0.75 else "LOW"
    )

    return {
        "Symbol": symbol,
        "L2 Available": True,
        "L2 Error": "",
        "L2 Depth Levels": max(len(bids), len(asks)),
        "L2 Best Bid": round(best_bid, 6) if math.isfinite(best_bid) else None,
        "L2 Best Ask": round(best_ask, 6) if math.isfinite(best_ask) else None,
        "L2 Mid Price": round(mid, 6) if math.isfinite(mid) else None,
        "L2 Spread": round(spread, 6) if math.isfinite(spread) else None,
        "L2 Spread %": round(spread_pct, 6) if math.isfinite(spread_pct) else None,
        "L2 Bid Depth": round(bid_depth, 2),
        "L2 Ask Depth": round(ask_depth, 2),
        "L2 Depth Imbalance": round(depth_imbalance, 6),
        "L2 Top5 Bid Depth": round(top5_bid_depth, 2),
        "L2 Top5 Ask Depth": round(top5_ask_depth, 2),
        "L2 Top5 Imbalance": round(top5_imbalance, 6),
        "L2 Microprice": round(microprice, 6) if math.isfinite(microprice) else None,
        "L2 Weighted Mid": round(weighted_mid, 6) if math.isfinite(weighted_mid) else None,
        "BID_ASK_SPREAD_PCT": round(spread_pct, 6) if math.isfinite(spread_pct) else None,
        "LIQUIDITY_SCORE": round(liquidity_score, 6) if math.isfinite(liquidity_score) else None,
        "Volume Pressure": round(volume_pressure, 6),
        "SHORT_FEASIBILITY": short_feasibility,
        "Primary_Entry_Source": "IB_LEVEL2",
        "DATA_SOURCE": "IB_LEVEL2+BAR",
    }


def fetch_level2_snapshot(symbol: str, cfg: Optional[IBLevel2Config] = None) -> Dict[str, Any]:
    cfg = cfg or IBLevel2Config()
    default = {
        "Symbol": symbol,
        "L2 Available": False,
        "L2 Error": "Level2 disabled",
        "L2 Depth Levels": 0,
        "L2 Best Bid": None,
        "L2 Best Ask": None,
        "L2 Mid Price": None,
        "L2 Spread": None,
        "L2 Spread %": None,
        "L2 Bid Depth": 0.0,
        "L2 Ask Depth": 0.0,
        "L2 Depth Imbalance": 0.0,
        "L2 Top5 Bid Depth": 0.0,
        "L2 Top5 Ask Depth": 0.0,
        "L2 Top5 Imbalance": 0.0,
        "L2 Microprice": None,
        "L2 Weighted Mid": None,
    }
    if not cfg.enabled:
        return default

    try:
        from ib_insync import IB  # type: ignore
    except Exception as exc:
        out = dict(default)
        out["L2 Error"] = f"ib_insync not installed: {exc}"
        return out

    ib = IB()
    try:
        ib_raw = IB()
        ib_raw.connect(cfg.host, cfg.port, clientId=cfg.client_id, timeout=cfg.connect_timeout_seconds, readonly=True,)
        ib = SafeIB(ib_raw)
        contract = _normalize_contract(ib, symbol, cfg)
        ticker = ib.reqMktDepth(contract, numRows=cfg.depth_rows, isSmartDepth=True)
        started = time.time()
        while time.time() - started < cfg.snapshot_wait_seconds:
            ib.sleep(0.2)
            depth = _read_depth_rows(ticker)
            if depth["bids"] or depth["asks"]:
                out = _compute_metrics(symbol, depth)
                return out
        out = dict(default)
        out["L2 Error"] = "No market depth returned"
        return out
    except Exception as exc:
        out = dict(default)
        out["L2 Error"] = str(exc)
        return out
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


def merge_level2_into_row(row: Dict[str, Any], l2: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    out.update({k: v for k, v in l2.items() if v is not None and v != ""})

    best_bid = _safe_float(l2.get("L2 Best Bid"))
    best_ask = _safe_float(l2.get("L2 Best Ask"))
    if math.isfinite(best_bid) and math.isfinite(best_ask):
        out.setdefault("Current Price", round((best_bid + best_ask) / 2.0, 6))
        if "Primary_Entry_Price" not in out and "Primary Entry Price" not in out:
            out["Primary_Entry_Price"] = round(best_bid, 6)

    if "SHORTABLE_FLAG" not in out:
        out["SHORTABLE_FLAG"] = "YES" if l2.get("SHORT_FEASIBILITY") in {"HIGH", "MEDIUM"} else "NO"

    return out
