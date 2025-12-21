# ib_connection.py
import random
from ib_insync import IB

_IB: IB | None = None

def connect_ib(host="127.0.0.1", port=7497, max_retries=5) -> IB:
    """
    Connect and store a singleton IB instance.
    Compatible with callers that do connect_ib() without capturing return.
    """
    global _IB
    if _IB is not None and _IB.isConnected():
        return _IB

    ib = IB()
    last_err = None
    for _ in range(max_retries):
        try:
            client_id = random.randint(1000, 9999)
            ib.connect(host, port, clientId=client_id, timeout=5)
            print(f"[IB] Connected with clientId={client_id}")
            _IB = ib
            return _IB
        except Exception as e:
            last_err = e
            print(f"[WARN] IB connect failed, retrying: {e}")

    raise RuntimeError(f"Unable to connect to IB after retries. Last error: {last_err}")


def get_ib(host="127.0.0.1", port=7497) -> IB:
    """
    Backward compatible API expected by data_fetcher.py:
    from ib_connection import get_ib
    """
    return connect_ib(host=host, port=port)


def disconnect_ib() -> None:
    """
    Backward compatible: sell.py calls disconnect_ib() with no args.
    """
    global _IB
    try:
        if _IB is not None and _IB.isConnected():
            _IB.disconnect()
            print("[IB] Disconnected")
    finally:
        _IB = None
