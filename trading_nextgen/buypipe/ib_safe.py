class SafeIB:
    def __init__(self, ib):
        self._ib = ib

    def __getattr__(self, name):
        blocked = {
            "placeOrder",
            "bracketOrder",
            "reqOpenOrders",
            "reqAllOpenOrders",
            "reqCompletedOrders",
            "cancelOrder",
            "whatIfOrder",
            "openOrders",
            "trades",
            "reqExecutions",
        }

        if name in blocked:
            raise RuntimeError(f"🚫 BLOCKED IB ORDER API CALL: {name}")

        return getattr(self._ib, name)