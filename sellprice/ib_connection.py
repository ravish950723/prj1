from ib_insync import IB

ib = IB()

def connect_ib():
    ib.connect('127.0.0.1', 7497, clientId=1)

def disconnect_ib():
    ib.disconnect()

def get_ib():
    return ib
