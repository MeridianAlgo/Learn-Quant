# finnhub.py
# Simple example: Connect to Finnhub WebSocket and print real-time trade data for a symbol

import json
import threading
import time
from datetime import datetime

import websocket

# Set your Finnhub API key directly here
FINNHUB_API_KEY = "FINNHUB_API_KEY"  # <-- Replace with your actual API key

# Symbol to subscribe to (e.g., 'BINANCE:BTCUSDT' for Bitcoin/USDT on Binance, or 'AAPL' for Apple)
SYMBOL = "AAPL"

# Finnhub WebSocket endpoint with your API key
FINNHUB_WS_URL = f"wss://ws.finnhub.io?token={FINNHUB_API_KEY}"


def on_open(ws):
    # Called when the WebSocket connection is opened
    print("WebSocket opened")
    # Subscribe to the symbol
    subscribe_message = json.dumps({"type": "subscribe", "symbol": SYMBOL})
    ws.send(subscribe_message)


def on_message(ws, message):
    # Called when a message is received from the server
    data = json.loads(message)
    if data.get("type") == "trade":
        for trade in data["data"]:
            # Extract and format trade info
            symbol = trade["s"]
            price = trade["p"]
            volume = trade["v"]
            timestamp = trade["t"]
            time_str = datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d %H:%M:%S")
            print(f"Symbol: {symbol} | Price: {price} | Volume: {volume} | Time: {time_str}")
    # else: handle other message types if needed


def on_error(ws, error):
    # Called on error
    print("WebSocket error:", error)


def on_close(ws, close_status_code, close_msg):
    # Called when the connection is closed
    print("WebSocket connection closed")


def run():
    # Set up and start the WebSocket connection in a background thread
    ws = websocket.WebSocketApp(
        FINNHUB_WS_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    wst = threading.Thread(target=ws.run_forever)
    wst.daemon = True
    wst.start()
    # Keep the connection open for 30 seconds to receive data
    time.sleep(30)
    print("Closing WebSocket after 30 seconds...")
    ws.close()
    wst.join()


if __name__ == "__main__":
    run()
