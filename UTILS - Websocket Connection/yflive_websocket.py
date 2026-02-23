"""
YFLive WebSocket Client for Real-Time Market Data

This module provides a WebSocket client for connecting to YFLive's WebSocket API
to receive real-time market data, including stock prices, options data, and more.

Requirements:
    - websocket-client: Install with 'pip install websocket-client'

Example Usage:
    from yflive_websocket import YFLiveWebSocket

    def on_message(data):
        print(f"Received data: {data}")

    ws = YFLiveWebSocket(
        symbols=["AAPL", "MSFT", "GOOGL"],
        on_message=on_message
    )
    ws.connect()
"""

import json
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import websocket


class YFLiveWebSocket:
    """
    A WebSocket client for YFLive real-time market data.

    This client connects to the YFLive WebSocket API and handles the connection,
    subscription to symbols, and message processing.

    Args:
        symbols: List of stock symbols to subscribe to
        on_message: Callback function to handle incoming messages
        on_error: Callback function for error handling
        on_close: Callback function when connection is closed
        on_open: Callback function when connection is opened
        reconnect: Whether to automatically reconnect on disconnection (default: True)
        reconnect_interval: Seconds between reconnection attempts (default: 5)
    """

    def __init__(
        self,
        symbols: List[str],
        on_message: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        on_close: Optional[Callable[[], None]] = None,
        on_open: Optional[Callable[[], None]] = None,
        reconnect: bool = True,
        reconnect_interval: int = 5,
    ):
        self.symbols = symbols
        self.on_message_callback = on_message or self._default_on_message
        self.on_error_callback = on_error or self._default_on_error
        self.on_close_callback = on_close or self._default_on_close
        self.on_open_callback = on_open or self._default_on_open
        self.reconnect = reconnect
        self.reconnect_interval = reconnect_interval
        self.ws = None
        self.keep_running = True
        self.thread = None

        # YFLive WebSocket endpoint
        self.ws_url = "wss://streamer.finance.yahoo.com"

    def _default_on_message(self, data: Dict[str, Any]) -> None:
        """Default message handler that prints the received data."""
        print(f"[YFLive] {datetime.now().isoformat()} - Received data: {data}")

    def _default_on_error(self, error: str) -> None:
        """Default error handler that prints the error."""
        print(f"[YFLive] {datetime.now().isoformat()} - Error: {error}")

    def _default_on_close(self) -> None:
        """Default close handler that prints a message."""
        print(f"[YFLive] {datetime.now().isoformat()} - Connection closed")

    def _default_on_open(self) -> None:
        """Default open handler that prints a message and subscribes to symbols."""
        print(f"[YFLive] {datetime.now().isoformat()} - Connection opened")
        self.subscribe(self.symbols)

    def _on_message(self, ws: websocket.WebSocketApp, message: str) -> None:
        """Internal WebSocket message handler."""
        try:
            data = json.loads(message)
            self.on_message_callback(data)
        except json.JSONDecodeError as e:
            self.on_error_callback(f"Failed to parse message: {e}")

    def _on_error(self, ws: websocket.WebSocketApp, error: Exception) -> None:
        """Internal WebSocket error handler."""
        self.on_error_callback(str(error))

    def _on_close(self, ws: websocket.WebSocketApp, close_status_code: int, close_msg: str) -> None:
        """Internal WebSocket close handler."""
        self.on_close_callback()

        # Attempt to reconnect if enabled
        if self.reconnect and self.keep_running:
            print(f"[YFLive] Attempting to reconnect in {self.reconnect_interval} seconds...")
            time.sleep(self.reconnect_interval)
            self.connect()

    def _on_open(self, ws: websocket.WebSocketApp) -> None:
        """Internal WebSocket open handler."""
        self.on_open_callback()

    def _run_websocket(self) -> None:
        """Run the WebSocket client in a separate thread."""
        while self.keep_running:
            try:
                self.ws = websocket.WebSocketApp(
                    self.ws_url,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_open=self._on_open,
                )
                self.ws.run_forever()
            except Exception as e:
                self.on_error_callback(f"WebSocket error: {e}")
                if self.keep_running:
                    time.sleep(self.reconnect_interval)

    def connect(self) -> None:
        """
        Connect to the YFLive WebSocket API and start listening for messages.

        This method starts a new thread for the WebSocket connection.
        """
        if self.thread is not None and self.thread.is_alive():
            print("[YFLive] WebSocket is already running")
            return

        self.keep_running = True
        self.thread = threading.Thread(target=self._run_websocket, daemon=True)
        self.thread.start()

    def disconnect(self) -> None:
        """Disconnect from the WebSocket API and clean up resources."""
        self.keep_running = False
        if self.ws:
            self.ws.close()
        if self.thread:
            self.thread.join(timeout=5)

    def subscribe(self, symbols: List[str]) -> None:
        """
        Subscribe to real-time updates for the given symbols.

        Args:
            symbols: List of stock symbols to subscribe to
        """
        if not self.ws or not self.ws.sock or not self.ws.sock.connected:
            print("[YFLive] Not connected to WebSocket")
            return

        # YFLive expects a subscription message in this format
        subscription = {"subscribe": symbols, "unsubscribe": []}

        try:
            self.ws.send(json.dumps(subscription))
            print(f"[YFLive] Subscribed to symbols: {', '.join(symbols)}")
        except Exception as e:
            self.on_error_callback(f"Failed to subscribe: {e}")

    def unsubscribe(self, symbols: List[str]) -> None:
        """
        Unsubscribe from real-time updates for the given symbols.

        Args:
            symbols: List of stock symbols to unsubscribe from
        """
        if not self.ws or not self.ws.sock or not self.ws.sock.connected:
            print("[YFLive] Not connected to WebSocket")
            return

        # YFLive expects an unsubscription message in this format
        unsubscription = {"subscribe": [], "unsubscribe": symbols}

        try:
            self.ws.send(json.dumps(unsubscription))
            print(f"[YFLive] Unsubscribed from symbols: {', '.join(symbols)}")
        except Exception as e:
            self.on_error_callback(f"Failed to unsubscribe: {e}")


def example_usage():
    """Example usage of the YFLiveWebSocket class."""

    def on_message(data):
        # Process the received data
        if "id" in data and "price" in data:
            print(f"{data['id']}: ${data['price']} (Change: {data.get('changePercent', 0):.2f}%)")

    def on_error(error):
        print(f"Error: {error}")

    def on_close():
        print("Connection closed")

    def on_open():
        print("Connected to YFLive WebSocket")

    # Create and start the WebSocket client
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    ws = YFLiveWebSocket(
        symbols=symbols,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open,
        reconnect=True,
        reconnect_interval=5,
    )

    try:
        print("Connecting to YFLive WebSocket...")
        ws.connect()

        # Keep the main thread alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nDisconnecting...")
        ws.disconnect()
        print("Disconnected")


if __name__ == "__main__":
    example_usage()
