# WebSocket Connection Utilities

This project provides WebSocket clients for connecting to various financial data providers, including YFLive and Finnhub. These utilities are designed for real-time market data streaming and analysis.

## Available Clients

### 1. YFLive WebSocket Client
A robust WebSocket client for connecting to YFLive's real-time market data feed.

#### Features
- Real-time stock price updates
- Support for multiple symbols
- Automatic reconnection
- Customizable callbacks
- Thread-safe implementation

#### Requirements
- Python 3.7+
- websocket-client
- python-dateutil

#### Installation
```bash
pip install -r requirements.txt
```

#### Quick Start
```python
from yflive_websocket import YFLiveWebSocket

def on_message(data):
    print(f"Received data: {data}")

# Create and start the WebSocket client
ws = YFLiveWebSocket(
    symbols=["AAPL", "MSFT", "GOOGL"],
    on_message=on_message
)
ws.connect()

# Keep the script running
import time
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    ws.disconnect()
```

#### Example Output
```
[YFLive] 2023-09-27T12:00:00.000000 - Connection opened
[YFLive] Subscribed to symbols: AAPL, MSFT, GOOGL
[YFLive] 2023-09-27T12:00:01.123456 - Received data: {'id': 'AAPL', 'price': 150.25, 'changePercent': 0.5}
[YFLive] 2023-09-27T12:00:01.234567 - Received data: {'id': 'MSFT', 'price': 325.10, 'changePercent': 0.3}
```

## Documentation

### YFLiveWebSocket Class API

#### Initialization
```python
YFLiveWebSocket(
    symbols: List[str],
    on_message: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_error: Optional[Callable[[str], None]] = None,
    on_close: Optional[Callable[[], None]] = None,
    on_open: Optional[Callable[[], None]] = None,
    reconnect: bool = True,
    reconnect_interval: int = 5
)
```

#### Key Methods
- `connect()`: Start the WebSocket connection and background thread.
- `disconnect()`: Gracefully close the connection and stop the thread.
- `subscribe(symbols)`: Subscribe to additional tickers at runtime.
- `unsubscribe(symbols)`: Stop receiving updates for specific tickers.

### Error Handling & Reconnection
- Automatic reconnection with exponential backoff when `reconnect=True`.
- Custom callbacks for open, close, error, and message events.
- JSON parsing safety with graceful error reporting.

## Requirements
- Python 3.8 or higher.
- Install dependencies with:
  ```bash
  pip install -r requirements.txt
  ```

## Project Structure
- `yflive_websocket.py`: Main YFLive client implementation.
- `finnhub.py`: Legacy Finnhub example (kept for reference).
- `requirements.txt`: Python dependencies.
- `README.md`: This documentation.

## Usage Workflow
1. Install requirements.
2. Review `example_usage()` in `yflive_websocket.py` for a template.
3. Run your script or interactively explore in Jupyter/VS Code.

## Educational Notes
- Experiment with different symbol lists to observe simultaneous streams.
- Implement persistence by writing data to CSV/SQLite.
- Combine with the portfolio utilities in `UTILS - Portfolio Tracker/` for live monitoring.

## License
MIT

## References
- [YFLive WebSocket Docs](https://streamer.finance.yahoo.com)
- [websocket-client Documentation](https://websocket-client.readthedocs.io/)
