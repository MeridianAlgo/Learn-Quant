<p class="lq-badges"><span class="lq-badge lq-beginner">Beginner</span><span class="lq-badge lq-cat">Utilities & Tools</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Websocket Connection"
    python "finnhub.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Websocket%20Connection)

---
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


---

## Continue in Utilities & Tools

<div class="grid cards" markdown>

-   :material-tools: __[Core Utilities](Core Utilities.md)__

    This folder contains core mathematical and date/time utilities that form the foundation for quantitative finance calculations.

-   :material-tools: __[Currency Converter](Currency Converter.md)__

    **This utility does NOT use any external APIs.** All exchange rates are entered manually for learning and experimentation.

-   :material-tools: __[Data Processing](Data Processing.md)__

    This folder contains utilities for data processing, validation, and manipulation in financial applications.

-   :material-tools: __[Economic Calendar](Economic Calendar.md)__

    **This utility does NOT use any external APIs.** All data is managed locally for learning and experimentation.

-   :material-tools: __[Historical Data](Historical Data.md)__

    A Node.js script that fetches historical bars (OHLCV data) for stocks or crypto from the Alpaca Market Data API. It prompts interactively for the symbol type, symbol, timeframe, and date range, then prints the results as JSON.

-   :material-tools: __[Logging](Logging.md)__

    A pair of minimal, dependency-light logging utilities implemented in both Python and JavaScript. Each supports adding, reading, editing, and deleting log entries through an interactive command-line menu. All entries are persisted to a plain-text `log.txt` file in the working directory.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
