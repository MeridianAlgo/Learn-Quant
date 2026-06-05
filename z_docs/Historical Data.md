<p class="lq-badges"><span class="lq-badge lq-beginner">Beginner</span><span class="lq-badge lq-cat">Utilities & Tools</span><span class="lq-badge lq-lang">JavaScript</span></p>

!!! tip "Run this module"
    ```bash
    cd "Historical Data"
    node "FetchBars.js"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Historical%20Data)

---
# Alpaca Historical Data Fetcher

A Node.js script that fetches historical bars (OHLCV data) for stocks or crypto from the Alpaca Market Data API. It prompts interactively for the symbol type, symbol, timeframe, and date range, then prints the results as JSON.

> **External API:** This utility calls the Alpaca Market Data API. A network connection and valid API credentials are required.

## Files

| File | Description |
|---|---|
| `FetchBars.js` | Interactive Node.js historical-bars fetcher |
| `package.json` | Node.js dependencies |

## Requirements

- Node.js v14 or higher
- An Alpaca account with API credentials

## Setup

1. Install dependencies:
   ```sh
   npm install node-fetch dotenv
   ```
2. Create a `.env` file in the project root to hold your credentials:
   ```env
   ALPACA_API_KEY=your_alpaca_api_key_here
   ALPACA_API_SECRET=your_alpaca_api_secret_here
   ```

## Usage

```sh
node FetchBars.js
```

You will be prompted for:

- Symbol type — `stock` or `crypto`
- Symbol — e.g. `AAPL` for stocks, `BTC/USD` for crypto
- Timeframe — e.g. `1Day`, `1Hour`, `5Min`
- Start date — `YYYY-MM-DD`
- End date — `YYYY-MM-DD`

The script fetches the matching bars and prints them in JSON format.

### Example session

```
Enter symbol type (stock/crypto): stock
Enter symbol (e.g. AAPL for stock, BTC/USD for crypto): AAPL
Enter timeframe (e.g. 1Day, 1Hour, 5Min): 1Day
Enter start date (YYYY-MM-DD): 2023-01-01
Enter end date (YYYY-MM-DD): 2023-01-31
```

## Notes

- Ensure your API key has permission for the data class you request.
- See the [Alpaca Stock Bars API docs](https://docs.alpaca.markets/reference/stockbars) and [Alpaca Crypto Bars API docs](https://docs.alpaca.markets/reference/cryptobars-1) for the full list of supported timeframes and parameters.

## License

MIT


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

-   :material-tools: __[Logging](Logging.md)__

    A pair of minimal, dependency-light logging utilities implemented in both Python and JavaScript. Each supports adding, reading, editing, and deleting log entries through an interactive command-line menu. All entries are persisted to a plain-text `log.txt` file in the working directory.

-   :material-tools: __[Market Data](Market Data.md)__

    This folder contains utilities for processing, analyzing, and fetching market data for financial applications.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
