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
