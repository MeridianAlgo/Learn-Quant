<<<<<<< HEAD
# Alpaca Historical Data Fetcher

This Node.js script fetches historical bars (OHLCV data) for stocks or crypto from the Alpaca Market Data API. It prompts you for the symbol type, symbol, timeframe, and date range, then displays the results in JSON format.

## Features
- Fetches historical data for both stocks and crypto
- Interactive prompts for symbol type, symbol, timeframe, and date range
- Uses environment variables for secure API credential management

## Prerequisites
- Node.js (v14 or higher recommended)
- An Alpaca account with API credentials

## Setup
1. **Clone or download this repository.**
2. **Install dependencies:**
   ```bash
   npm install node-fetch dotenv
   ```
3. **Create a `.env` file in the project root.**
   This file is required to securely store your Alpaca API credentials. Create a file named `.env` (no filename, just the extension) and add the following lines:
   ```env
   ALPACA_API_KEY=your_alpaca_api_key_here
   ALPACA_API_SECRET=your_alpaca_api_secret_here
   ```
   Replace `your_alpaca_api_key_here` and `your_alpaca_api_secret_here` with your actual Alpaca API key and secret.

## Usage
Run the script with Node.js:
```bash
node fetchAlpacaHistorical.js
```
You will be prompted for:
- Symbol type (`stock` or `crypto`)
- Symbol (e.g., `AAPL` for stocks, `BTC/USD` for crypto)
- Timeframe (e.g., `1Day`, `1Hour`, `5Min`)
- Start date (`YYYY-MM-DD`)
- End date (`YYYY-MM-DD`)

The script will fetch and print the historical bars in JSON format.

## Example
```
Enter symbol type (stock/crypto): stock
Enter symbol (e.g. AAPL for stock, BTC/USD for crypto): AAPL
Enter timeframe (e.g. 1Day, 1Hour, 5Min): 1Day
Enter start date (YYYY-MM-DD): 2023-01-01
Enter end date (YYYY-MM-DD): 2023-01-31
```

## Notes
- For more information on available timeframes and API details, see the [Alpaca Stock Bars API docs](https://docs.alpaca.markets/reference/stockbars) and [Alpaca Crypto Bars API docs](https://docs.alpaca.markets/reference/cryptobars-1).
- Ensure your API key has the necessary permissions for the data you are requesting.

## License
MIT 
=======
# Alpaca Historical Data Fetcher (USES Alpaca API)

**This utility uses the Alpaca API to fetch historical bars.** All other logic and data are managed locally for learning and experimentation.

This Node.js script fetches historical bars (OHLCV data) for stocks or crypto from the Alpaca Market Data API. It prompts you for the symbol type, symbol, timeframe, and date range, then displays the results in JSON format.

## Features
- Fetches historical data for both stocks and crypto
- Interactive prompts for symbol type, symbol, timeframe, and date range
- Uses environment variables for secure API credential management
- **Beginner-friendly:** All code is commented for learning

## Requirements
- Node.js (v14 or higher recommended)
- An Alpaca account with API credentials

## Setup
1. Clone or download this repository.
2. Install dependencies:
   ```sh
   npm install node-fetch dotenv
   ```
3. Create a `.env` file in the project root. This file is required to securely store your Alpaca API credentials:
   ```env
   ALPACA_API_KEY=your_alpaca_api_key_here
   ALPACA_API_SECRET=your_alpaca_api_secret_here
   ```

## Usage Workflow (Step-by-Step)
1. Run the script:
   ```sh
   node FetchBars.js
   ```
2. Follow the prompts:
   - Enter symbol type (`stock` or `crypto`)
   - Enter symbol (e.g., `AAPL` for stocks, `BTC/USD` for crypto)
   - Enter timeframe (e.g., `1Day`, `1Hour`, `5Min`)
   - Enter start and end dates
3. The script will fetch and print the historical bars in JSON format.

**Note:** This tool fetches data using the Alpaca API. You need an internet connection and valid API credentials.

## Educational Notes
- **How does it work?** The script sends your request to the Alpaca API and prints the response.
- **How is the code structured?** Each function is commented to explain its purpose. The code is designed for easy modification.
- **How can you extend it?** Try adding support for more timeframes, or saving results to a file!

## License
MIT

## References
- [Alpaca Stock Bars API docs](https://docs.alpaca.markets/reference/stockbars)
- [Alpaca Crypto Bars API docs](https://docs.alpaca.markets/reference/cryptobars-1) 
>>>>>>> 8944b09 (Initial commit: Comprehensive Python & JS Finance Utilities for Beginners (API & API-free, with detailed docs))
