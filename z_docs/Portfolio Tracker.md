# Portfolio Tracker Utility (USES yfinance API)

<p class="lq-meta">Advanced <span>/</span> Portfolio Management <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Portfolio%20Tracker">source</a></p>


**This utility uses the yfinance API to fetch current prices automatically.** All other calculations and data are managed locally for learning and experimentation.

This utility helps you track your stock or crypto portfolio, including holdings, P&L, and allocation. **Current prices are fetched automatically using Yahoo Finance (via yfinance)**, so you don't have to enter them manually. All data is managed locally for privacy and educational purposes.

## Features
- Add, edit, and remove holdings (stocks, crypto, or any asset with a Yahoo Finance ticker)
- Track quantity, average cost, and current price (fetched automatically)
- Calculate total portfolio value, P&L, and allocation percentages
- Save and load your portfolio from a local file (JSON)
- Simple CLI interface (Python script)
- **Beginner-friendly:** All code is commented for learning

## Requirements
- Python 3.7+
- Install required package:
  ```sh
  pip install yfinance
  ```

## Setup
1. Copy `portfolio_tracker.py` to your desired folder.
2. (Optional) Prepare a JSON file with your initial holdings, or start from scratch.

## Usage Workflow (Step-by-Step)
1. Run the script:
   ```sh
   python portfolio_tracker.py
   ```
2. Follow the menu prompts:
   - Add a new holding (enter ticker, quantity, average cost)
   - Edit or remove holdings
   - View portfolio summary (value, P&L, allocation)
   - Save/load portfolio to/from file
   - Exit when done.

**Example tickers:**
- Stocks: `AAPL`, `TSLA`, `MSFT`
- Crypto: `BTC-USD`, `ETH-USD`

**Note:** This tool fetches prices using the yfinance API. You need an internet connection for price updates.

## Example Session
```
====================================
Welcome to the Portfolio Tracker!
This tool helps you learn Python and finance by tracking your investments.
- Add stocks or crypto by ticker (e.g., AAPL, BTC-USD)
- Prices are fetched automatically from Yahoo Finance
- All code is commented for beginners
====================================

Portfolio Tracker Menu:
1. Add holding
2. Edit holding
3. Remove holding
4. View summary
5. Save portfolio
6. Load portfolio
7. Exit
Enter your choice: 1
Enter asset ticker (e.g., AAPL, BTC-USD): AAPL
Enter quantity: 10
Enter average cost: 150
Holding added!
```

## Learning Notes
- **How does it fetch prices?** Uses the `yfinance` package to get the latest close price for any ticker.
- **How is the code structured?** Each function is commented to explain its purpose. The `Holding` class models each asset.
- **How can you extend it?** Try adding new features, like tracking dividends or plotting your portfolio value over time!

## License
MIT

## See Also
- [yfinance documentation](https://github.com/ranaroussi/yfinance)
- [Yahoo Finance Ticker Symbols](https://finance.yahoo.com/lookup)

---

## Running it

```bash
cd "Portfolio Tracker"
python "portfolio_tracker.py"
```

## More in Portfolio Management

- [Monte Carlo Portfolio Simulator](Monte Carlo Portfolio Simulator.md): This utility helps you forecast possible futures for a portfolio using random simulations—a key idea in finance, risk management, and statistics!
- [Portfolio Analytics - JavaScript](Portfolio Analytics - JavaScript.md): A portfolio is not the sum of its holdings, and the whole of portfolio theory follows from that one fact.
- [Portfolio Management](Portfolio Management.md): This folder contains utilities for portfolio management, risk analysis, and investment optimization.
- [Portfolio Management - Black Litterman](Portfolio Management - Black Litterman.md): The Black-Litterman (1990) model addresses the instability of mean-variance optimization by blending market equilibrium returns with investor views using Bayesian updating.
- [Portfolio Management - Hierarchical Risk Parity](Portfolio Management - Hierarchical Risk Parity.md): Markowitz optimisation needs the inverse of the covariance matrix, and the inverse of a noisy matrix is mostly noise.
- [Portfolio Management - Risk Parity](Portfolio Management - Risk Parity.md): Risk parity builds a portfolio where every asset contributes the same amount of risk to the total — not the same amount of capital.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
