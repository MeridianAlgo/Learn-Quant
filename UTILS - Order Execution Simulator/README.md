# Order Execution Simulator Utility (NO API)

**This utility does NOT use any external APIs.** All trades and portfolio data are managed locally for learning and experimentation.

This tool lets you simulate buy and sell orders, track a virtual portfolio, and analyze trade performance. All prices are entered manually.

## Features
- Simulate buy and sell orders for any asset (stocks, crypto, etc.)
- Track cash balance, holdings, and trade history
- Calculate realized and unrealized P&L
- Support for market and limit orders (simulated)
- Save and load your virtual portfolio and trade history
- CLI interface (Python script)
- **Beginner-friendly:** All code is commented for learning

## Requirements
- Python 3.7+
- No external libraries required (uses only Python standard library)

## Setup
1. Copy `order_execution_simulator.py` to your desired folder.
2. Open a terminal in that folder.

## Usage Workflow (Step-by-Step)
1. Run the script:
   ```sh
   python order_execution_simulator.py
   ```
2. Follow the menu prompts:
   - Place a buy or sell order (market or limit)
   - View portfolio and cash balance
   - View trade history and P&L
   - Save/load portfolio and trades
   - Exit when done.

**No real market data is used. This is for learning only!**

## Example Session
```
Welcome to the Order Execution Simulator!
1. Place order
2. View portfolio
3. View trade history
4. Save
5. Load
6. Exit
Enter your choice: 1
Order type (buy/sell): buy
Asset: TSLA
Quantity: 5
Price: 700
Order executed! Portfolio updated.
```

## Learning Notes
- **No API:** All calculations and data are managed in Python, so you can see and modify the logic yourself.
- **How does it work?** The code is structured with classes and functions, with comments explaining each step.
- **How can you extend it?** Try adding support for commissions, or tracking portfolio value over time!

## License
MIT
