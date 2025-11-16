# -----------------------------
# Order Execution Simulator Utility (NO API)
# -----------------------------
# This script lets you simulate buy/sell orders and track a virtual portfolio.
# All calculations and data are managed locally (no APIs, no real market data).
# All code is commented for beginners to learn Python and finance.
# -----------------------------

import json
import os

# Trade class: represents a single buy or sell order
class Trade:
    def __init__(self, asset, qty, price, side, order_type):
        self.asset = asset  # Ticker symbol (e.g., 'AAPL')
        self.qty = qty      # Number of shares/units
        self.price = price  # Price per share/unit
        self.side = side    # 'buy' or 'sell'
        self.order_type = order_type  # 'market' or 'limit'

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(d):
        return Trade(d['asset'], d['qty'], d['price'], d['side'], d['order_type'])

# Portfolio class: manages cash, holdings, and trade history
class Portfolio:
    def __init__(self, cash=10000):
        self.cash = cash
        self.holdings = {}  # asset -> qty
        self.trades = []
        self.avg_cost = {}  # asset -> avg cost

    # Execute a buy or sell order
    def execute_order(self, asset, qty, price, side, order_type):
        if side == 'buy':
            cost = qty * price
            if self.cash < cost:
                print("Not enough cash.")
                return False
            self.cash -= cost
            self.holdings[asset] = self.holdings.get(asset, 0) + qty
            # Update average cost
            prev_qty = self.holdings[asset] - qty
            prev_cost = self.avg_cost.get(asset, 0) * prev_qty
            self.avg_cost[asset] = (prev_cost + cost) / (prev_qty + qty) if (prev_qty + qty) > 0 else price
        else:  # sell
            if self.holdings.get(asset, 0) < qty:
                print("Not enough holdings to sell.")
                return False
            self.cash += qty * price
            self.holdings[asset] -= qty
            if self.holdings[asset] == 0:
                self.avg_cost[asset] = 0
        self.trades.append(Trade(asset, qty, price, side, order_type))
        print("Order executed! Portfolio updated.")
        return True

    # Calculate total portfolio value (prompt user for current prices)
    def portfolio_value(self):
        value = self.cash
        for asset, qty in self.holdings.items():
            if qty > 0:
                price = input_float(f"Enter current price for {asset}: ")
                value += qty * price
        return value

    # Calculate realized profit and loss
    def realized_pnl(self):
        pnl = 0
        for t in self.trades:
            if t.side == 'sell':
                cost = self.avg_cost.get(t.asset, 0)
                pnl += (t.price - cost) * t.qty
        return pnl

    def to_dict(self):
        return {
            'cash': self.cash,
            'holdings': self.holdings,
            'avg_cost': self.avg_cost,
            'trades': [t.to_dict() for t in self.trades]
        }

    @staticmethod
    def from_dict(d):
        p = Portfolio(d['cash'])
        p.holdings = d['holdings']
        p.avg_cost = d.get('avg_cost', {})
        p.trades = [Trade.from_dict(td) for td in d['trades']]
        return p

# Helper to get a float from user input
def input_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Invalid input. Please enter a number.")

# Print the main menu for the CLI
def print_menu():
    print("\nOrder Execution Simulator Menu:")
    print("1. Place order")
    print("2. View portfolio")
    print("3. View trade history")
    print("4. Save")
    print("5. Load")
    print("6. Exit")

# Place a buy or sell order
def place_order(portfolio):
    side = input("Order type (buy/sell): ").strip().lower()
    if side not in ('buy', 'sell'):
        print("Invalid side.")
        return
    asset = input("Asset: ").strip().upper()
    qty = input_float("Quantity: ")
    order_type = input("Order type (market/limit): ").strip().lower()
    if order_type not in ('market', 'limit'):
        print("Invalid order type.")
        return
    price = input_float("Price: ")
    portfolio.execute_order(asset, qty, price, side, order_type)

# View current portfolio and cash balance
def view_portfolio(portfolio):
    print(f"\nCash balance: {portfolio.cash:.2f}")
    print(f"{'Asset':<10} {'Qty':>8} {'Avg Cost':>10}")
    for asset, qty in portfolio.holdings.items():
        if qty > 0:
            print(f"{asset:<10} {qty:>8.2f} {portfolio.avg_cost.get(asset, 0):>10.2f}")
    print(f"Realized P&L: {portfolio.realized_pnl():.2f}")
    print(f"Total Portfolio Value (prompt for prices): {portfolio.portfolio_value():.2f}")

# View trade history
def view_trades(portfolio):
    print(f"\nTrade History:")
    print(f"{'Side':<6} {'Asset':<8} {'Qty':>6} {'Price':>10} {'Type':>8}")
    for t in portfolio.trades:
        print(f"{t.side:<6} {t.asset:<8} {t.qty:>6.2f} {t.price:>10.2f} {t.order_type:>8}")

# Save portfolio and trades to a file
def save(portfolio):
    fname = input(f"Enter filename to save (default: sim_portfolio.json): ").strip() or 'sim_portfolio.json'
    with open(fname, 'w') as f:
        json.dump(portfolio.to_dict(), f, indent=2)
    print(f"Saved to {fname}.")

# Load portfolio and trades from a file
def load():
    fname = input(f"Enter filename to load (default: sim_portfolio.json): ").strip() or 'sim_portfolio.json'
    if not os.path.exists(fname):
        print("File not found.")
        return Portfolio()
    with open(fname, 'r') as f:
        d = json.load(f)
    print(f"Loaded from {fname}.")
    return Portfolio.from_dict(d)

# Main workflow for the CLI
def main():
    print("""
====================================
Welcome to the Order Execution Simulator!
This tool helps you learn Python and trading basics by simulating orders and portfolio management.
- No APIs or real market data are used.
- All code is commented for beginners.
====================================
""")
    portfolio = Portfolio()
    while True:
        print_menu()
        choice = input("Enter your choice: ").strip()
        if choice == '1':
            place_order(portfolio)
        elif choice == '2':
            view_portfolio(portfolio)
        elif choice == '3':
            view_trades(portfolio)
        elif choice == '4':
            save(portfolio)
        elif choice == '5':
            portfolio = load()
        elif choice == '6':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
