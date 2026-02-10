# -----------------------------
# Portfolio Tracker Utility (with yfinance)
# -----------------------------
# This script helps you track your stock or crypto portfolio.
# It fetches current prices automatically using yfinance (for stocks/crypto on Yahoo Finance).
# All code is commented for beginners to learn Python step by step.
# -----------------------------

import json
import os

import yfinance as yf  # Make sure to install yfinance: pip install yfinance

PORTFOLIO_FILE = "portfolio.json"  # Default file for saving/loading


class Holding:
    """
    Represents a single asset holding in your portfolio.
    name: The ticker symbol (e.g., 'AAPL', 'BTC-USD')
    quantity: Number of shares/units owned
    avg_cost: Average cost per share/unit
    """

    def __init__(self, name, quantity, avg_cost):
        self.name = name.upper()
        self.quantity = quantity
        self.avg_cost = avg_cost

    def fetch_current_price(self):
        """
        Fetch the current price using yfinance.
        Returns the latest close price, or None if not found.
        """
        try:
            ticker = yf.Ticker(self.name)
            price = ticker.history(period="1d")["Close"][-1]
            return float(price)
        except Exception as e:
            print(f"Could not fetch price for {self.name}: {e}")
            return None

    def market_value(self):
        price = self.fetch_current_price()
        if price is None:
            return 0
        return self.quantity * price

    def cost_basis(self):
        return self.quantity * self.avg_cost

    def pnl(self):
        price = self.fetch_current_price()
        if price is None:
            return 0
        return (price - self.avg_cost) * self.quantity

    def to_dict(self):
        return {"name": self.name, "quantity": self.quantity, "avg_cost": self.avg_cost}

    @staticmethod
    def from_dict(d):
        return Holding(d["name"], d["quantity"], d["avg_cost"])


# -----------------------------
# Helper Functions
# -----------------------------
def print_menu():
    print("\nPortfolio Tracker Menu:")
    print("1. Add holding")
    print("2. Edit holding")
    print("3. Remove holding")
    print("4. View summary")
    print("5. Save portfolio")
    print("6. Load portfolio")
    print("7. Exit")


def input_float(prompt):
    """Prompt the user for a float value, retrying on invalid input."""
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Invalid input. Please enter a number.")


def add_holding(portfolio):
    """Add a new asset to the portfolio."""
    name = input("Enter asset ticker (e.g., AAPL, BTC-USD): ").strip().upper()
    quantity = input_float("Enter quantity: ")
    avg_cost = input_float("Enter average cost: ")
    portfolio.append(Holding(name, quantity, avg_cost))
    print("Holding added!")


def find_holding(portfolio, name):
    """Find the index of a holding by ticker symbol."""
    for i, h in enumerate(portfolio):
        if h.name == name.upper():
            return i
    return -1


def edit_holding(portfolio):
    """Edit an existing holding's details."""
    name = input("Enter asset ticker to edit: ").strip().upper()
    idx = find_holding(portfolio, name)
    if idx == -1:
        print("Holding not found.")
        return
    h = portfolio[idx]
    print(f"Editing {h.name} (qty: {h.quantity}, avg cost: {h.avg_cost})")
    h.quantity = input_float("Enter new quantity: ")
    h.avg_cost = input_float("Enter new average cost: ")
    print("Holding updated!")


def remove_holding(portfolio):
    """Remove a holding from the portfolio."""
    name = input("Enter asset ticker to remove: ").strip().upper()
    idx = find_holding(portfolio, name)
    if idx == -1:
        print("Holding not found.")
        return
    del portfolio[idx]
    print("Holding removed!")


def view_summary(portfolio):
    """Display a summary of the portfolio, including P&L and allocation."""
    if not portfolio:
        print("Portfolio is empty.")
        return
    total_value = 0
    total_cost = 0
    print(f"\n{'Asset':<10} {'Qty':>8} {'Avg Cost':>10} {'Price':>10} {'Value':>12} {'P&L':>10} {'Alloc%':>8}")
    values = []
    for h in portfolio:
        price = h.fetch_current_price()
        if price is None:
            price = 0
        value = h.quantity * price
        pnl = (price - h.avg_cost) * h.quantity
        values.append(value)
        total_value += value
        total_cost += h.cost_basis()
        alloc = 0  # Will be calculated after total_value
        print(f"{h.name:<10} {h.quantity:>8.2f} {h.avg_cost:>10.2f} {price:>10.2f} {value:>12.2f} {pnl:>10.2f}")
    # Print allocation
    print("\nAllocation:")
    for i, h in enumerate(portfolio):
        alloc = (values[i] / total_value) * 100 if total_value else 0
        print(f"{h.name:<10}: {alloc:>6.2f}%")
    print(f"\nTotal Value: {total_value:.2f}")
    print(f"Total Cost: {total_cost:.2f}")
    print(f"Total P&L: {total_value - total_cost:.2f}")


def save_portfolio(portfolio):
    """Save the portfolio to a JSON file."""
    fname = input(f"Enter filename to save (default: {PORTFOLIO_FILE}): ").strip() or PORTFOLIO_FILE
    data = [h.to_dict() for h in portfolio]
    with open(fname, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Portfolio saved to {fname}.")


def load_portfolio():
    """Load the portfolio from a JSON file."""
    fname = input(f"Enter filename to load (default: {PORTFOLIO_FILE}): ").strip() or PORTFOLIO_FILE
    if not os.path.exists(fname):
        print("File not found.")
        return []
    with open(fname) as f:
        data = json.load(f)
    portfolio = [Holding.from_dict(d) for d in data]
    print(f"Loaded {len(portfolio)} holdings from {fname}.")
    return portfolio


def main():
    print(
        """
====================================
Welcome to the Portfolio Tracker!
This tool helps you learn Python and finance by tracking your investments.
- Add stocks or crypto by ticker (e.g., AAPL, BTC-USD)
- Prices are fetched automatically from Yahoo Finance
- All code is commented for beginners
====================================
"""
    )
    portfolio = []
    while True:
        print_menu()
        choice = input("Enter your choice: ").strip()
        if choice == "1":
            add_holding(portfolio)
        elif choice == "2":
            edit_holding(portfolio)
        elif choice == "3":
            remove_holding(portfolio)
        elif choice == "4":
            view_summary(portfolio)
        elif choice == "5":
            save_portfolio(portfolio)
        elif choice == "6":
            portfolio = load_portfolio()
        elif choice == "7":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
