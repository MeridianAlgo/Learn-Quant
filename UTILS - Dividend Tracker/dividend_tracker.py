# -----------------------------
# Dividend Tracker Utility (NO API)
# -----------------------------
# This script lets you track upcoming dividends and expected income.
# All data is managed locally (no APIs, no real market data).
# All code is commented for beginners to learn Python and finance.
# -----------------------------

import json
import os
from datetime import datetime

DIVIDENDS_FILE = "dividends.json"  # Default file for saving/loading dividends


# Dividend class: represents a single dividend entry
class Dividend:
    def __init__(self, ticker, ex_date, pay_date, amount, shares):
        self.ticker = ticker.upper()
        self.ex_date = ex_date  # YYYY-MM-DD
        self.pay_date = pay_date  # YYYY-MM-DD
        self.amount = amount  # per share
        self.shares = shares

    def total_income(self):
        return self.amount * self.shares

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(d):
        return Dividend(
            d["ticker"], d["ex_date"], d["pay_date"], d["amount"], d["shares"]
        )


# Helper to get a date from user input
def input_date(prompt):
    while True:
        s = input(prompt)
        try:
            datetime.strptime(s, "%Y-%m-%d")
            return s
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")


# Print the main menu for the CLI
def print_menu():
    print("\nDividend Tracker Menu:")
    print("1. Add dividend")
    print("2. Edit dividend")
    print("3. Remove dividend")
    print("4. View calendar")
    print("5. Calculate total income")
    print("6. Exit")


# Add a new dividend entry
def add_dividend(dividends):
    ticker = input("Enter ticker: ").strip().upper()
    ex_date = input_date("Enter ex-date (YYYY-MM-DD): ")
    pay_date = input_date("Enter pay date (YYYY-MM-DD): ")
    amount = float(input("Enter dividend amount per share: "))
    shares = float(input("Enter number of shares: "))
    dividends.append(Dividend(ticker, ex_date, pay_date, amount, shares))
    print("Dividend added!")


# Find dividend by ticker and ex-date
def find_dividend(dividends, ticker, ex_date):
    for i, d in enumerate(dividends):
        if d.ticker == ticker.upper() and d.ex_date == ex_date:
            return i
    return -1


# Edit an existing dividend entry
def edit_dividend(dividends):
    ticker = input("Enter ticker to edit: ").strip().upper()
    ex_date = input_date("Enter ex-date to edit (YYYY-MM-DD): ")
    idx = find_dividend(dividends, ticker, ex_date)
    if idx == -1:
        print("Dividend not found.")
        return
    d = dividends[idx]
    print(
        f"Editing {d.ticker} ex-date {d.ex_date}: {d.amount} per share, {d.shares} shares"
    )
    d.pay_date = input_date("Enter new pay date (YYYY-MM-DD): ")
    d.amount = float(input("Enter new dividend amount per share: "))
    d.shares = float(input("Enter new number of shares: "))
    print("Dividend updated!")


# Remove a dividend entry
def remove_dividend(dividends):
    ticker = input("Enter ticker to remove: ").strip().upper()
    ex_date = input_date("Enter ex-date to remove (YYYY-MM-DD): ")
    idx = find_dividend(dividends, ticker, ex_date)
    if idx == -1:
        print("Dividend not found.")
        return
    del dividends[idx]
    print("Dividend removed!")


# View all dividends in date order
def view_calendar(dividends):
    if not dividends:
        print("No dividends scheduled.")
        return
    print("\nUpcoming Dividends:")
    for d in sorted(dividends, key=lambda x: x.ex_date):
        print(
            f"{d.ex_date} | {d.ticker:<8} | Pay: {d.pay_date} | {d.amount:.2f}/sh | {d.shares} shares | Total: {d.total_income():.2f}"
        )


# Calculate total expected dividend income
def calculate_total_income(dividends):
    total = sum(d.total_income() for d in dividends)
    print(f"Total expected dividend income: {total:.2f}")


# Save dividends to a file
def save_dividends(dividends):
    fname = DIVIDENDS_FILE
    data = [d.to_dict() for d in dividends]
    with open(fname, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Dividends saved to {fname}.")


# Load dividends from a file
def load_dividends():
    fname = DIVIDENDS_FILE
    if not os.path.exists(fname):
        return []
    with open(fname, "r") as f:
        data = json.load(f)
    dividends = [Dividend.from_dict(d) for d in data]
    return dividends


# Main workflow for the CLI
def main():
    print(
        """
====================================
Welcome to the Dividend Tracker!
This tool helps you learn Python and dividend investing by tracking payouts and income.
- No APIs or real market data are used.
- All code is commented for beginners.
====================================
"""
    )
    dividends = load_dividends()
    while True:
        print_menu()
        choice = input("Enter your choice: ").strip()
        if choice == "1":
            add_dividend(dividends)
            save_dividends(dividends)
        elif choice == "2":
            edit_dividend(dividends)
            save_dividends(dividends)
        elif choice == "3":
            remove_dividend(dividends)
            save_dividends(dividends)
        elif choice == "4":
            view_calendar(dividends)
        elif choice == "5":
            calculate_total_income(dividends)
        elif choice == "6":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
