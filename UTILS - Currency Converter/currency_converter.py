# -----------------------------
# Currency Converter Utility (NO API)
# -----------------------------
# This script lets you convert between currencies using user-supplied rates.
# All data is managed locally (no APIs, no real market data).
# All code is commented for beginners to learn Python and finance.
# -----------------------------

import json
import os

RATES_FILE = 'rates.json'  # Default file for saving/loading rates

# ExchangeRate class: represents a single currency pair and rate
class ExchangeRate:
    def __init__(self, base, quote, rate):
        self.base = base.upper()
        self.quote = quote.upper()
        self.rate = rate  # 1 base = rate quote

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(d):
        return ExchangeRate(d['base'], d['quote'], d['rate'])

# Print the main menu for the CLI
def print_menu():
    print("\nCurrency Converter Menu:")
    print("1. Add rate")
    print("2. Edit rate")
    print("3. Remove rate")
    print("4. Convert currency")
    print("5. View all rates")
    print("6. Exit")

# Add a new exchange rate
def add_rate(rates):
    base = input("Enter base currency (e.g., USD): ").strip().upper()
    quote = input("Enter quote currency (e.g., EUR): ").strip().upper()
    rate = float(input(f"Enter exchange rate (1 {base} = ? {quote}): "))
    rates.append(ExchangeRate(base, quote, rate))
    print("Rate added!")

# Find rate by base and quote
def find_rate(rates, base, quote):
    for i, r in enumerate(rates):
        if r.base == base.upper() and r.quote == quote.upper():
            return i
    return -1

# Edit an existing rate
def edit_rate(rates):
    base = input("Enter base currency to edit: ").strip().upper()
    quote = input("Enter quote currency to edit: ").strip().upper()
    idx = find_rate(rates, base, quote)
    if idx == -1:
        print("Rate not found.")
        return
    rate = float(input(f"Enter new exchange rate (1 {base} = ? {quote}): "))
    rates[idx].rate = rate
    print("Rate updated!")

# Remove a rate
def remove_rate(rates):
    base = input("Enter base currency to remove: ").strip().upper()
    quote = input("Enter quote currency to remove: ").strip().upper()
    idx = find_rate(rates, base, quote)
    if idx == -1:
        print("Rate not found.")
        return
    del rates[idx]
    print("Rate removed!")

# Convert between two currencies
def convert_currency(rates):
    base = input("Enter base currency: ").strip().upper()
    quote = input("Enter quote currency: ").strip().upper()
    idx = find_rate(rates, base, quote)
    if idx == -1:
        print("Rate not found. Please add it first.")
        return
    amount = float(input(f"Enter amount in {base}: "))
    rate = rates[idx].rate
    converted = amount * rate
    print(f"{amount:.2f} {base} = {converted:.2f} {quote}")

# View all stored rates
def view_all_rates(rates):
    if not rates:
        print("No rates stored.")
        return
    print("\nStored Exchange Rates:")
    for r in rates:
        print(f"1 {r.base} = {r.rate:.4f} {r.quote}")

# Save rates to a file
def save_rates(rates):
    fname = RATES_FILE
    data = [r.to_dict() for r in rates]
    with open(fname, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Rates saved to {fname}.")

# Load rates from a file
def load_rates():
    fname = RATES_FILE
    if not os.path.exists(fname):
        return []
    with open(fname, 'r') as f:
        data = json.load(f)
    rates = [ExchangeRate.from_dict(d) for d in data]
    return rates

# Main workflow for the CLI
def main():
    print("""
====================================
Welcome to the Currency Converter!
This tool helps you learn Python and FX basics by converting currencies with your own rates.
- No APIs or real market data are used.
- All code is commented for beginners.
====================================
""")
    rates = load_rates()
    while True:
        print_menu()
        choice = input("Enter your choice: ").strip()
        if choice == '1':
            add_rate(rates)
            save_rates(rates)
        elif choice == '2':
            edit_rate(rates)
            save_rates(rates)
        elif choice == '3':
            remove_rate(rates)
            save_rates(rates)
        elif choice == '4':
            convert_currency(rates)
        elif choice == '5':
            view_all_rates(rates)
        elif choice == '6':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
