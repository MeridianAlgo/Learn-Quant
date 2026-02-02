# -----------------------------
# Technical Indicators Calculator Utility (NO API)
# -----------------------------
# This script lets you calculate technical indicators for learning.
# All calculations are done locally (no APIs, no real market data).
# All code is commented for beginners to learn Python and finance.
# -----------------------------

import csv


# Helper function: Get prices from user input (comma-separated)
def input_prices():
    s = input("Enter prices separated by commas: ")
    return [float(x.strip()) for x in s.split(",") if x.strip()]


# Helper function: Load prices from a CSV file (one price per line)
def load_prices_csv():
    fname = input("Enter CSV filename: ").strip()
    prices = []
    try:
        with open(fname) as f:
            reader = csv.reader(f)
            for row in reader:
                for val in row:
                    try:
                        prices.append(float(val))
                    except ValueError:
                        continue
        print(f"Loaded {len(prices)} prices from {fname}.")
    except Exception as e:
        print(f"Error loading file: {e}")
    return prices


# Calculate Simple Moving Average (SMA)
def sma(prices, window):
    return [
        sum(prices[i - window + 1 : i + 1]) / window if i >= window - 1 else None
        for i in range(len(prices))
    ]


# Calculate Exponential Moving Average (EMA)
def ema(prices, window):
    ema_vals = []
    k = 2 / (window + 1)
    for i, price in enumerate(prices):
        if i == 0:
            ema_vals.append(price)
        else:
            ema_vals.append(price * k + ema_vals[-1] * (1 - k))
    return [v if i >= window - 1 else None for i, v in enumerate(ema_vals)]


# Calculate Relative Strength Index (RSI)
def rsi(prices, period=14):
    gains = [0]
    losses = [0]
    for i in range(1, len(prices)):
        change = prices[i] - prices[i - 1]
        gains.append(max(change, 0))
        losses.append(-min(change, 0))
    rsis = [None] * period
    for i in range(period, len(prices)):
        avg_gain = sum(gains[i - period + 1 : i + 1]) / period
        avg_loss = sum(losses[i - period + 1 : i + 1]) / period
        if avg_loss == 0:
            rsis.append(100)
        else:
            rs = avg_gain / avg_loss
            rsis.append(100 - 100 / (1 + rs))
    return rsis


# Calculate MACD (Moving Average Convergence Divergence)
def macd(prices, fast=12, slow=26, signal=9):
    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    macd_line = [
        f - s if f is not None and s is not None else None
        for f, s in zip(ema_fast, ema_slow)
    ]
    signal_line = ema([x for x in macd_line if x is not None], signal)
    # Pad signal_line to align with macd_line
    signal_line = [None] * (len(macd_line) - len(signal_line)) + signal_line
    hist = [
        m - s if m is not None and s is not None else None
        for m, s in zip(macd_line, signal_line)
    ]
    return macd_line, signal_line, hist


# Print a simple text-based plot of indicator values
def print_plot(values, label):
    print(f"\n{label}:")
    for i, v in enumerate(values):
        if v is not None:
            bar = "*" * int(abs(v) // 1)
            print(f"{i:>3}: {v:>8.2f} {bar}")


# Print the main menu for the CLI
def print_menu():
    print("\nTechnical Indicators Calculator Menu:")
    print("1. Enter prices manually")
    print("2. Load prices from CSV")
    print("3. Calculate SMA")
    print("4. Calculate EMA")
    print("5. Calculate RSI")
    print("6. Calculate MACD")
    print("7. Exit")


# Helper to get an integer from user input
def input_int(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Invalid input. Please enter an integer.")


# Main workflow for the CLI
def main():
    print(
        """
====================================
Welcome to the Technical Indicators Calculator!
This tool helps you learn Python and technical analysis by calculating indicators.
- No APIs or real market data are used.
- All code is commented for beginners.
====================================
"""
    )
    prices = []
    while True:
        print_menu()
        choice = input("Enter your choice: ").strip()
        if choice == "1":
            prices = input_prices()
        elif choice == "2":
            prices = load_prices_csv()
        elif choice == "3":
            if not prices:
                print("No prices loaded.")
                continue
            window = input_int("Enter SMA window: ")
            vals = sma(prices, window)
            print_plot(vals, f"SMA({window})")
        elif choice == "4":
            if not prices:
                print("No prices loaded.")
                continue
            window = input_int("Enter EMA window: ")
            vals = ema(prices, window)
            print_plot(vals, f"EMA({window})")
        elif choice == "5":
            if not prices:
                print("No prices loaded.")
                continue
            period = input_int("Enter RSI period (default 14): ")
            vals = rsi(prices, period)
            print_plot(vals, f"RSI({period})")
        elif choice == "6":
            if not prices:
                print("No prices loaded.")
                continue
            macd_line, signal_line, hist = macd(prices)
            print_plot(macd_line, "MACD Line")
            print_plot(signal_line, "Signal Line")
            print_plot(hist, "Histogram")
        elif choice == "7":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
