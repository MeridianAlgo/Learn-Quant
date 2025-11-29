"""Python Control Flow Tutorial for Beginners.

Run with:
    python control_flow_tutorial.py

This module teaches conditionals (if/elif/else), loops (for/while), 
and list comprehensions with finance-focused examples.
"""

from decimal import Decimal
from typing import List, Dict


def intro() -> None:
    """Print orientation details for the learner."""
    print("\n" + "#" * 60)
    print("PYTHON BASICS – CONTROL FLOW WALKTHROUGH")
    print("#" * 60)
    print("We'll cover if/elif/else, for loops, while loops, and comprehensions.\n")


def conditional_statements() -> None:
    """Demonstrate if/elif/else statements."""
    print("=" * 60)
    print("CONDITIONAL STATEMENTS (if/elif/else)")
    print("=" * 60)
    
    # Example 1: Risk assessment based on volatility
    volatility = 0.25
    
    if volatility < 0.15:
        risk_level = "Low"
    elif volatility < 0.30:
        risk_level = "Medium"
    else:
        risk_level = "High"
    
    print(f"Volatility: {volatility:.2%}")
    print(f"Risk Level: {risk_level}")
    
    # Example 2: Portfolio rebalancing check
    target_allocation = 0.60
    current_allocation = 0.58
    tolerance = 0.05
    
    difference = abs(current_allocation - target_allocation)
    
    if difference > tolerance:
        action = "REBALANCE NEEDED"
    else:
        action = "No action required"
    
    print(f"\nTarget Allocation: {target_allocation:.2%}")
    print(f"Current Allocation: {current_allocation:.2%}")
    print(f"Tolerance: {tolerance:.2%}")
    print(f"Action: {action}")
    
    # Example 3: Nested conditionals for trade decision
    price = 150.50
    moving_average = 148.00
    volume = 1_500_000
    avg_volume = 1_000_000
    
    if price > moving_average:
        if volume > avg_volume:
            signal = "STRONG BUY"
        else:
            signal = "WEAK BUY"
    else:
        if volume > avg_volume:
            signal = "STRONG SELL"
        else:
            signal = "WEAK SELL"
    
    print(f"\nPrice: ${price:.2f}, MA: ${moving_average:.2f}")
    print(f"Volume: {volume:,}, Avg Volume: {avg_volume:,}")
    print(f"Signal: {signal}")


def for_loops() -> None:
    """Demonstrate for loops with iterations."""
    print("\n" + "=" * 60)
    print("FOR LOOPS")
    print("=" * 60)
    
    # Example 1: Iterate through portfolio holdings
    portfolio = {
        "AAPL": 50,
        "GOOGL": 20,
        "MSFT": 30,
        "TSLA": 15
    }
    
    print("Portfolio Holdings:")
    for ticker, shares in portfolio.items():
        print(f"  {ticker}: {shares} shares")
    
    # Example 2: Calculate cumulative returns
    monthly_returns = [0.02, -0.01, 0.03, 0.015, -0.005, 0.025]
    
    print("\nMonthly Returns:")
    cumulative_return = 1.0
    for month, ret in enumerate(monthly_returns, 1):
        cumulative_return *= (1 + ret)
        print(f"  Month {month}: {ret:+.2%} (Cumulative: {cumulative_return - 1:.2%})")
    
    # Example 3: Range iteration for compound interest
    principal = Decimal("1000.00")
    annual_rate = Decimal("0.05")
    
    print(f"\nCompound Interest on ${principal} at {annual_rate:.1%}:")
    for year in range(1, 6):
        balance = principal * (1 + annual_rate) ** year
        print(f"  Year {year}: ${balance:.2f}")


def while_loops() -> None:
    """Demonstrate while loops."""
    print("\n" + "=" * 60)
    print("WHILE LOOPS")
    print("=" * 60)
    
    # Example 1: Find when investment doubles
    initial_investment = 1000.0
    annual_return = 0.08
    target = initial_investment * 2
    
    years = 0
    current_value = initial_investment
    
    print(f"Initial Investment: ${initial_investment:.2f}")
    print(f"Annual Return: {annual_return:.1%}")
    print(f"Target: ${target:.2f}\n")
    
    while current_value < target:
        years += 1
        current_value *= (1 + annual_return)
        print(f"Year {years}: ${current_value:.2f}")
    
    print(f"\nInvestment doubles in {years} years!")
    
    # Example 2: Stop loss monitoring
    entry_price = 100.0
    stop_loss_pct = 0.05
    stop_price = entry_price * (1 - stop_loss_pct)
    
    # Simulated price movements
    prices = [99.5, 98.0, 97.5, 96.0, 94.5]
    
    print(f"\nStop Loss Monitoring:")
    print(f"Entry Price: ${entry_price:.2f}")
    print(f"Stop Loss Price: ${stop_price:.2f}\n")
    
    i = 0
    triggered = False
    
    while i < len(prices) and not triggered:
        current_price = prices[i]
        print(f"Price update: ${current_price:.2f}", end="")
        
        if current_price <= stop_price:
            print(" - STOP LOSS TRIGGERED!")
            triggered = True
        else:
            print(" - Holding...")
        
        i += 1


def list_comprehensions() -> None:
    """Demonstrate list comprehensions for concise iterations."""
    print("\n" + "=" * 60)
    print("LIST COMPREHENSIONS")
    print("=" * 60)
    
    # Example 1: Calculate percentage changes
    prices = [100, 102, 98, 101, 105, 103]
    
    # Traditional approach
    pct_changes_traditional = []
    for i in range(1, len(prices)):
        change = (prices[i] - prices[i-1]) / prices[i-1]
        pct_changes_traditional.append(change)
    
    # List comprehension approach (more concise!)
    pct_changes = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
    
    print("Price changes using list comprehension:")
    for i, change in enumerate(pct_changes, 1):
        print(f"  Day {i}: {change:+.2%}")
    
    # Example 2: Filter profitable trades
    trades = [
        {"symbol": "AAPL", "pnl": 150.50},
        {"symbol": "GOOGL", "pnl": -75.25},
        {"symbol": "MSFT", "pnl": 200.00},
        {"symbol": "TSLA", "pnl": -50.00},
        {"symbol": "AMZN", "pnl": 125.75}
    ]
    
    # Get only profitable trades
    profitable_trades = [trade for trade in trades if trade["pnl"] > 0]
    
    print("\nProfitable Trades:")
    for trade in profitable_trades:
        print(f"  {trade['symbol']}: ${trade['pnl']:.2f}")
    
    # Example 3: Create a multiplication table for position sizing
    base_position = 100
    multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]
    
    position_sizes = [int(base_position * mult) for mult in multipliers]
    
    print(f"\nPosition Sizing (Base: {base_position} shares):")
    for mult, size in zip(multipliers, position_sizes):
        print(f"  {mult}x: {size} shares")
    
    # Example 4: Dictionary comprehension for portfolio values
    holdings = {"AAPL": 50, "GOOGL": 20, "MSFT": 30}
    current_prices = {"AAPL": 175.50, "GOOGL": 140.25, "MSFT": 380.75}
    
    portfolio_values = {
        ticker: holdings[ticker] * current_prices[ticker] 
        for ticker in holdings
    }
    
    print("\nPortfolio Values:")
    for ticker, value in portfolio_values.items():
        print(f"  {ticker}: ${value:,.2f}")
    
    total_value = sum(portfolio_values.values())
    print(f"\nTotal Portfolio Value: ${total_value:,.2f}")


def break_continue_statements() -> None:
    """Demonstrate break and continue statements."""
    print("\n" + "=" * 60)
    print("BREAK AND CONTINUE STATEMENTS")
    print("=" * 60)
    
    # Example 1: Break when target profit reached
    daily_pnl = [50, 75, 120, 200, 150, 90, 60]
    target_profit = 500
    
    cumulative_pnl = 0
    days_to_target = 0
    
    print(f"Target Profit: ${target_profit:.2f}\n")
    
    for day, pnl in enumerate(daily_pnl, 1):
        cumulative_pnl += pnl
        days_to_target = day
        print(f"Day {day}: +${pnl:.2f} (Total: ${cumulative_pnl:.2f})")
        
        if cumulative_pnl >= target_profit:
            print(f"\n✓ Target reached in {days_to_target} days!")
            break
    
    # Example 2: Skip losing days with continue
    print("\n" + "-" * 60)
    all_trades = [
        {"day": 1, "pnl": 100},
        {"day": 2, "pnl": -50},
        {"day": 3, "pnl": 150},
        {"day": 4, "pnl": -25},
        {"day": 5, "pnl": 200}
    ]
    
    print("Profitable Days Only:")
    for trade in all_trades:
        if trade["pnl"] < 0:
            continue  # Skip losing days
        print(f"  Day {trade['day']}: ${trade['pnl']:.2f}")


def nested_loops() -> None:
    """Demonstrate nested loops for multi-dimensional data."""
    print("\n" + "=" * 60)
    print("NESTED LOOPS")
    print("=" * 60)
    
    # Example: Correlation matrix between assets
    tickers = ["AAPL", "GOOGL", "MSFT"]
    
    # Simulated correlation coefficients
    correlations = [
        [1.00, 0.65, 0.72],
        [0.65, 1.00, 0.58],
        [0.72, 0.58, 1.00]
    ]
    
    print("Correlation Matrix:")
    print("        ", end="")
    for ticker in tickers:
        print(f"{ticker:>8}", end="")
    print()
    
    for i, ticker in enumerate(tickers):
        print(f"{ticker:>8}", end="")
        for j in range(len(tickers)):
            print(f"{correlations[i][j]:>8.2f}", end="")
        print()


def main() -> None:
    """Run all control flow examples."""
    intro()
    conditional_statements()
    for_loops()
    while_loops()
    list_comprehensions()
    break_continue_statements()
    nested_loops()
    print("\n🎉 Control Flow tutorial complete! Practice with loops and conditionals to build powerful trading logic.")


if __name__ == "__main__":
    main()
