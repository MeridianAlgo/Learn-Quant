"""Python Functions Tutorial for Beginners.

Run with:
    python functions_tutorial.py

This module teaches function definitions, parameters, return values, 
*args/**kwargs, lambda functions, and decorators with finance examples.
"""

from typing import List, Dict, Tuple, Callable, Optional
from functools import wraps
import time


def intro() -> None:
    """Print orientation details for the learner."""
    print("\n" + "#" * 60)
    print("PYTHON BASICS – FUNCTIONS WALKTHROUGH")
    print("#" * 60)
    print("We'll cover function definitions, parameters, returns,")
    print("*args/**kwargs, lambda functions, and decorators.\n")


def basic_functions() -> None:
    """Demonstrate basic function definitions and calls."""
    print("=" * 60)
    print("BASIC FUNCTIONS")
    print("=" * 60)
    
    # Example 1: Simple function with no parameters
    def welcome_trader():
        """Display welcome message."""
        print("Welcome to the trading platform!")
    
    welcome_trader()
    
    # Example 2: Function with parameters
    def calculate_position_size(account_balance: float, risk_percent: float) -> float:
        """
        Calculate position size based on account balance and risk tolerance.
        
        Args:
            account_balance: Total account balance in dollars
            risk_percent: Percentage of account to risk (0-1)
            
        Returns:
            Dollar amount to risk on trade
        """
        return account_balance * risk_percent
    
    balance = 10000
    risk = 0.02
    position_size = calculate_position_size(balance, risk)
    print(f"\nAccount Balance: ${balance:,.2f}")
    print(f"Risk per Trade: {risk:.1%}")
    print(f"Position Size: ${position_size:,.2f}")
    
    # Example 3: Function with default parameters
    def calculate_profit(entry_price: float, exit_price: float, 
                        shares: int = 100, commission: float = 0.0) -> float:
        """
        Calculate profit from a trade.
        
        Args:
            entry_price: Entry price per share
            exit_price: Exit price per share
            shares: Number of shares (default: 100)
            commission: Commission per trade (default: 0.0)
            
        Returns:
            Net profit after commissions
        """
        gross_profit = (exit_price - entry_price) * shares
        total_commission = commission * 2  # Entry and exit
        return gross_profit - total_commission
    
    profit1 = calculate_profit(100, 105)
    profit2 = calculate_profit(100, 105, shares=50)
    profit3 = calculate_profit(100, 105, shares=100, commission=5.0)
    
    print(f"\nProfit (100 shares, no commission): ${profit1:.2f}")
    print(f"Profit (50 shares, no commission): ${profit2:.2f}")
    print(f"Profit (100 shares, $5 commission): ${profit3:.2f}")


def return_values() -> None:
    """Demonstrate different return value types."""
    print("\n" + "=" * 60)
    print("RETURN VALUES")
    print("=" * 60)
    
    # Example 1: Return single value
    def calculate_return(start_price: float, end_price: float) -> float:
        """Calculate simple return."""
        return (end_price - start_price) / start_price
    
    ret = calculate_return(100, 110)
    print(f"Simple Return: {ret:.2%}")
    
    # Example 2: Return multiple values (tuple)
    def calculate_metrics(prices: List[float]) -> Tuple[float, float, float]:
        """
        Calculate min, max, and average price.
        
        Returns:
            Tuple of (min_price, max_price, avg_price)
        """
        return min(prices), max(prices), sum(prices) / len(prices)
    
    prices = [100, 105, 98, 110, 102]
    min_price, max_price, avg_price = calculate_metrics(prices)
    
    print(f"\nPrice Statistics:")
    print(f"  Min: ${min_price:.2f}")
    print(f"  Max: ${max_price:.2f}")
    print(f"  Avg: ${avg_price:.2f}")
    
    # Example 3: Return dictionary
    def analyze_trade(entry: float, exit: float, shares: int) -> Dict[str, float]:
        """
        Analyze trade and return metrics.
        
        Returns:
            Dictionary with trade metrics
        """
        profit = (exit - entry) * shares
        return_pct = (exit - entry) / entry
        
        return {
            "profit": profit,
            "return_pct": return_pct,
            "entry_price": entry,
            "exit_price": exit,
            "shares": shares
        }
    
    trade_metrics = analyze_trade(100, 108, 50)
    
    print("\nTrade Analysis:")
    for metric, value in trade_metrics.items():
        if metric == "return_pct":
            print(f"  {metric}: {value:.2%}")
        elif metric in ["profit", "entry_price", "exit_price"]:
            print(f"  {metric}: ${value:.2f}")
        else:
            print(f"  {metric}: {value}")
    
    # Example 4: Optional return (None)
    def find_ticker(symbol: str, watchlist: List[str]) -> Optional[str]:
        """
        Find ticker in watchlist.
        
        Returns:
            Ticker if found, None otherwise
        """
        if symbol in watchlist:
            return symbol
        return None
    
    watchlist = ["AAPL", "GOOGL", "MSFT"]
    result1 = find_ticker("AAPL", watchlist)
    result2 = find_ticker("TSLA", watchlist)
    
    print(f"\nSearch for 'AAPL': {result1}")
    print(f"Search for 'TSLA': {result2}")


def args_and_kwargs() -> None:
    """Demonstrate *args and **kwargs for variable arguments."""
    print("\n" + "=" * 60)
    print("*ARGS AND **KWARGS")
    print("=" * 60)
    
    # Example 1: *args for variable positional arguments
    def calculate_portfolio_value(*positions: float) -> float:
        """
        Calculate total portfolio value from variable number of positions.
        
        Args:
            *positions: Variable number of position values
            
        Returns:
            Total portfolio value
        """
        return sum(positions)
    
    total1 = calculate_portfolio_value(1000, 2000, 1500)
    total2 = calculate_portfolio_value(1000, 2000, 1500, 3000, 2500)
    
    print(f"Portfolio Value (3 positions): ${total1:,.2f}")
    print(f"Portfolio Value (5 positions): ${total2:,.2f}")
    
    # Example 2: **kwargs for variable keyword arguments
    def create_trade_order(symbol: str, quantity: int, **kwargs) -> Dict:
        """
        Create trade order with optional parameters.
        
        Args:
            symbol: Stock ticker symbol
            quantity: Number of shares
            **kwargs: Optional parameters (limit_price, stop_price, etc.)
            
        Returns:
            Order dictionary
        """
        order = {
            "symbol": symbol,
            "quantity": quantity,
            "order_type": kwargs.get("order_type", "MARKET"),
            "time_in_force": kwargs.get("time_in_force", "DAY")
        }
        
        # Add limit price if provided
        if "limit_price" in kwargs:
            order["limit_price"] = kwargs["limit_price"]
        
        # Add stop price if provided
        if "stop_price" in kwargs:
            order["stop_price"] = kwargs["stop_price"]
        
        return order
    
    order1 = create_trade_order("AAPL", 100)
    order2 = create_trade_order("GOOGL", 50, order_type="LIMIT", limit_price=140.50)
    order3 = create_trade_order("MSFT", 75, order_type="STOP", stop_price=375.00, time_in_force="GTC")
    
    print("\nMarket Order:")
    print(f"  {order1}")
    
    print("\nLimit Order:")
    print(f"  {order2}")
    
    print("\nStop Order:")
    print(f"  {order3}")


def lambda_functions() -> None:
    """Demonstrate lambda (anonymous) functions."""
    print("\n" + "=" * 60)
    print("LAMBDA FUNCTIONS")
    print("=" * 60)
    
    # Example 1: Simple lambda for percentage change
    pct_change = lambda start, end: (end - start) / start
    
    change = pct_change(100, 105)
    print(f"Percentage Change: {change:.2%}")
    
    # Example 2: Lambda with sorted()
    portfolio = [
        {"ticker": "AAPL", "value": 5000},
        {"ticker": "GOOGL", "value": 3000},
        {"ticker": "MSFT", "value": 7000},
        {"ticker": "TSLA", "value": 2000}
    ]
    
    # Sort by value (descending)
    sorted_portfolio = sorted(portfolio, key=lambda x: x["value"], reverse=True)
    
    print("\nPortfolio sorted by value:")
    for holding in sorted_portfolio:
        print(f"  {holding['ticker']}: ${holding['value']:,}")
    
    # Example 3: Lambda with filter()
    prices = [45, 120, 85, 200, 30, 150]
    
    # Filter prices above $100
    high_prices = list(filter(lambda x: x > 100, prices))
    
    print(f"\nAll prices: {prices}")
    print(f"Prices > $100: {high_prices}")
    
    # Example 4: Lambda with map()
    daily_returns = [0.02, -0.01, 0.03, -0.015, 0.01]
    
    # Convert to percentages
    pct_returns = list(map(lambda x: f"{x:.2%}", daily_returns))
    
    print(f"\nDaily Returns:")
    for i, ret in enumerate(pct_returns, 1):
        print(f"  Day {i}: {ret}")


def decorators_intro() -> None:
    """Demonstrate basic decorators."""
    print("\n" + "=" * 60)
    print("DECORATORS")
    print("=" * 60)
    
    # Example 1: Simple timing decorator
    def timer_decorator(func: Callable) -> Callable:
        """Decorator to measure function execution time."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed = end_time - start_time
            print(f"  ⏱ {func.__name__}() took {elapsed:.4f} seconds")
            return result
        return wrapper
    
    @timer_decorator
    def calculate_portfolio_stats(prices: List[float]) -> Dict:
        """Calculate portfolio statistics."""
        # Simulate some processing
        time.sleep(0.1)
        return {
            "mean": sum(prices) / len(prices),
            "min": min(prices),
            "max": max(prices)
        }
    
    print("\nTiming decorator example:")
    stats = calculate_portfolio_stats([100, 105, 98, 110, 102])
    print(f"  Stats: {stats}")
    
    # Example 2: Validation decorator
    def validate_positive(func: Callable) -> Callable:
        """Decorator to validate that result is positive."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if result < 0:
                print(f"  ⚠ Warning: {func.__name__}() returned negative value: {result}")
            return result
        return wrapper
    
    @validate_positive
    def calculate_profit(buy_price: float, sell_price: float, shares: int) -> float:
        """Calculate profit from trade."""
        return (sell_price - buy_price) * shares
    
    print("\nValidation decorator example:")
    profit1 = calculate_profit(100, 105, 50)
    print(f"  Profitable trade: ${profit1:.2f}")
    
    profit2 = calculate_profit(100, 95, 50)
    print(f"  Losing trade: ${profit2:.2f}")


def scope_and_global() -> None:
    """Demonstrate variable scope."""
    print("\n" + "=" * 60)
    print("VARIABLE SCOPE")
    print("=" * 60)
    
    # Global variable
    trading_fee = 0.001  # 0.1% trading fee
    
    def calculate_net_profit(gross_profit: float) -> float:
        """Calculate profit after fees using global fee rate."""
        # Access global variable
        fee = abs(gross_profit) * trading_fee
        return gross_profit - fee
    
    gross = 1000
    net = calculate_net_profit(gross)
    
    print(f"Gross Profit: ${gross:.2f}")
    print(f"Trading Fee: {trading_fee:.1%}")
    print(f"Net Profit: ${net:.2f}")
    
    # Local variable scope
    def calculate_with_local_fee(gross_profit: float) -> float:
        """Calculate with locally defined fee."""
        local_fee = 0.002  # 0.2% fee (different from global)
        fee = abs(gross_profit) * local_fee
        return gross_profit - fee
    
    net_local = calculate_with_local_fee(gross)
    
    print(f"\nWith local fee (0.2%):")
    print(f"Net Profit: ${net_local:.2f}")


def recursive_functions() -> None:
    """Demonstrate recursive functions."""
    print("\n" + "=" * 60)
    print("RECURSIVE FUNCTIONS")
    print("=" * 60)
    
    def compound_interest_recursive(principal: float, rate: float, 
                                   years: int, current_year: int = 1) -> float:
        """
        Calculate compound interest recursively.
        
        Args:
            principal: Starting amount
            rate: Annual interest rate
            years: Number of years
            current_year: Current iteration (used internally)
            
        Returns:
            Final value after compound interest
        """
        # Base case
        if current_year > years:
            return principal
        
        # Recursive case
        new_principal = principal * (1 + rate)
        print(f"  Year {current_year}: ${new_principal:,.2f}")
        return compound_interest_recursive(new_principal, rate, years, current_year + 1)
    
    print("Recursive compound interest calculation:")
    final_value = compound_interest_recursive(1000, 0.08, 5)
    print(f"\nFinal Value: ${final_value:,.2f}")


def main() -> None:
    """Run all function examples."""
    intro()
    basic_functions()
    return_values()
    args_and_kwargs()
    lambda_functions()
    decorators_intro()
    scope_and_global()
    recursive_functions()
    print("\n🎉 Functions tutorial complete! Master functions to write modular, reusable code.")


if __name__ == "__main__":
    main()
