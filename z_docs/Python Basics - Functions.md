<p class="lq-badges"><span class="lq-badge lq-beginner">Beginner</span><span class="lq-badge lq-cat">Python Fundamentals</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Python Basics - Functions"
    python "functions_tutorial.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Python%20Basics%20-%20Functions)

---
# Python Basics – Functions Utility

## Overview

This utility teaches Python functions - the building blocks of modular, reusable code. Learn to write efficient trading algorithms and financial tools using proper function design.

## Concepts Covered

### **Function Basics**
- **Function definition**: `def` keyword and naming conventions
- **Parameters**: Positional and keyword arguments
- **Return values**: Single values, tuples, dictionaries
- **Default parameters**: Optional arguments with defaults
- **Type hints**: Improve code clarity and IDE support

### **Advanced Parameters**
- **`*args`**: Variable positional arguments
- **`**kwargs`**: Variable keyword arguments
- **Unpacking**: Spread operators for arguments
- **Combining**: Mix required, optional, *args, and **kwargs

### **Lambda Functions**
- **Anonymous functions**: One-line function expressions
- **Use with built-ins**: `map()`, `filter()`, `sorted()`
- **Closures**: Functions that capture variables
- **When to use**: Short, throwaway functions

### **Decorators**
- **Function wrappers**: Modify function behavior
- **Common patterns**: Timing, logging, validation
- **`@wraps`**: Preserve function metadata
- **Practical uses**: Cache results, retry logic

### **Scope and Lifetime**
- **Local variables**: Function-specific variables
- **Global variables**: Module-level variables
- **`nonlocal` keyword**: Nested function scope
- **Best practices**: Minimize global state

### **Recursive Functions**
- **Base case**: Stop condition
- **Recursive case**: Call function within itself
- **Use cases**: Tree traversal, factorial, compound interest
- **Stack limits**: Python recursion depth

## Key Examples

### Position Sizing Function
```python
def calculate_position_size(account_balance: float,
 risk_percent: float) -> float:
 """Calculate position size based on risk."""
 return account_balance * risk_percent

size = calculate_position_size(10000, 0.02) # $200 risk
```

### Multiple Return Values
```python
def analyze_trade(entry: float, exit: float, shares: int) -> Tuple[float, float]:
 """Return profit and return percentage."""
 profit = (exit - entry) * shares
 return_pct = (exit - entry) / entry
 return profit, return_pct

profit, ret_pct = analyze_trade(100, 105, 50)
```

### Variable Arguments
```python
def calculate_portfolio_value(*positions: float) -> float:
 """Sum any number of position values."""
 return sum(positions)

total = calculate_portfolio_value(1000, 2000, 1500, 3000)
```

### Lambda with Sorting
```python
portfolio = [{"ticker": "AAPL", "value": 5000}, ...]
sorted_portfolio = sorted(portfolio, key=lambda x: x["value"], reverse=True)
```

### Simple Decorator
```python
def timer(func):
 """Measure function execution time."""
 def wrapper(*args, **kwargs):
 start = time.time()
 result = func(*args, **kwargs)
 print(f"Took {time.time() - start:.4f}s")
 return result
 return wrapper

@timer
def backtest_strategy():
 # ... strategy code ...
 pass
```

## Files
- `functions_tutorial.py`: Comprehensive function tutorial

## How to Run
```bash
python functions_tutorial.py
```

## Practice Ideas

1. **Position Sizing Library**
 - Create functions for fixed fractional, Kelly criterion
 - Include validation and error handling

2. **Technical Indicator Functions**
 - Write functions for SMA, EMA, RSI
 - Use decorators for caching results

3. **Portfolio Analysis Module**
 - Functions for returns, volatility, Sharpe ratio
 - Return comprehensive dictionaries

4. **Order Builder**
 - Use **kwargs for flexible order creation
 - Support market, limit, stop orders

5. **Backtesting Framework**
 - Recursive function for sequential trades
 - Lambda functions for filtering signals

## Next Steps
- Move to `UTILS - Advanced Python - OOP/` for classes and objects
- Explore `UTILS - Advanced Python - Error Handling/` for robust code
- Apply in `UTILS - Technical Indicators/` for real implementations

## Best Practices

### Function Design
```python
 DO:
def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
 """
 Calculate annualized Sharpe ratio.

 Args:
 returns: List of periodic returns
 risk_free_rate: Annual risk-free rate (default: 2%)

 Returns:
 Annualized Sharpe ratio
 """
 # Clear, documented, single responsibility

 DON'T:
def calc(r, rf=0.02): # Unclear naming, no docs
 # Do multiple unrelated things
```

### Type Hints
```python
 DO:
def get_price(ticker: str) -> Optional[float]:
 """Fetch current price, return None if unavailable."""

 DON'T:
def get_price(ticker): # No type information
```

### DRY (Don't Repeat Yourself)
```python
 DO:
def calculate_return(start: float, end: float) -> float:
 return (end - start) / start

# Reuse in multiple places
daily_return = calculate_return(100, 102)
monthly_return = calculate_return(100, 110)

 DON'T:
daily_return = (102 - 100) / 100 # Repeated calculation
monthly_return = (110 - 100) / 100
```

## Common Pitfalls

### Mutable Default Arguments
```python
 WRONG:
def add_trade(trade, portfolio=[]): # Dangerous!
 portfolio.append(trade)
 return portfolio

 CORRECT:
def add_trade(trade, portfolio=None):
 if portfolio is None:
 portfolio = []
 portfolio.append(trade)
 return portfolio
```

### Global State
```python
 WRONG:
total_profit = 0 # Global state

def record_trade(profit):
 global total_profit # Avoid this
 total_profit += profit

 CORRECT:
def record_trade(profit, total_profit):
 return total_profit + profit

# Or use a class to encapsulate state
```

---

*Master functions to build modular, testable, and maintainable trading systems!*

---

## Continue in Python Fundamentals

<div class="grid cards" markdown>

-   :material-language-python: __[Python Basics - Comprehensions](Python Basics - Comprehensions.md)__

    Comprehensions are Python's most elegant way to transform data—replacing loops with readable, performant one-liners. This module teaches **list, dict, set comprehensions**, **generator expressions**, and **functional tools** (`map`, `filter`, `reduce`, `accumulate`) used constantly in quantitative finance for data cleaning, signal generation, and portfolio calculations.

-   :material-language-python: __[Python Basics - Control Flow](Python Basics - Control Flow.md)__

    Control flow structures (`if/elif/else`, `for`, `while`, comprehensions, `break`, `continue`) are the foundation of all algorithms. This module teaches how to make decisions, iterate through data, and build the logic patterns used in trading systems, backtests, and risk management tools.

-   :material-language-python: __[Python Basics - Dates and Times](Python Basics - Dates and Times.md)__

    Markets run on a calendar, not a clock. Interest accrues over **days**, options

-   :material-language-python: __[Python Basics - Essential Libraries](Python Basics - Essential Libraries.md)__

    A working quant leans on a small set of libraries for almost everything. A few of

-   :material-language-python: __[Python Basics - Imports and Modules](Python Basics - Imports and Modules.md)__

    Almost every Python program begins with a few import lines. An import is how you

-   :material-language-python: __[Python Basics - NumPy](Python Basics - NumPy.md)__

    Covers the NumPy primitives that appear in virtually every quant codebase — from vectorised return calculations to portfolio variance via the quadratic form. All examples use realistic financial data so the connection between the NumPy API and actual quant work is immediate.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
