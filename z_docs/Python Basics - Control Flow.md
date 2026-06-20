<p class="lq-badges"><span class="lq-badge lq-beginner">Beginner</span><span class="lq-badge lq-cat">Python Fundamentals</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Python Basics - Control Flow"
    python "control_flow_tutorial.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Python%20Basics%20-%20Control%20Flow)

---
# Python Basics – Control Flow

## Overview

Control flow structures (`if/elif/else`, `for`, `while`, comprehensions, `break`, `continue`) are the foundation of all algorithms. This module teaches how to make decisions, iterate through data, and build the logic patterns used in trading systems, backtests, and risk management tools.

**Why this matters**: Every trading strategy, every market model, every risk control system is built on control flow. Master these structures and you can implement any algorithm.

## Learning Objectives

After this module, you'll:

- **Use `if/elif/else`** to create decision trees and conditional logic
- **Iterate with `for` and `while`** loops in different contexts
- **Combine conditions** with logical operators (`and`, `or`, `not`)
- **Exit early** or skip iterations with `break` and `continue`
- **Recognize patterns** and choose the right structure for each problem

## Core Concepts

### 1. Conditional Statements (if/elif/else)

Make decisions based on conditions. Each condition is a **boolean** (True or False).

```python
if condition:
    # Do this if condition is True
elif other_condition:
    # Do this if first condition was False but this one is True
else:
    # Do this if all conditions were False
```

**Finance example: Risk level classification**

```python
volatility = 0.25

if volatility < 0.15:
    risk_level = "Low"          # Safe, boring stocks
elif volatility < 0.30:
    risk_level = "Medium"       # Normal stocks
elif volatility < 0.50:
    risk_level = "High"         # Volatile, growth stocks
else:
    risk_level = "Extreme"      # Meme stocks, extreme risk
```

**Key comparison operators:**

| Operator | Meaning | Example |
|----------|---------|---------|
| `==` | Equal | `price == 100` |
| `!=` | Not equal | `ticker != 'TSLA'` |
| `>` | Greater than | `portfolio_value > 100000` |
| `<` | Less than | `loss < -5000` |
| `>=` | Greater or equal | `return >= 0.05` |
| `<=` | Less or equal | `volatility <= 0.30` |

### 2. Logical Operators

Combine multiple conditions.

```python
# AND: Both must be True
if price > 100 and volume > 1000000:
    print("Good liquidity, good price")

# OR: At least one must be True
if rsi > 70 or price > resistance_level:
    print("Sell signal")

# NOT: Reverse the condition
if not position_closed:
    print("Position still open")
```

**Finance use case:**

```python
# Complex trading rule
if (price > moving_average_50 and 
    volume > avg_volume and 
    not trend_is_down):
    signal = "BUY"
else:
    signal = "HOLD"
```

### 3. For Loops

Repeat code for each item in a sequence.

**Basic form:**

```python
for item in sequence:
    # Process item
```

**Common patterns:**

```python
# Iterate through list
prices = [100, 102, 98, 101]
for price in prices:
    print(price)

# Iterate with index
for i in range(len(prices)):
    print(f"Price {i}: {prices[i]}")

# Better: iterate with enumerate
for index, price in enumerate(prices):
    print(f"Price {index}: {price}")

# Iterate through dictionary
portfolio = {"AAPL": 50, "MSFT": 20, "TSLA": 10}
for ticker, shares in portfolio.items():
    print(f"{ticker}: {shares} shares")

# Generate number sequence
for i in range(5):  # 0, 1, 2, 3, 4
    print(i)

# Iterate with step
for i in range(0, 100, 10):  # 0, 10, 20, ..., 90
    print(i)
```

**Finance example: Calculate returns**

```python
prices = [100, 102, 101, 105, 103]
returns = []

for i in range(1, len(prices)):
    daily_return = (prices[i] - prices[i-1]) / prices[i-1]
    returns.append(daily_return)
```

### 4. While Loops

Repeat code **until a condition becomes False**.

```python
while condition:
    # Do this while condition is True
```

**Use case: Process until condition met**

```python
# Find how many years to reach $1M with 7% annual return
investment = 100000
target = 1000000
years = 0
rate = 0.07

while investment < target:
    investment = investment * (1 + rate)
    years += 1

print(f"Reached $1M in {years} years")
```

**WARNING: Infinite loops are dangerous!**

```python
# WRONG: This runs forever
while True:
    print("Help me")

# RIGHT: Add an exit condition
max_tries = 10
tries = 0
while tries < max_tries:
    if condition_met:
        break  # Exit loop
    tries += 1
```

### 5. Break and Continue

Control loop flow.

**`break`**: Exit the loop immediately

```python
prices = [100, 102, 98, 101, 105]
stop_loss = 99

for price in prices:
    if price < stop_loss:
        print("Stop loss triggered!")
        break  # Exit loop
    print(f"Price: {price}")

# Output:
# Price: 100
# Price: 102
# Stop loss triggered!
# (loop ends early)
```

**`continue`**: Skip to next iteration

```python
returns = [0.02, -0.01, 0.03, -0.02, 0.01]
min_return = 0

for ret in returns:
    if ret < min_return:
        continue  # Skip negative returns
    print(f"Positive return: {ret:.2%}")

# Output:
# Positive return: 2.00%
# Positive return: 3.00%
# Positive return: 1.00%
```

### 6. Nested Loops

Loop inside a loop. Useful for matrices or multi-dimensional data.

```python
# Check all pairs of stocks for correlation
stocks = ['AAPL', 'MSFT', 'TSLA']

for stock1 in stocks:
    for stock2 in stocks:
        if stock1 != stock2:
            correlation = calculate_correlation(stock1, stock2)
            print(f"{stock1} vs {stock2}: {correlation:.2f}")
```

## Control Flow Patterns

### Pattern 1: Simple Validation
```python
# Check if trade is valid
trade_size = 1000
account_balance = 50000
max_risk_pct = 0.02

if trade_size > account_balance * max_risk_pct:
    print("Trade too large, rejected")
else:
    print("Trade accepted")
```

### Pattern 2: State Machine
```python
# Track trade lifecycle
position_state = "OPEN"

if position_state == "OPEN":
    if stop_loss_hit:
        position_state = "CLOSED_LOSS"
    elif profit_target_hit:
        position_state = "CLOSED_GAIN"
elif position_state == "CLOSED_LOSS":
    process_loss()
elif position_state == "CLOSED_GAIN":
    process_gain()
```

### Pattern 3: Accumulate with Loop
```python
# Calculate portfolio value
positions = [
    {"symbol": "AAPL", "shares": 10, "price": 150},
    {"symbol": "MSFT", "shares": 5, "price": 300},
    {"symbol": "TSLA", "shares": 2, "price": 250}
]

total_value = 0
for pos in positions:
    position_value = pos["shares"] * pos["price"]
    total_value += position_value
    print(f"{pos['symbol']}: ${position_value:,.0f}")

print(f"Portfolio total: ${total_value:,.0f}")
```

### Pattern 4: Find and Exit
```python
# Find first price where stop loss triggers
prices = [100, 102, 101, 98, 97, 96]
stop_loss = 99
exit_price = None

for price in prices:
    if price <= stop_loss:
        exit_price = price
        break

if exit_price:
    print(f"Exited at ${exit_price}")
```

## Common Finance Algorithms

### Algorithm 1: Portfolio Rebalancing
```python
portfolio = {"AAPL": 5000, "MSFT": 3000, "TSLA": 2000}
total = sum(portfolio.values())
target_pcts = {"AAPL": 0.50, "MSFT": 0.30, "TSLA": 0.20}
tolerance = 0.05

print("Rebalancing needed:")
for ticker, value in portfolio.items():
    current_pct = value / total
    target_pct = target_pcts[ticker]
    
    if abs(current_pct - target_pct) > tolerance:
        new_value = target_pct * total
        trades_needed = new_value - value
        print(f"{ticker}: {trades_needed:,.0f}")
```

### Algorithm 2: Risk Control
```python
positions = [
    {"ticker": "AAPL", "value": 5000, "beta": 1.2},
    {"ticker": "MSFT", "value": 3000, "beta": 1.1},
    {"ticker": "GE", "value": 2000, "beta": 0.9}
]

# Calculate portfolio beta
portfolio_value = sum(p["value"] for p in positions)
portfolio_beta = sum(p["value"] * p["beta"] for p in positions) / portfolio_value

print(f"Portfolio beta: {portfolio_beta:.2f}")

# Check for over-concentration
max_position_size = portfolio_value * 0.30
for pos in positions:
    if pos["value"] > max_position_size:
        print(f"WARNING: {pos['ticker']} is {pos['value']/portfolio_value:.1%} of portfolio")
```

### Algorithm 3: Backtest Simulation
```python
prices = [100, 102, 101, 105, 103, 108]
moving_avg_period = 3

for i in range(moving_avg_period, len(prices)):
    # Calculate moving average
    window = prices[i - moving_avg_period:i]
    ma = sum(window) / len(window)
    
    # Generate signal
    if prices[i] > ma:
        signal = "BUY"
    else:
        signal = "SELL"
    
    print(f"Price: {prices[i]}, MA: {ma:.1f}, Signal: {signal}")
```

## When to Use What

| Structure | Use When | Example |
|-----------|----------|---------|
| **if** | Single decision | Check if price hit limit |
| **if/elif** | Multiple options | Classify risk level |
| **for** | Iterate known count | Loop through list of prices |
| **while** | Iterate until condition | Compound interest until target |
| **break** | Exit early | Stop loss triggered |
| **continue** | Skip iteration | Skip negative values |
| **Comprehension** | Transform list | Calculate all returns |

## Files

- **`control_flow_tutorial.py`**: Interactive examples with finance use cases
  - Real decision trees
  - Loop patterns
  - Risk management logic

## How to Run

```bash
python control_flow_tutorial.py
```

## Practice Problems

### Problem 1: Risk Classification
```python
volatility = 0.35

# Write if/elif/else to classify as: Low, Medium, High, or Extreme
# Thresholds: <0.15 = Low, <0.30 = Medium, <0.50 = High, >=0.50 = Extreme
```

### Problem 2: Portfolio Iteration
```python
portfolio = {"AAPL": 10, "MSFT": 5, "TSLA": 3}
prices = {"AAPL": 150, "MSFT": 300, "TSLA": 250}

# Calculate and print each position value
# Expected output:
# AAPL: $1500
# MSFT: $1500
# TSLA: $750
```

### Problem 3: Compound Interest
```python
principal = 10000
rate = 0.06
target = 20000

# Use while loop to count years until target reached
# Expected: ~12 years
```

### Problem 4: Stop Loss Detection
```python
prices = [100, 102, 101, 105, 103, 98, 95]
stop_loss = 99

# Use for loop and break to find exit price
# Expected: First price below 99 is 98
```

### Problem 5: Filter Positions
```python
positions = [1000, 2500, 800, 5000, 1200]
min_size = 1000

# Use for loop to skip positions below minimum
# Print only positions >= $1000
```

## Performance Notes

- **List comprehension**: Faster than loops, use when transforming data
- **For loop**: Normal speed, most readable, use for complex logic
- **While loop**: Variable speed, watch for infinite loops
- **Break/continue**: No performance impact, but improves readability

## Common Mistakes

### Mistake 1: Off-by-one Errors with Range
```python
# WRONG: range(5) goes 0-4, so misses last price
prices = [100, 102, 101, 105, 103]
for i in range(len(prices)):
    if i < len(prices) - 1:  # Unnecessary check
        return_pct = (prices[i+1] - prices[i]) / prices[i]

# RIGHT: Slice or use range correctly
for i in range(len(prices) - 1):
    return_pct = (prices[i+1] - prices[i]) / prices[i]
```

### Mistake 2: Modifying While Loop Condition in Loop
```python
# WRONG: Exit condition might never be reached
while portfolio_value > 0:
    if market_crashes:
        # Forgot to break!
    else:
        portfolio_value -= daily_loss

# RIGHT: Explicit exit
while portfolio_value > 0 and not liquidated:
    if market_crashes:
        break
    portfolio_value -= daily_loss
```

### Mistake 3: Infinite Loop Hang
```python
# WRONG: Loop never exits
while price > stop_loss:
    # Forgot to update price!
    print("Waiting...")

# RIGHT: Update inside loop
while price > stop_loss:
    price = get_latest_price()  # Update!
    if price <= stop_loss:
        break
```

## Learning Path

**Prerequisites**:
- [Python Basics – Numbers](Python Basics - Numbers.md)
- [Python Basics – Strings](Python Basics - Strings.md)

**Builds into**:
- [Python Basics – Functions](Python Basics - Functions.md)
- [Python Basics – Comprehensions](Python Basics - Comprehensions.md)
- [Data Structures – Lists](Data Structures - Lists.md)
- [Technical Indicators](Technical Indicators.md) (implement first algorithms)

## FAQ

**Q: `for` or `while`?**
A: `for` when you know the count (iterate through list). `while` when you don't (until price hits level).

**Q: When do I use `break` vs `else`?**
A: `break` to exit early. Use `else` on the loop only if needed (runs if loop completes without break).

**Q: Can I nest loops?**
A: Yes, but be careful with performance. Nested loop is O(n²)—slow for large datasets.

**Q: What's the difference between `if` and `while`?**
A: `if` executes once. `while` executes repeatedly as long as condition is True.

**Q: How do I avoid infinite loops?**
A: Always have an exit condition that WILL eventually be met. Test locally before running on live data!

## Further Reading

- Python docs: https://docs.python.org/3/tutorial/controlflow.html
- Loop patterns: https://www.python.org/dev/peps/pep-0202/ (list comprehensions)


---

## Continue in Python Fundamentals

<div class="grid cards" markdown>

-   :material-language-python: __[Python Basics - Comprehensions](Python Basics - Comprehensions.md)__

    Comprehensions are Python's most elegant way to transform data—replacing loops with readable, performant one-liners. This module teaches **list, dict, set comprehensions**, **generator expressions**, and **functional tools** (`map`, `filter`, `reduce`, `accumulate`) used constantly in quantitative finance for data cleaning, signal generation, and portfolio calculations.

-   :material-language-python: __[Python Basics - Dates and Times](Python Basics - Dates and Times.md)__

    Markets run on a calendar, not a clock. Interest accrues over **days**, options

-   :material-language-python: __[Python Basics - Essential Libraries](Python Basics - Essential Libraries.md)__

    A working quant leans on a small set of libraries for almost everything. A few of

-   :material-language-python: __[Python Basics - Functions](Python Basics - Functions.md)__

    This utility teaches Python functions - the building blocks of modular, reusable code. Learn to write efficient trading algorithms and financial tools using proper function design.

-   :material-language-python: __[Python Basics - Imports and Modules](Python Basics - Imports and Modules.md)__

    Almost every Python program begins with a few import lines. An import is how you

-   :material-language-python: __[Python Basics - NumPy](Python Basics - NumPy.md)__

    Covers the NumPy primitives that appear in virtually every quant codebase — from vectorised return calculations to portfolio variance via the quadratic form. All examples use realistic financial data so the connection between the NumPy API and actual quant work is immediate.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
