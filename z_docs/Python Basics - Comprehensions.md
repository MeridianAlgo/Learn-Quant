# Python Basics – Comprehensions

## Overview

Comprehensions are Python's most elegant way to transform data—replacing loops with readable, performant one-liners. This module teaches **list, dict, set comprehensions**, **generator expressions**, and **functional tools** (`map`, `filter`, `reduce`, `accumulate`) used constantly in quantitative finance for data cleaning, signal generation, and portfolio calculations.

**Why this matters**: Professional quant code uses comprehensions everywhere. They're faster, more readable, and less error-prone than loops.

## Learning Objectives

After this module, you'll:

- **Write list/dict/set comprehensions** to transform data without loops
- **Use generator expressions** for memory-efficient processing of large datasets
- **Apply `map` and `filter`** for functional data pipelines
- **Chain transformations** with `functools.reduce` and `itertools.accumulate` for finance calculations
- **Recognize** when each tool is the right choice vs. a loop

## Core Concepts

### 1. List Comprehensions
Transform a list into a new list using a concise one-liner.

**Syntax**: `[expression for item in iterable if condition]`

```python
# Loop version (verbose)
returns = []
for price_change in price_changes:
    returns.append(price_change / 100)

# Comprehension version (concise)
returns = [pc / 100 for pc in price_changes]

# With condition
positive_returns = [r for r in returns if r > 0]
```

**Finance use cases**:
- Normalize ticker symbols: `[t.upper() for t in tickers]`
- Filter prices above a threshold: `[p for p in prices if p > 100]`
- Calculate daily returns: `[(p2 - p1) / p1 for p1, p2 in zip(prices[:-1], prices[1:])]`

### 2. Dict Comprehensions
Build dictionaries from iterables efficiently.

**Syntax**: `{key_expr: value_expr for item in iterable if condition}`

```python
# Build ticker -> price dictionary
ticker_prices = {ticker: price for ticker, price in zip(tickers, prices)}

# Transform values
portfolio_pct = {ticker: value/total for ticker, value in portfolio.items()}

# Filter and transform
large_positions = {t: v for t, v in portfolio.items() if v > 10000}
```

### 3. Set Comprehensions
Remove duplicates and transform simultaneously.

**Syntax**: `{expression for item in iterable if condition}`

```python
# Unique sectors from all positions
sectors = {position['sector'] for position in portfolio}

# Formatted unique tickers (automatically deduplicated)
unique_symbols = {t.upper().strip() for t in raw_tickers}
```

### 4. Generator Expressions
Like list comprehensions, but lazy (only compute when needed). Ideal for huge datasets.

**Syntax**: `(expression for item in iterable if condition)`

```python
# List comprehension: compute all at once, uses memory
daily_returns = [log(prices[i+1]/prices[i]) for i in range(len(prices)-1)]

# Generator: compute on demand, saves memory
daily_returns_gen = (log(prices[i+1]/prices[i]) for i in range(len(prices)-1))

# Iterate through generator as needed
for ret in daily_returns_gen:
    print(ret)  # Computed only when accessed
```

**When to use**: Processing price histories with millions of rows where you don't need the full list in memory.

### 5. `map()` – Apply Function to Every Item

**Syntax**: `map(function, iterable)`

```python
# Convert string prices to floats
prices = ['100.50', '102.25', '98.75']
float_prices = list(map(float, prices))

# With lambda: apply formula to each return
returns = [0.01, -0.02, 0.015]
scaled_returns = list(map(lambda r: r * 100, returns))  # Convert to percentage
```

### 6. `filter()` – Keep Items Meeting a Condition

**Syntax**: `filter(function, iterable)`

```python
# Keep only positive returns
returns = [0.01, -0.02, 0.015, -0.01, 0.02]
gains = list(filter(lambda r: r > 0, returns))

# Keep tickers above a price threshold
prices = {'AAPL': 150, 'TSLA': 200, 'GE': 85}
expensive = list(filter(lambda x: x[1] > 100, prices.items()))
```

### 7. `functools.reduce()` – Combine All Items into One Result

**Syntax**: `reduce(function, iterable, initial_value)`

```python
from functools import reduce

# Compound daily returns to total return
# (1 + r1) × (1 + r2) × (1 + r3) - 1
returns = [0.01, 0.02, -0.01]
total_return = reduce(lambda x, r: x * (1 + r), returns, 1.0) - 1

# Portfolio total value (sum of positions)
positions = [5000, 3000, 2000]
portfolio_value = reduce(lambda x, y: x + y, positions)
```

### 8. `itertools.accumulate()` – Build Running Total

**Syntax**: `accumulate(iterable, function, initial=value)`

```python
from itertools import accumulate
import operator

# Build equity curve from returns
returns = [0.01, -0.02, 0.015]
equity_curve = list(accumulate([1.0] + returns, operator.mul))
# Result: [1.0, 1.01, 0.9899, 1.00495]

# Running P&L
daily_pnl = [-500, 1200, -300, 800]
running_total = list(accumulate(daily_pnl, operator.add))
# Result: [-500, 700, 400, 1200]
```

## Common Finance Patterns

### Pattern 1: Normalize Tickers
```python
raw_tickers = ['aapl', ' MSFT ', 'TSLA']
normalized = [t.upper().strip() for t in raw_tickers]
# Result: ['AAPL', 'MSFT', 'TSLA']
```

### Pattern 2: Calculate Daily Returns
```python
prices = [100, 102, 99.5, 101]
returns = [(prices[i+1] - prices[i]) / prices[i] for i in range(len(prices)-1)]
# Or with zip (cleaner):
returns = [(p2 - p1) / p1 for p1, p2 in zip(prices[:-1], prices[1:])]
```

### Pattern 3: Filter Strong Signals
```python
signals = [
    {'ticker': 'AAPL', 'strength': 0.85},
    {'ticker': 'MSFT', 'strength': 0.45},
    {'ticker': 'TSLA', 'strength': 0.92}
]
strong = [s for s in signals if s['strength'] > 0.8]
```

### Pattern 4: Build Correlation Matrix (Without NumPy)
```python
# For assets A, B, C with return histories
assets = ['A', 'B', 'C']
returns_data = {'A': [...], 'B': [...], 'C': [...]}

correlation_matrix = {
    (a1, a2): pearson_correlation(returns_data[a1], returns_data[a2])
    for a1 in assets for a2 in assets
}
```

## Files

- **`comprehensions_tutorial.py`**: Step-by-step walkthrough
  - Before/after comparisons (loop vs comprehension)
  - Real finance examples
  - Performance demonstrations
  - Nested comprehension examples

## How to Run

```bash
python comprehensions_tutorial.py
```

The tutorial is self-contained—no external dependencies beyond Python's standard library.

## Practice Problems

### Problem 1: Clean Ticker List
```python
# Input: tickers with duplicates, spaces, mixed case
raw = ['aapl', ' MSFT', 'AAPL', 'tsla ']

# Task: Normalize and deduplicate
# Expected: {'AAPL', 'MSFT', 'TSLA'}
```

### Problem 2: Filter and Transform Prices
```python
# Input: list of price dictionaries
prices = [
    {'ticker': 'AAPL', 'price': 150},
    {'ticker': 'GE', 'price': 80},
    {'ticker': 'MSFT', 'price': 300}
]

# Task: Keep only stocks above $100, return ticker list
# Expected: ['AAPL', 'MSFT']
```

### Problem 3: Compound Returns
```python
# Input: daily returns
returns = [0.01, -0.02, 0.015, 0.005]

# Task: Calculate total return using reduce
# Expected: ~0.00548 (0.548%)
```

### Problem 4: Build Portfolio Allocation
```python
# Input: positions and total value
positions = {'AAPL': 5000, 'MSFT': 3000, 'TSLA': 2000}
total = 10000

# Task: Create dict with ticker as key, percentage as value
# Expected: {'AAPL': 0.50, 'MSFT': 0.30, 'TSLA': 0.20}
```

## Performance Tip: Comprehensions vs Loops

Comprehensions are faster:

```python
# Loop: ~10ms for 1M items
result = []
for item in range(1000000):
    result.append(item * 2)

# Comprehension: ~6ms for 1M items
result = [item * 2 for item in range(1000000)]

# Generator: ~0ms (lazy evaluation)
result = (item * 2 for item in range(1000000))
```

**Use generators** for large datasets where you don't need the full list upfront.

## Comparison Table

| Tool | Use Case | Returns | Memory |
|------|----------|---------|--------|
| **List comp** | Transform list | List | All at once |
| **Generator** | Large dataset, streaming | Iterator | On demand |
| **Dict comp** | Build dictionary | Dict | All at once |
| **Set comp** | Unique + transform | Set | All at once |
| **`map()`** | Apply function | Iterator | On demand |
| **`filter()`** | Keep matching items | Iterator | On demand |
| **`reduce()`** | Combine to single value | Single value | Single value |
| **`accumulate()`** | Running total | Iterator | On demand |

## Learning Path

**Prerequisites**:
- [Python Basics – Numbers](../Python%20Basics%20-%20Numbers/)
- [Python Basics – Functions](../Python%20Basics%20-%20Functions/)

**Builds into**:
- [Python Basics – Pandas](../Python%20Basics%20-%20Pandas/) (DataFrames use similar patterns)
- [Data Processing](../Data%20Processing/) (cleaning and transforming real market data)
- [Strategies – Statistical Arbitrage](../Strategies%20-%20Statistical%20Arbitrage/) (signal generation)

## Common Mistakes

### Mistake 1: Readability Sacrifice
```python
# DON'T: Too nested, hard to read
result = [abs(x) for x in [y for y in data if y < 0]]

# DO: Use intermediate variables
negative = [y for y in data if y < 0]
result = [abs(x) for x in negative]
```

### Mistake 2: Using Comprehension When Loop is Clearer
```python
# Sometimes a loop is more readable:
portfolio = {}
for ticker, shares, price in transactions:
    portfolio[ticker] = portfolio.get(ticker, 0) + shares * price
```

### Mistake 3: Generator Exhaustion
```python
# WRONG: Generator can only be iterated once
gen = (x*2 for x in range(5))
list1 = list(gen)  # [0, 2, 4, 6, 8]
list2 = list(gen)  # [] - empty! Generator is exhausted

# RIGHT: Create new generator each time
gen1 = (x*2 for x in range(5))
gen2 = (x*2 for x in range(5))
```

## FAQ

**Q: List comprehension or map/filter?**
A: List comprehensions are more Pythonic and readable. Use `map`/`filter` when chaining multiple operations.

**Q: When should I use a generator?**
A: When processing very large datasets (millions of rows) where you can't afford to load everything into memory at once.

**Q: Should I always use comprehensions?**
A: Only if it's clearer than a loop. Single-line data transformation? Yes. Complex multi-step logic? Stick with a loop.

**Q: Can I nest comprehensions?**
A: Yes, but only 1-2 levels deep before it becomes unreadable. Complex nesting belongs in a loop.

## Further Reading

- Python docs: https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions
- Functional tools: https://docs.python.org/3/library/functools.html
- Itertools: https://docs.python.org/3/library/itertools.html
