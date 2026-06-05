<p class="lq-badges"><span class="lq-badge lq-beginner">Beginner</span><span class="lq-badge lq-cat">Data Structures</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Data Structures - Lists"
    python "lists.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Data%20Structures%20-%20Lists)

---
# Data Structures – Lists

## Overview

Lists are Python's **most fundamental data structure**—ordered, mutable collections used for storing time series data, portfolio holdings, transaction logs, and any sequence of values. Master list operations and you unlock efficient data processing essential for trading systems and quantitative analysis.

**Why lists matter in finance**:
- Store price histories (time series)
- Track portfolio positions and transactions
- Build datasets for backtesting
- Store signals and indicators
- Process market data feeds

## Learning Objectives

After this module, you'll:

- **Create, access, and modify lists** efficiently
- **Use methods** like `append`, `extend`, `insert`, `remove`, `pop`
- **Slice lists** to extract subsets of data
- **Sort and search** lists for analysis
- **Choose between lists, tuples, dicts** for different data problems

## Core Concepts

### 1. List Basics

Lists are **ordered, mutable collections** that can hold any data type.

```python
# Create lists
tickers = ['AAPL', 'MSFT', 'TSLA']       # Strings
prices = [150.25, 300.50, 250.75]       # Floats
returns = [0.02, -0.01, 0.03]           # Mixed floats
portfolio = [
    ('AAPL', 100, 150.25),              # Tuples (immutable)
    ('MSFT', 50, 300.50)
]

# Empty list
empty = []

# List with mixed types (valid but not recommended)
mixed = [1, 'AAPL', 150.25, True]       # Avoid this!
```

**Key properties**:
- **Ordered**: Index 0 is always first
- **Mutable**: Can add/remove/change elements
- **Dynamic**: Grows and shrinks as needed
- **Heterogeneous**: Can hold different types (usually don't)

### 2. Accessing Elements

```python
prices = [100, 102, 98, 105, 103]

# Index (0-based)
first = prices[0]           # 100
second = prices[1]          # 102
last = prices[-1]           # 103
second_last = prices[-2]    # 105

# Slice: list[start:end:step]
# Note: end is exclusive!
first_three = prices[0:3]   # [100, 102, 98]
every_other = prices[::2]   # [100, 98, 103]
reversed_list = prices[::-1]  # [103, 105, 98, 102, 100]

# Length
count = len(prices)          # 5
```

**Finance example: Get yesterday's price**

```python
price_history = [100, 102, 98, 105, 103]
today = price_history[-1]           # 103 (current)
yesterday = price_history[-2]       # 105 (previous)
daily_change = today - yesterday    # -2
```

### 3. Adding Elements

```python
tickers = ['AAPL', 'MSFT']

# append: Add single element to end
tickers.append('TSLA')
# Result: ['AAPL', 'MSFT', 'TSLA']

# extend: Add multiple elements
tickers.extend(['GE', 'IBM'])
# Result: ['AAPL', 'MSFT', 'TSLA', 'GE', 'IBM']

# insert: Add at specific position
tickers.insert(1, 'GOOGL')
# Result: ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'GE', 'IBM']

# NOTE: Use extend for lists, not append!
# BAD:
list1 = [1, 2]
list1.append([3, 4])
# Result: [1, 2, [3, 4]] - nested!

# GOOD:
list1 = [1, 2]
list1.extend([3, 4])
# Result: [1, 2, 3, 4]
```

### 4. Removing Elements

```python
tickers = ['AAPL', 'MSFT', 'TSLA', 'GE']

# remove: Remove first occurrence of value
tickers.remove('MSFT')
# Result: ['AAPL', 'TSLA', 'GE']

# pop: Remove and return element
last = tickers.pop()        # Returns 'GE', list becomes ['AAPL', 'TSLA']
second = tickers.pop(1)     # Returns 'TSLA', list becomes ['AAPL']

# del: Delete by index
del tickers[0]              # Remove 'AAPL'

# clear: Remove all
tickers.clear()             # Result: []

# NOTE: Be careful with remove()!
prices = [100, 102, 100, 105]
prices.remove(100)          # Only removes FIRST 100
# Result: [102, 100, 105]
```

### 5. Searching and Counting

```python
prices = [100, 102, 98, 105, 103, 98]

# in: Check membership
if 100 in prices:
    print("Price 100 exists")

# index: Find first position
pos = prices.index(102)     # Returns 1
pos = prices.index(98)      # Returns 2 (first occurrence)

# count: Count occurrences
count = prices.count(98)    # Returns 2 (appears twice)

# Get all indices of a value
value = 98
indices = [i for i, p in enumerate(prices) if p == value]
# Result: [2, 5]
```

### 6. Sorting

```python
prices = [100, 102, 98, 105, 103]
tickers = ['AAPL', 'MSFT', 'TSLA', 'GE', 'IBM']

# Sort in place (modifies original)
prices.sort()
# Result: [98, 100, 102, 103, 105]

# Reverse sort
prices.sort(reverse=True)
# Result: [105, 103, 102, 100, 98]

# Sort strings alphabetically
tickers.sort()
# Result: ['AAPL', 'GE', 'IBM', 'MSFT', 'TSLA']

# sorted(): Create new sorted list (doesn't modify original)
original = [3, 1, 4, 1, 5]
sorted_list = sorted(original)
# original: [3, 1, 4, 1, 5], sorted_list: [1, 1, 3, 4, 5]

# Sort with key function (sort by second element)
portfolio = [('AAPL', 100), ('MSFT', 50), ('TSLA', 200)]
by_count = sorted(portfolio, key=lambda x: x[1])
# Result: [('MSFT', 50), ('AAPL', 100), ('TSLA', 200)]

# Sort by value, descending
by_value = sorted(portfolio, key=lambda x: x[1], reverse=True)
# Result: [('TSLA', 200), ('AAPL', 100), ('MSFT', 50)]
```

### 7. Useful Methods

```python
prices = [100, 102, 98, 105, 103]

# min/max
lowest = min(prices)        # 98
highest = max(prices)       # 105

# sum
total = sum(prices)         # 508

# average
avg = sum(prices) / len(prices)  # 101.6

# reverse
prices_reversed = list(reversed(prices))
# Or in place:
prices.reverse()
```

## Common Finance Patterns

### Pattern 1: Time Series Processing

```python
# Last N prices
prices = [100, 102, 101, 105, 103, 108, 106]
lookback = 3
recent = prices[-lookback:]  # [108, 106, 107] — last 3

# Calculate moving average
window = 5
ma = sum(prices[-window:]) / window

# Price changes
changes = [prices[i+1] - prices[i] for i in range(len(prices)-1)]
```

### Pattern 2: Portfolio Management

```python
# Track holdings as list of tuples
portfolio = [
    ('AAPL', 100, 150.25),   # (ticker, shares, price)
    ('MSFT', 50, 300.50),
    ('TSLA', 200, 250.75)
]

# Calculate portfolio value
total_value = sum(shares * price for _, shares, price in portfolio)

# Find largest position by value
largest = max(portfolio, key=lambda x: x[1] * x[2])

# Filter positions above threshold
large_positions = [p for p in portfolio if p[1] * p[2] > 10000]
```

### Pattern 3: Transaction Log

```python
# Track all trades
transactions = [
    {'type': 'BUY', 'ticker': 'AAPL', 'shares': 100, 'price': 150},
    {'type': 'BUY', 'ticker': 'MSFT', 'shares': 50, 'price': 300},
    {'type': 'SELL', 'ticker': 'AAPL', 'shares': 50, 'price': 155}
]

# Count buy vs sell
buys = len([t for t in transactions if t['type'] == 'BUY'])
sells = len([t for t in transactions if t['type'] == 'SELL'])

# Total shares bought
total_bought = sum(t['shares'] for t in transactions if t['type'] == 'BUY')
```

### Pattern 4: Backtesting

```python
prices = [100, 102, 98, 105, 103, 108, 106, 110]

# Simple SMA crossover
sma_period = 3
equity_curve = []
position = 0  # 0 = no position, 1 = long

for i in range(sma_period, len(prices)):
    sma = sum(prices[i-sma_period:i]) / sma_period
    
    # Buy signal
    if prices[i] > sma and position == 0:
        position = 1
        entry_price = prices[i]
    
    # Sell signal
    if prices[i] < sma and position == 1:
        position = 0
        profit = prices[i] - entry_price
    
    equity_curve.append(prices[i])
```

## Performance Considerations

| Operation | Time | Notes |
|-----------|------|-------|
| **Access by index** | O(1) | `list[0]` is instant |
| **Append** | O(1) | Add to end is fast |
| **Insert at start** | O(n) | Shifts all elements |
| **Remove by value** | O(n) | Must search first |
| **Sort** | O(n log n) | Uses Timsort algorithm |
| **Search/index** | O(n) | Linear scan |

**Best practice**: For large datasets, avoid removing from start/middle. Use comprehensions instead of repeated appends.

```python
# SLOW: Many appends
result = []
for x in huge_list:
    if x > threshold:
        result.append(x)

# FAST: One comprehension
result = [x for x in huge_list if x > threshold]
```

## Comparison: Lists vs Other Data Structures

| Structure | Ordered | Mutable | Use Case |
|-----------|---------|---------|----------|
| **List** | Yes | Yes | Time series, flexible data |
| **Tuple** | Yes | No | Immutable records, dict keys |
| **Set** | No | Yes | Unique values, deduplication |
| **Dict** | Yes* | Yes | Key-value lookup, portfolios |

*Dicts maintain insertion order since Python 3.7

**Choose list when**: Storing sequences, need to modify, care about order
**Choose tuple when**: Data shouldn't change, need hashable (for set/dict key)
**Choose dict when**: Looking up by key/name (like `portfolio['AAPL']`)

## Files

- **`lists_tutorial.py`**: Interactive examples with finance use cases

## How to Run

```bash
python lists_tutorial.py
```

## Practice Problems

### Problem 1: Calculate Daily Returns
```python
prices = [100, 102, 101, 105, 103]

# Calculate list of daily returns (percentage changes)
# Expected: [0.02, -0.0098, 0.0396, -0.0191]
```

### Problem 2: Filter High Prices
```python
prices = {'AAPL': 150, 'GE': 80, 'MSFT': 300, 'TSLA': 250}
threshold = 200

# Find all stocks above threshold
# Expected: ['MSFT', 'TSLA']
```

### Problem 3: Portfolio Rebalancing
```python
portfolio = [('AAPL', 10, 150), ('MSFT', 5, 300), ('TSLA', 2, 250)]

# Calculate total portfolio value
# Expected: $3900
```

### Problem 4: Find Extremes
```python
prices = [100, 102, 98, 105, 103, 108, 101, 110]

# Find highest and lowest prices
# Find dates of highest and lowest
```

### Problem 5: Remove Duplicates
```python
tickers = ['AAPL', 'MSFT', 'AAPL', 'TSLA', 'MSFT']

# Remove duplicates while maintaining order
# Expected: ['AAPL', 'MSFT', 'TSLA']
```

## Learning Path

**Prerequisites**:
- [Python Basics – Control Flow](Python Basics - Control Flow.md)
- [Python Basics – Functions](Python Basics - Functions.md)

**Builds into**:
- [Data Structures – Dictionaries](Data Structures - Dictionaries.md)
- [Python Basics – Comprehensions](Python Basics - Comprehensions.md)
- [Python Basics – Pandas](Python Basics - Pandas.md) (lists become DataFrames)
- [Data Processing](Data Processing.md)

## FAQ

**Q: List or tuple?**
A: List for modifiable sequences. Tuple for fixed records or when you need hashability (dict key).

**Q: Why is list.remove() dangerous?**
A: It only removes the FIRST occurrence. If duplicates exist, you might leave data.

**Q: Append vs extend?**
A: `append()` adds one item (nests lists). `extend()` adds multiple items (flattens).

**Q: Performance: list or NumPy array?**
A: Lists for small/mixed data. NumPy arrays for large numerical data (100x faster).

**Q: Can I modify a list while iterating?**
A: Dangerous! Use comprehension or iterate over a copy instead.

```python
# WRONG
for item in list1:
    if item > 5:
        list1.remove(item)  # Modifies while iterating!

# RIGHT
list1 = [x for x in list1 if x <= 5]
```

## Further Reading

- Python docs: https://docs.python.org/3/tutorial/datastructures.html#more-on-lists
- Sorting guide: https://docs.python.org/3/howto/sorting.html
- List comprehensions: https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions


---

## Continue in Data Structures

<div class="grid cards" markdown>

-   :material-database-outline: __[Data Structures - Arrays](Data Structures - Arrays.md)__

    Welcome to the comprehensive guide to NumPy arrays! This utility is designed to help both beginners and experienced Python programmers master array operations for data analysis, scientific computing, and quantitative finance.

-   :material-database-outline: __[Data Structures - Dictionaries](Data Structures - Dictionaries.md)__

    This utility provides comprehensive Python dictionary operations essential for financial data organization, lookup tables, and key-value mappings. Dictionaries are the backbone of feature engineering and data lookup in quantitative finance.

-   :material-database-outline: __[Data Structures - Tuples and Sets](Data Structures - Tuples and Sets.md)__

    Tuples and Sets are fundamental Python data structures that complement Lists and Dictionaries. Understanding when to use them is key to writing efficient, Pythonic code for financial applications.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
