<p class="lq-badges"><span class="lq-badge lq-beginner">Beginner</span><span class="lq-badge lq-cat">Data Structures</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Data Structures - Tuples and Sets"
    python "tuples_sets_tutorial.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Data%20Structures%20-%20Tuples%20and%20Sets)

---
# Data Structures – Tuples and Sets

## Overview

Tuples and Sets are fundamental Python data structures that complement Lists and Dictionaries. Understanding when to use them is key to writing efficient, Pythonic code for financial applications.

## Key Concepts

### **Tuples `(item1, item2)`**
- **Immutable**: Cannot be changed after creation
- **Ordered**: Items keep their order
- **Faster**: Slightly more memory efficient than lists
- **Hashable**: Can be used as dictionary keys
- **Use cases**: Fixed records, return values, coordinates

### **Sets `{item1, item2}`**
- **Unique**: No duplicate elements allowed
- **Unordered**: No guaranteed order
- **Fast Lookups**: O(1) membership testing
- **Math Operations**: Union, intersection, difference
- **Use cases**: Removing duplicates, membership testing, filtering

## Key Examples

### Tuples
```python
# Fixed record (Ticker, Price, Shares)
trade = ("AAPL", 150.50, 100)

# Unpacking
ticker, price, shares = trade

# Returning multiple values
def get_range():
 return 10.0, 20.0 # Returns a tuple

low, high = get_range()
```

### Sets
```python
# Unique collection
tickers = {"AAPL", "GOOGL", "AAPL", "MSFT"}
print(tickers) # {'AAPL', 'MSFT', 'GOOGL'}

# Fast lookup
if "AAPL" in tickers:
 print("Found!")
```

### Set Operations
```python
portfolio_a = {"AAPL", "GOOGL", "MSFT"}
portfolio_b = {"MSFT", "AMZN", "TSLA"}

# Intersection (In both)
both = portfolio_a & portfolio_b # {'MSFT'}

# Union (In either)
all_stocks = portfolio_a | portfolio_b

# Difference (In A but not B)
only_a = portfolio_a - portfolio_b
```

## Files
- `tuples_sets_tutorial.py`: Interactive tutorial with examples

## How to Run
```bash
python tuples_sets_tutorial.py
```

## Financial Applications

### 1. Trade Records (Tuples)
Store immutable trade execution details that shouldn't change:
```python
execution = (order_id, timestamp, symbol, price, quantity)
```

### 2. Watchlist Management (Sets)
Maintain a list of unique symbols to monitor:
```python
watchlist.add("AAPL") # Won't add duplicate if already exists
```

### 3. Portfolio Reconciliation (Set Ops)
Compare expected vs actual holdings:
```python
expected_holdings = {"AAPL", "GOOGL"}
actual_holdings = {"AAPL", "MSFT"}

missing = expected_holdings - actual_holdings # {'GOOGL'}
unexpected = actual_holdings - expected_holdings # {'MSFT'}
```

### 4. Correlation Keys (Tuples)
Use tuples as dictionary keys for pair data:
```python
correlations = {
 ("AAPL", "MSFT"): 0.75,
 ("GOOGL", "AMZN"): 0.82
}
```

## Best Practices

- Use **Tuples** for heterogeneous data (different types) that belongs together (like a struct).
- Use **Lists** for homogeneous data (same type) that may change size.
- Use **Sets** when order doesn't matter and uniqueness is required.
- Use **FrozenSets** if you need an immutable set (e.g., as a dict key).

---

*Master tuples and sets to write cleaner, faster, and more robust financial code!*

---

## Continue in Data Structures

<div class="grid cards" markdown>

-   :material-database-outline: __[Data Structures - Arrays](Data Structures - Arrays.md)__

    Welcome to the comprehensive guide to NumPy arrays! This utility is designed to help both beginners and experienced Python programmers master array operations for data analysis, scientific computing, and quantitative finance.

-   :material-database-outline: __[Data Structures - Dictionaries](Data Structures - Dictionaries.md)__

    This utility provides comprehensive Python dictionary operations essential for financial data organization, lookup tables, and key-value mappings. Dictionaries are the backbone of feature engineering and data lookup in quantitative finance.

-   :material-database-outline: __[Data Structures - Lists](Data Structures - Lists.md)__

    Lists are Python's **most fundamental data structure**—ordered, mutable collections used for storing time series data, portfolio holdings, transaction logs, and any sequence of values. Master list operations and you unlock efficient data processing essential for trading systems and quantitative analysis.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
