# Python Basics – Control Flow Utility

## 📋 Overview

This utility teaches Python control flow structures essential for building trading algorithms and financial applications. Master conditionals, loops, and comprehensions to write efficient, readable code.

## 🎯 Concepts Covered

### **Conditional Statements**
- **if/elif/else**: Make decisions based on conditions
- **Nested conditionals**: Complex decision trees
- **Comparison operators**: `>`, `<`, `>=`, `<=`, `==`, `!=`
- **Logical operators**: `and`, `or`, `not`

### **For Loops**
- **Iterating sequences**: Lists, tuples, dictionaries
- **Range function**: Generate number sequences
- **Enumerate**: Get index and value
- **Dictionary iteration**: Keys, values, items

### **While Loops**
- **Condition-based loops**: Run until condition is False
- **Infinite loops**: Use with caution!
- **Loop control**: break, continue

### **List Comprehensions**
- **Concise syntax**: Replace loops with one-liners
- **Filtering**: Add conditionals to comprehensions
- **Dictionary comprehensions**: Create dictionaries efficiently
- **Performance**: Faster than traditional loops

### **Break and Continue**
- **break**: Exit loop early
- **continue**: Skip current iteration
- **Use cases**: Early termination, filtering

## 💻 Key Examples

### Risk Assessment with Conditionals
```python
volatility = 0.25

if volatility < 0.15:
    risk_level = "Low"
elif volatility < 0.30:
    risk_level = "Medium"
else:
    risk_level = "High"
```

### Portfolio Iteration
```python
portfolio = {"AAPL": 50, "GOOGL": 20, "MSFT": 30}

for ticker, shares in portfolio.items():
    print(f"{ticker}: {shares} shares")
```

### List Comprehensions for Returns
```python
prices = [100, 102, 98, 101, 105]
pct_changes = [(prices[i] - prices[i-1]) / prices[i-1] 
               for i in range(1, len(prices))]
```

## 📂 Files
- `control_flow_tutorial.py`: Interactive tutorial with finance examples

## 🚀 How to Run
```bash
python control_flow_tutorial.py
```

## 🧠 Practice Ideas

1. **Risk Management System**
   - Use conditionals to create a multi-level risk assessment
   - Include position size limits based on account balance

2. **Backtesting Loop**
   - Iterate through historical prices
   - Track entry/exit signals and P&L

3. **Portfolio Rebalancing**
   - Check if any allocation exceeds tolerance
   - Calculate required trades to rebalance

4. **Watchlist Filter**
   - Use list comprehension to filter stocks by criteria
   - (e.g., price > 50, volume > 1M, PE ratio < 20)

5. **Compound Interest Calculator**
   - Calculate balance year-by-year with a loop
   - Find how many years to reach a target amount

## 📚 Next Steps
- Move to `UTILS - Python Basics - Functions/` to learn function definitions
- Explore `UTILS - Data Structures - Lists/` for advanced list operations
- Apply control flow in `UTILS - Technical Indicators/` for real algorithms

## 💡 Financial Applications

### Trading Signals
```python
# Multi-condition signal generation
if price > moving_average_50 and volume > avg_volume:
    if rsi < 70:
        signal = "BUY"
    else:
        signal = "OVERBOUGHT"
else:
    signal = "HOLD"
```

### Stop Loss Monitoring
```python
# Check prices until stop loss triggered
while current_price > stop_loss_price and not position_closed:
    current_price = get_latest_price()
    if current_price <= stop_loss_price:
        close_position()
```

### Portfolio Analysis
```python
# Find all holdings above target allocation
overweight = [ticker for ticker, allocation in allocations.items() 
              if allocation > target_allocation[ticker]]
```

---

*Master control flow to build dynamic trading systems and sophisticated financial applications!*
