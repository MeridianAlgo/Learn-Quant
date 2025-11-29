# Lists - Python List Operations for Financial Data

## 📋 Overview

This utility provides comprehensive Python list operations essential for financial data processing, algorithmic trading, and data manipulation. Lists are flexible containers that can store heterogeneous data types and are fundamental to Python programming.

## 🎯 Key Concepts

### **Python Lists**
- **Ordered Collections**: Maintain insertion order
- **Mutable**: Can be modified after creation
- **Dynamic Size**: Grow and shrink as needed
- **Heterogeneous**: Store different data types

### **Financial Applications**
- **Price Tickers**: Lists of stock symbols, asset identifiers
- **Transaction Logs**: Order history, trade records
- **Time Series**: Historical price data, returns series
- **Portfolio Holdings**: Asset lists, position tracking

### **List Operations**
- **CRUD Operations**: Create, Read, Update, Delete elements
- **Sorting & Searching**: Efficient data organization
- **Filtering & Mapping**: Data transformation
- **Stack/Queue Operations**: LIFO/FIFO behaviors

## 💻 Implementation

### **Basic List Operations**
```python
# Create and manipulate lists
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']  # Stock symbols
prices = [150.25, 2800.50, 350.75, 3200.00]  # Corresponding prices

# Add elements
tickers.append('TSLA')        # Add to end
tickers.insert(0, 'NVDA')     # Insert at index

# Remove elements
tickers.remove('GOOGL')       # Remove specific value
del tickers[2]                # Remove by index
last_ticker = tickers.pop()   # Remove and return last element
```

### **Advanced List Operations**
```python
# List comprehensions for financial calculations
prices = [100, 102, 98, 105, 107]
returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]

# Filter operations
high_prices = [price for price in prices if price > 100]
volatile_assets = [ticker for ticker, price in zip(tickers, prices) if price > 1000]

# Sorting with custom keys
portfolio = [('AAPL', 150, 0.3), ('GOOGL', 2800, 0.7)]
sorted_by_price = sorted(portfolio, key=lambda x: x[1])
sorted_by_weight = sorted(portfolio, key=lambda x: x[2], reverse=True)
```

### **List Performance Considerations**
```python
# Efficient list operations
import time

# Pre-allocate for known sizes
large_list = [0] * 1000000

# Use list comprehensions instead of loops
squares = [x**2 for x in range(1000)]  # Faster than append in loop

# Avoid repeated list concatenation (creates new lists)
# Bad:
# result = []
# for i in range(1000):
#     result = result + [i]

# Good:
result = [i for i in range(1000)]
```

## 📊 Examples

### **Example 1: Portfolio Management with Lists**
```python
class PortfolioManager:
    def __init__(self):
        self.holdings = []  # List of (ticker, shares, price) tuples
        self.transactions = []  # List of transaction records

    def add_position(self, ticker: str, shares: int, price: float):
        """Add a new position to the portfolio."""
        self.holdings.append((ticker, shares, price))
        self.transactions.append({
            'type': 'BUY',
            'ticker': ticker,
            'shares': shares,
            'price': price,
            'timestamp': time.time()
        })

    def remove_position(self, ticker: str, shares: int):
        """Remove shares from a position."""
        for i, (t, s, p) in enumerate(self.holdings):
            if t == ticker:
                if s <= shares:
                    # Remove entire position
                    del self.holdings[i]
                else:
                    # Reduce position size
                    self.holdings[i] = (t, s - shares, p)

                self.transactions.append({
                    'type': 'SELL',
                    'ticker': ticker,
                    'shares': shares,
                    'timestamp': time.time()
                })
                break

    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        return sum(shares * price for _, shares, price in self.holdings)

    def get_positions_by_value(self) -> list:
        """Get positions sorted by current value."""
        positions_with_value = []
        for ticker, shares, price in self.holdings:
            current_value = shares * price
            positions_with_value.append((ticker, shares, price, current_value))

        return sorted(positions_with_value, key=lambda x: x[3], reverse=True)

# Usage
portfolio = PortfolioManager()
portfolio.add_position('AAPL', 100, 150.25)
portfolio.add_position('GOOGL', 10, 2800.50)
portfolio.add_position('MSFT', 50, 350.75)

print(f"Portfolio value: ${portfolio.get_portfolio_value():.2f}")
print("Positions by value:", portfolio.get_positions_by_value())
```

### **Example 2: Transaction Processing**
```python
def process_transactions(transactions: list) -> dict:
    """
    Process a list of financial transactions.

    Args:
        transactions: List of transaction dictionaries

    Returns:
        dict: Summary statistics
    """
    summary = {
        'total_transactions': len(transactions),
        'buy_transactions': 0,
        'sell_transactions': 0,
        'total_volume': 0,
        'total_value': 0,
        'unique_tickers': set(),
        'transactions_by_type': {}
    }

    for tx in transactions:
        tx_type = tx['type']
        volume = tx.get('shares', 0)
        value = tx.get('price', 0) * volume

        summary['total_volume'] += volume
        summary['total_value'] += value
        summary['unique_tickers'].add(tx.get('ticker', 'UNKNOWN'))

        # Count by type
        if tx_type in summary['transactions_by_type']:
            summary['transactions_by_type'][tx_type] += 1
        else:
            summary['transactions_by_type'][tx_type] = 1

        # Specific counters
        if tx_type == 'BUY':
            summary['buy_transactions'] += 1
        elif tx_type == 'SELL':
            summary['sell_transactions'] += 1

    summary['unique_tickers'] = list(summary['unique_tickers'])
    return summary

# Sample transactions
transactions = [
    {'type': 'BUY', 'ticker': 'AAPL', 'shares': 100, 'price': 150.25},
    {'type': 'BUY', 'ticker': 'GOOGL', 'shares': 10, 'price': 2800.50},
    {'type': 'SELL', 'ticker': 'AAPL', 'shares': 50, 'price': 155.00},
    {'type': 'BUY', 'ticker': 'MSFT', 'shares': 75, 'price': 350.75},
]

summary = process_transactions(transactions)
print("Transaction Summary:")
for key, value in summary.items():
    print(f"{key}: {value}")
```

### **Example 3: Algorithmic Trading with Lists**
```python
class SimpleTradingStrategy:
    def __init__(self, prices: list, window: int = 20):
        """
        Simple moving average crossover strategy.

        Args:
            prices: List of historical prices
            window: Moving average window size
        """
        self.prices = prices
        self.window = window
        self.signals = []  # List of trading signals

    def calculate_moving_average(self, start_idx: int) -> float:
        """Calculate moving average for given window."""
        if start_idx < self.window - 1:
            return None
        return sum(self.prices[start_idx - self.window + 1:start_idx + 1]) / self.window

    def generate_signals(self) -> list:
        """Generate buy/sell signals based on MA crossover."""
        self.signals = []

        for i in range(len(self.prices)):
            ma_current = self.calculate_moving_average(i)

            if ma_current is None:
                self.signals.append('HOLD')
                continue

            if i > 0:
                ma_previous = self.calculate_moving_average(i - 1)
                if ma_previous is None:
                    self.signals.append('HOLD')
                    continue

                # Generate signals
                if self.prices[i] > ma_current and self.prices[i-1] <= ma_previous:
                    self.signals.append('BUY')
                elif self.prices[i] < ma_current and self.prices[i-1] >= ma_previous:
                    self.signals.append('SELL')
                else:
                    self.signals.append('HOLD')
            else:
                self.signals.append('HOLD')

        return self.signals

    def backtest(self, initial_capital: float = 10000) -> dict:
        """Backtest the trading strategy."""
        self.generate_signals()

        capital = initial_capital
        shares = 0
        portfolio_values = [capital]

        for i in range(1, len(self.prices)):
            signal = self.signals[i]

            if signal == 'BUY' and capital > 0:
                # Buy as many shares as possible
                shares_to_buy = capital // self.prices[i]
                if shares_to_buy > 0:
                    shares += shares_to_buy
                    capital -= shares_to_buy * self.prices[i]

            elif signal == 'SELL' and shares > 0:
                # Sell all shares
                capital += shares * self.prices[i]
                shares = 0

            # Calculate portfolio value
            portfolio_value = capital + shares * self.prices[i]
            portfolio_values.append(portfolio_value)

        return {
            'final_capital': capital,
            'final_shares': shares,
            'total_return': (portfolio_values[-1] - initial_capital) / initial_capital,
            'portfolio_values': portfolio_values
        }

# Usage
prices = [100, 102, 98, 105, 107, 110, 108, 112, 115, 118,
          120, 122, 119, 125, 128, 130, 127, 132, 135, 138]

strategy = SimpleTradingStrategy(prices)
results = strategy.backtest()

print(f"Initial capital: $10,000")
print(f"Final capital: ${results['final_capital']:.2f}")
print(f"Total return: {results['total_return']:.2%}")
print(f"Signals generated: {len(strategy.signals)}")
```

## 🧪 Testing

Run the test suite to verify functionality:
```bash
python -m pytest tests/test_lists.py -v
```

## 📚 References

- [Python Lists Documentation](https://docs.python.org/3/tutorial/datastructures.html)
- [List Comprehensions](https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions)
- [Sorting Techniques](https://docs.python.org/3/howto/sorting.html)
- [Algorithmic Trading with Python](https://www.oreilly.com/library/view/python-for-algorithmic/9781492053347/)

## 🎓 Learning Path

### **Prerequisites**
- Basic Python programming
- Understanding of financial markets

### **Next Steps**
- **Dictionaries**: Key-value data structures for financial data
- **Sets**: Unique collections for asset tracking
- **Tuples**: Immutable data for financial records

### **Assessment**
1. Implement a function that calculates portfolio diversification using list operations
2. Create a transaction logger that maintains ordered transaction history
3. Build a simple algorithmic trading strategy using list-based indicators

---

*This utility demonstrates the power of Python lists in financial applications. Master list operations to handle complex financial data efficiently.*
