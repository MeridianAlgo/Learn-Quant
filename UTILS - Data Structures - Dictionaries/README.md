# Dictionaries - Key-Value Data Structures for Financial Analysis

## 📋 Overview

This utility provides comprehensive Python dictionary operations essential for financial data organization, lookup tables, and key-value mappings. Dictionaries are the backbone of feature engineering and data lookup in quantitative finance.

## 🎯 Key Concepts

### **Python Dictionaries**
- **Key-Value Pairs**: Fast lookups by unique keys
- **Mutable**: Can be modified after creation
- **Unordered**: No guaranteed order (Python 3.7+ maintains insertion order)
- **Hash Tables**: O(1) average-case lookup complexity

### **Financial Applications**
- **Asset Lookups**: Ticker symbols to company data
- **Portfolio Mapping**: Account IDs to holdings
- **Market Data**: Dates to price information
- **Configuration**: Settings and parameters
- **Caching**: API responses and calculations

### **Dictionary Operations**
- **CRUD Operations**: Create, Read, Update, Delete key-value pairs
- **Nested Structures**: Multi-level data organization
- **Default Values**: Safe access with defaults
- **Dictionary Comprehensions**: Efficient data transformation

## 💻 Implementation

### **Basic Dictionary Operations**
```python
# Create financial data dictionary
asset_data = {
    'AAPL': {
        'company': 'Apple Inc.',
        'sector': 'Technology',
        'price': 150.25,
        'market_cap': 2.5e12,
        'pe_ratio': 25.5,
        'dividend_yield': 0.6
    },
    'GOOGL': {
        'company': 'Alphabet Inc.',
        'sector': 'Technology',
        'price': 2800.50,
        'market_cap': 1.8e12,
        'pe_ratio': 28.2,
        'dividend_yield': 0.0
    }
}

# Access data
aapl_price = asset_data['AAPL']['price']
aapl_info = asset_data.get('AAPL', {})

# Add new asset
asset_data['MSFT'] = {
    'company': 'Microsoft Corporation',
    'sector': 'Technology',
    'price': 350.75,
    'market_cap': 2.2e12,
    'pe_ratio': 32.1,
    'dividend_yield': 0.8
}

# Update existing data
asset_data['AAPL']['price'] = 152.50
asset_data['AAPL']['volume'] = 50000000
```

### **Advanced Dictionary Operations**
```python
# Dictionary comprehensions
prices = {'AAPL': 150.25, 'GOOGL': 2800.50, 'MSFT': 350.75}
price_changes = {ticker: price * 0.01 for ticker, price in prices.items()}

# Nested dictionary creation
portfolio = {
    'account_001': {
        'holdings': {
            'AAPL': {'shares': 100, 'avg_cost': 145.50},
            'GOOGL': {'shares': 10, 'avg_cost': 2750.00}
        },
        'cash': 50000,
        'total_value': 0  # Will be calculated
    }
}

# Safe access with defaults
current_price = asset_data.get('TSLA', {}).get('price', 0.0)

# Merge dictionaries
market_data = {**prices, **{'TSLA': 850.25, 'AMZN': 3200.00}}
```

### **Performance Considerations**
```python
# Efficient dictionary operations
from collections import defaultdict

# Use defaultdict for automatic key initialization
sector_stocks = defaultdict(list)
for ticker, data in asset_data.items():
    sector_stocks[data['sector']].append(ticker)

# Dictionary views for memory efficiency
keys_view = asset_data.keys()      # Dynamic view of keys
values_view = asset_data.values()  # Dynamic view of values
items_view = asset_data.items()    # Dynamic view of items

# Memory-efficient iteration
for ticker in asset_data:  # Iterates over keys
    print(ticker)

for data in asset_data.values():  # Iterates over values
    print(data['company'])
```

## 📊 Examples

### **Example 1: Financial Database with Dictionaries**
```python
class FinancialDatabase:
    def __init__(self):
        self.assets = {}  # Main asset database
        self.price_history = {}  # Historical price data
        self.fundamentals = {}  # Fundamental data
        self.sector_index = {}  # Sector-based indexing

    def add_asset(self, ticker: str, company_data: dict):
        """Add or update asset information."""
        self.assets[ticker] = company_data

        # Update sector index
        sector = company_data.get('sector', 'Unknown')
        if sector not in self.sector_index:
            self.sector_index[sector] = []
        if ticker not in self.sector_index[sector]:
            self.sector_index[sector].append(ticker)

    def get_asset_info(self, ticker: str) -> dict:
        """Get comprehensive asset information."""
        asset = self.assets.get(ticker, {})
        if not asset:
            return {}

        # Combine with fundamentals if available
        fundamentals = self.fundamentals.get(ticker, {})
        price_history = self.price_history.get(ticker, [])

        return {
            **asset,
            'fundamentals': fundamentals,
            'recent_prices': price_history[-10:],  # Last 10 prices
            'avg_volume': sum(price_history[-20:]) / min(20, len(price_history)) if price_history else 0
        }

    def update_price(self, ticker: str, price: float, volume: int = 0):
        """Update price and volume data."""
        if ticker not in self.price_history:
            self.price_history[ticker] = []

        self.price_history[ticker].append({
            'price': price,
            'volume': volume,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

        # Keep only last 1000 entries for memory efficiency
        if len(self.price_history[ticker]) > 1000:
            self.price_history[ticker] = self.price_history[ticker][-1000:]

    def get_sector_peers(self, ticker: str) -> list:
        """Get peer companies in the same sector."""
        asset = self.assets.get(ticker, {})
        sector = asset.get('sector', 'Unknown')

        if sector not in self.sector_index:
            return []

        return [t for t in self.sector_index[sector] if t != ticker]

    def calculate_sector_averages(self, sector: str) -> dict:
        """Calculate average metrics for a sector."""
        if sector not in self.sector_index:
            return {}

        tickers = self.sector_index[sector]
        metrics = {
            'avg_pe': 0,
            'avg_price': 0,
            'avg_market_cap': 0,
            'count': len(tickers)
        }

        total_pe = 0
        total_price = 0
        total_market_cap = 0
        count = 0

        for ticker in tickers:
            asset = self.assets.get(ticker, {})
            if asset.get('pe_ratio'):
                total_pe += asset['pe_ratio']
                count += 1
            total_price += asset.get('price', 0)
            total_market_cap += asset.get('market_cap', 0)

        if count > 0:
            metrics['avg_pe'] = total_pe / count
        metrics['avg_price'] = total_price / len(tickers)
        metrics['avg_market_cap'] = total_market_cap / len(tickers)

        return metrics

# Usage
db = FinancialDatabase()

# Add assets
db.add_asset('AAPL', {
    'company': 'Apple Inc.',
    'sector': 'Technology',
    'price': 150.25,
    'market_cap': 2.5e12,
    'pe_ratio': 25.5
})

db.add_asset('GOOGL', {
    'company': 'Alphabet Inc.',
    'sector': 'Technology',
    'price': 2800.50,
    'market_cap': 1.8e12,
    'pe_ratio': 28.2
})

# Update prices
db.update_price('AAPL', 152.00, 50000000)
db.update_price('GOOGL', 2820.00, 1500000)

# Get information
aapl_info = db.get_asset_info('AAPL')
print(f"AAPL Price: ${aapl_info['price']:.2f}")
print(f"Technology Sector Peers: {', '.join(db.get_sector_peers('AAPL'))}")

sector_avg = db.calculate_sector_averages('Technology')
print(f"Tech Sector Avg P/E: {sector_avg['avg_pe']:.2f}")
    """
    Create comprehensive portfolio summary using dictionaries.

    Args:
        portfolio: Dictionary of holdings {ticker: shares}
        market_data: Dictionary of market data {ticker: price_info}

    Returns:
        dict: Portfolio summary
    """
    summary = {
        'total_value': 0,
        'total_shares': 0,
        'positions': {},
        'sector_allocation': {},
        'top_holdings': [],
        'diversification_score': 0
    }

    # Calculate position values
    for ticker, shares in portfolio.items():
        if ticker in market_data:
            price = market_data[ticker]['price']
            position_value = shares * price
            sector = market_data[ticker].get('sector', 'Unknown')

            summary['positions'][ticker] = {
                'shares': shares,
                'price': price,
                'value': position_value,
                'sector': sector,
                'weight': 0  # Will be calculated
            }

            # Update totals
            summary['total_value'] += position_value
            summary['total_shares'] += shares

            # Update sector allocation
            if sector in summary['sector_allocation']:
                summary['sector_allocation'][sector] += position_value
            else:
                summary['sector_allocation'][sector] = position_value

    # Calculate weights
    if summary['total_value'] > 0:
        for ticker in summary['positions']:
            summary['positions'][ticker]['weight'] = (
                summary['positions'][ticker]['value'] / summary['total_value']
            )

    # Get top holdings
    summary['top_holdings'] = sorted(
        summary['positions'].items(),
        key=lambda x: x[1]['value'],
        reverse=True
    )[:5]

    # Calculate diversification score (simplified)
    if summary['total_value'] > 0:
        sector_weights = list(summary['sector_allocation'].values())
        sector_weights = [w / summary['total_value'] for w in sector_weights]
        # Higher diversification score for more equal sector weights
        summary['diversification_score'] = 1 - sum((w - 1/len(sector_weights))**2
                                                   for w in sector_weights) / (2 / len(sector_weights))

    return summary

# Sample data
portfolio = {
    'AAPL': 100,
    'GOOGL': 10,
    'MSFT': 50,
    'JPM': 75,
    'JNJ': 25
}

market_data = {
    'AAPL': {'price': 150.25, 'sector': 'Technology'},
    'GOOGL': {'price': 2800.50, 'sector': 'Technology'},
    'MSFT': {'price': 350.75, 'sector': 'Technology'},
    'JPM': {'price': 125.50, 'sector': 'Financial'},
    'JNJ': {'price': 165.25, 'sector': 'Healthcare'}
}

summary = create_portfolio_summary(portfolio, market_data)
print(f"Portfolio Value: ${summary['total_value']:,.2f}")
top_holding_ticker, top_holding_data = summary['top_holdings'][0]
print(f"Top Holding: {top_holding_ticker} (${top_holding_data['value']:,.2f})")
print(f"Sector Allocation: {summary['sector_allocation']}")
```

### **Example 3: Risk Management Dictionary**
```python
def calculate_risk_metrics(price_data: dict) -> dict:
    """
    Calculate risk metrics using dictionary-based price data.

    Args:
        price_data: Dictionary of {ticker: [price1, price2, ...]}

    Returns:
        dict: Risk metrics for each asset
    """
    risk_metrics = {}

    for ticker, prices in price_data.items():
        if len(prices) < 2:
            continue

        # Calculate returns
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]

        # Basic statistics
        avg_return = sum(returns) / len(returns)
        variance = sum((r - avg_return)**2 for r in returns) / len(returns)
        volatility = variance ** 0.5

        # Value at Risk (95% confidence)
        sorted_returns = sorted(returns)
        var_95_idx = int(len(sorted_returns) * 0.05)
        var_95 = -sorted_returns[var_95_idx] if var_95_idx < len(sorted_returns) else 0

        # Maximum drawdown
        peak = prices[0]
        max_drawdown = 0
        current_drawdown = 0

        for price in prices:
            if price > peak:
                peak = price
                current_drawdown = 0
            else:
                current_drawdown = (peak - price) / peak
                max_drawdown = max(max_drawdown, current_drawdown)

        risk_metrics[ticker] = {
            'avg_return': avg_return,
            'volatility': volatility,
            'var_95': var_95,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': avg_return / volatility if volatility > 0 else 0,
            'num_observations': len(prices)
        }

    return risk_metrics

# Sample price data
price_data = {
    'AAPL': [100, 102, 98, 105, 107, 110, 108, 112, 115, 118],
    'GOOGL': [2800, 2820, 2780, 2850, 2870, 2900, 2880, 2920, 2950, 2980],
    'MSFT': [350, 355, 348, 360, 365, 370, 368, 375, 380, 385]
}

risk_metrics = calculate_risk_metrics(price_data)
for ticker, metrics in risk_metrics.items():
    print(f"{ticker}:")
    print(f"  Volatility: {metrics['volatility']".4f"}")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']".4f"}")
    print(f"  Max Drawdown: {metrics['max_drawdown']".4f"}")
    print()
```

## 🧪 Testing

Run the test suite to verify functionality:
```bash
python -m pytest tests/test_dictionaries.py -v
```

## 📚 References

- [Python Dictionaries](https://docs.python.org/3/tutorial/datastructures.html#dictionaries)
- [Dictionary Comprehensions](https://docs.python.org/3/tutorial/datastructures.html#dictionaries)
- [collections.defaultdict](https://docs.python.org/3/library/collections.html#collections.defaultdict)
- [Financial Data Structures](https://www.oreilly.com/library/view/python-for-finance/9781492024326/)

## 🎓 Learning Path

### **Prerequisites**
- Basic Python programming
- Understanding of financial data structures

### **Next Steps**
- **DataFrames**: Tabular data manipulation with pandas
- **JSON**: Data serialization and API integration
- **Caching**: Performance optimization techniques

### **Assessment**
1. Create a function that builds a financial database from CSV data
2. Implement a portfolio tracking system using nested dictionaries
3. Build a risk management dashboard with dictionary-based calculations

---

*This utility demonstrates the power of Python dictionaries in organizing and accessing financial data efficiently. Master dictionary operations for robust financial applications.*
