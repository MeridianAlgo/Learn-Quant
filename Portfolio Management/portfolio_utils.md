# Portfolio Management Utilities

This module provides comprehensive portfolio management utilities for financial applications, including portfolio valuation, allocation analysis, rebalancing, performance tracking, and diversification metrics.

## Functions

### `calculate_portfolio_value(holdings: Dict[str, Dict[str, Any]], prices: Dict[str, float]) -> float`
Calculates total portfolio value.

**Parameters:**
- `holdings`: Dictionary of holdings {symbol: {"shares": int, "avg_cost": float}}
- `prices`: Dictionary of current prices {symbol: price}

**Returns:**
- Total portfolio value

**Example:**
```python
>>> holdings = {"AAPL": {"shares": 100, "avg_cost": 150.0}}
>>> prices = {"AAPL": 155.25}
>>> calculate_portfolio_value(holdings, prices)
15525.0
```

### `calculate_portfolio_allocation(holdings: Dict[str, Dict[str, Any]], prices: Dict[str, float]) -> Dict[str, float]`
Calculates portfolio allocation percentages.

**Parameters:**
- `holdings`: Dictionary of holdings
- `prices`: Dictionary of current prices

**Returns:**
- Dictionary of allocation percentages {symbol: percentage}

**Example:**
```python
>>> holdings = {"AAPL": {"shares": 100}, "GOOGL": {"shares": 50}}
>>> prices = {"AAPL": 150, "GOOGL": 2800}
>>> allocation = calculate_portfolio_allocation(holdings, prices)
>>> print(allocation["AAPL"])
51.72
```

### `calculate_portfolio_return(holdings: Dict[str, Dict[str, Any]], prices: Dict[str, float]) -> float`
Calculates total portfolio return.

**Parameters:**
- `holdings`: Dictionary of holdings with average costs
- `prices`: Dictionary of current prices

**Returns:**
- Portfolio return as percentage

**Example:**
```python
>>> holdings = {"AAPL": {"shares": 100, "avg_cost": 140.0}}
>>> prices = {"AAPL": 150.0}
>>> calculate_portfolio_return(holdings, prices)
7.14
```

### `rebalance_portfolio(target_allocation: Dict[str, float], holdings: Dict[str, Dict[str, Any]], prices: Dict[str, float], portfolio_value: float) -> Dict[str, Dict[str, float]]`
Calculates trades needed to rebalance portfolio.

**Parameters:**
- `target_allocation`: Target allocation percentages
- `holdings`: Current holdings
- `prices`: Current prices
- `portfolio_value`: Total portfolio value

**Returns:**
- Dictionary of required trades {symbol: {"action": "BUY"/"SELL", "shares": float}}

**Example:**
```python
>>> target = {"AAPL": 0.6, "GOOGL": 0.4}
>>> holdings = {"AAPL": {"shares": 50}, "GOOGL": {"shares": 50}}
>>> prices = {"AAPL": 150, "GOOGL": 2800}
>>> trades = rebalance_portfolio(target, holdings, prices, 155000)
>>> print(trades["AAPL"]["action"])
"BUY"
```

### `calculate_diversification_metrics(holdings: Dict[str, Dict[str, Any]], prices: Dict[str, float], sectors: Dict[str, str]) -> Dict[str, Any]`
Calculates portfolio diversification metrics.

**Parameters:**
- `holdings`: Portfolio holdings
- `prices`: Current prices
- `sectors`: Sector mapping {symbol: sector}

**Returns:**
- Diversification metrics dictionary

**Example:**
```python
>>> holdings = {"AAPL": {"shares": 100}, "MSFT": {"shares": 50}}
>>> prices = {"AAPL": 150, "MSFT": 250}
>>> sectors = {"AAPL": "Technology", "MSFT": "Technology"}
>>> metrics = calculate_diversification_metrics(holdings, prices, sectors)
>>> print(metrics["sector_concentration"]["Technology"])
100.0
```

### `calculate_portfolio_beta(holdings: Dict[str, Dict[str, Any]], prices: Dict[str, float], betas: Dict[str, float]) -> float`
Calculates portfolio beta.

**Parameters:**
- `holdings`: Portfolio holdings
- `prices`: Current prices
- `betas`: Stock betas {symbol: beta}

**Returns:**
- Portfolio beta

**Example:**
```python
>>> holdings = {"AAPL": {"shares": 100}, "MSFT": {"shares": 100}}
>>> prices = {"AAPL": 150, "MSFT": 250}
>>> betas = {"AAPL": 1.2, "MSFT": 0.9}
>>> portfolio_beta = calculate_portfolio_beta(holdings, prices, betas)
>>> print(f"Portfolio beta: {portfolio_beta:.2f}")
1.04
```

## Usage

```python
from portfolio_utils import (
    calculate_portfolio_value, calculate_portfolio_allocation,
    calculate_portfolio_return, rebalance_portfolio,
    calculate_diversification_metrics, calculate_portfolio_beta
)

# Define portfolio
holdings = {
    "AAPL": {"shares": 100, "avg_cost": 140.0},
    "GOOGL": {"shares": 20, "avg_cost": 2600.0},
    "MSFT": {"shares": 50, "avg_cost": 230.0}
}

# Get current prices
prices = {"AAPL": 150.0, "GOOGL": 2800.0, "MSFT": 250.0}

# Calculate portfolio metrics
portfolio_value = calculate_portfolio_value(holdings, prices)
allocation = calculate_portfolio_allocation(holdings, prices)
portfolio_return = calculate_portfolio_return(holdings, prices)

print(f"Portfolio Value: ${portfolio_value:,.2f}")
print(f"Portfolio Return: {portfolio_return:.2f}%")
print("Allocation:")
for symbol, pct in allocation.items():
    print(f"  {symbol}: {pct:.2f}%")

# Rebalance to target allocation
target_allocation = {"AAPL": 0.5, "GOOGL": 0.3, "MSFT": 0.2}
trades = rebalance_portfolio(target_allocation, holdings, prices, portfolio_value)

print("Rebalancing trades:")
for symbol, trade in trades.items():
    print(f"  {symbol}: {trade['action']} {trade['shares']:.2f} shares")

# Analyze diversification
sectors = {"AAPL": "Technology", "GOOGL": "Technology", "MSFT": "Technology"}
diversity = calculate_diversification_metrics(holdings, prices, sectors)
print(f"Sector concentration: {diversity['sector_concentration']}")
```

## Installation

No additional dependencies required. Uses only Python standard library.

## Testing

Run the module directly to see demonstrations:

```bash
python portfolio_utils.py
```

## Common Use Cases

- **Portfolio Analysis**: Calculate current value, returns, and allocations
- **Rebalancing**: Determine trades needed to maintain target allocation
- **Risk Management**: Calculate portfolio beta and diversification metrics
- **Performance Tracking**: Monitor portfolio returns over time
- **Asset Allocation**: Optimize portfolio composition
- **Compliance**: Ensure portfolio meets allocation constraints

## Notes

- All calculations assume market prices are accurate and current
- Rebalancing calculations don't account for transaction costs
- Beta calculations use weighted average of individual stock betas
- Diversification metrics include sector and concentration analysis
- Portfolio returns are calculated based on average cost basis
