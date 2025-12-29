# Risk Analysis Utilities

This module provides comprehensive risk analysis utilities for financial applications, including Value at Risk (VaR), maximum drawdown, volatility calculations, correlation analysis, and stress testing.

## Functions

### `calculate_var(returns: List[float], confidence_level: float = 0.95) -> float`
Calculates Value at Risk (VaR) using historical method.

**Parameters:**
- `returns`: List of portfolio returns
- `confidence_level`: Confidence level (e.g., 0.95 for 95% VaR)

**Returns:**
- VaR value (negative number representing loss)

**Example:**
```python
>>> returns = [0.02, -0.01, 0.03, -0.02, 0.01]
>>> var = calculate_var(returns, 0.95)
>>> print(f"95% VaR: {var:.2%}")
-0.02
```

### `calculate_max_drawdown(prices: List[float]) -> Dict[str, float]`
Calculates maximum drawdown and related metrics.

**Parameters:**
- `prices`: List of prices or portfolio values

**Returns:**
- Dictionary with max_drawdown, drawdown_duration, recovery_time

**Example:**
```python
>>> prices = [100, 110, 105, 95, 90, 100, 115]
>>> dd = calculate_max_drawdown(prices)
>>> print(f"Max Drawdown: {dd['max_drawdown']:.2%}")
-18.18%
```

### `calculate_volatility(returns: List[float], annualize: bool = True, periods_per_year: int = 252) -> float`
Calculates volatility (standard deviation of returns).

**Parameters:**
- `returns`: List of returns
- `annualize`: Whether to annualize the volatility
- `periods_per_year`: Number of periods per year for annualization

**Returns:**
- Volatility

**Example:**
```python
>>> returns = [0.02, -0.01, 0.03, -0.02, 0.01]
>>> vol = calculate_volatility(returns)
>>> print(f"Annualized Volatility: {vol:.2%}")
25.30%
```

### `calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float`
Calculates Sharpe ratio.

**Parameters:**
- `returns`: List of returns
- `risk_free_rate`: Risk-free rate (annual)
- `periods_per_year`: Number of periods per year

**Returns:**
- Sharpe ratio

**Example:**
```python
>>> returns = [0.02, -0.01, 0.03, -0.02, 0.01]
>>> sharpe = calculate_sharpe_ratio(returns)
>>> print(f"Sharpe Ratio: {sharpe:.2f}")
0.79
```

### `calculate_sortino_ratio(returns: List[float], risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float`
Calculates Sortino ratio (downside risk-adjusted return).

**Parameters:**
- `returns`: List of returns
- `risk_free_rate`: Risk-free rate (annual)
- `periods_per_year`: Number of periods per year

**Returns:**
- Sortino ratio

**Example:**
```python
>>> returns = [0.02, -0.01, 0.03, -0.02, 0.01]
>>> sortino = calculate_sortino_ratio(returns)
>>> print(f"Sortino Ratio: {sortino:.2f}")
1.12
```

### `calculate_correlation_matrix(returns_data: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]`
Calculates correlation matrix for multiple assets.

**Parameters:**
- `returns_data`: Dictionary of returns {asset: [returns]}

**Returns:**
- Correlation matrix dictionary

**Example:**
```python
>>> returns_data = {"AAPL": [0.02, -0.01], "GOOGL": [0.01, -0.02]}
>>> corr = calculate_correlation_matrix(returns_data)
>>> print(f"AAPL-GOOGL correlation: {corr['AAPL']['GOOGL']:.2f}")
0.50
```

### `stress_test_portfolio(holdings: Dict[str, Dict[str, Any]], scenarios: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]`
Performs stress testing on portfolio under different scenarios.

**Parameters:**
- `holdings`: Portfolio holdings
- `scenarios`: Stress scenarios {name: {symbol: shock_pct}}

**Returns:**
- Portfolio values under each scenario

**Example:**
```python
>>> holdings = {"AAPL": {"shares": 100}}
>>> scenarios = {"Market Crash": {"AAPL": -0.30}}
>>> results = stress_test_portfolio(holdings, scenarios)
>>> print(f"Crash scenario value: ${results['Market Crash']['portfolio_value']:,.2f}")
10500.00
```

### `calculate_beta(asset_returns: List[float], market_returns: List[float]) -> float`
Calculates beta of an asset relative to market.

**Parameters:**
- `asset_returns`: Asset returns
- `market_returns`: Market returns

**Returns:**
- Beta value

**Example:**
```python
>>> asset_returns = [0.02, -0.01, 0.03]
>>> market_returns = [0.01, -0.005, 0.02]
>>> beta = calculate_beta(asset_returns, market_returns)
>>> print(f"Beta: {beta:.2f}")
1.50
```

## Usage

```python
from risk_utils import (
    calculate_var, calculate_max_drawdown, calculate_volatility,
    calculate_sharpe_ratio, calculate_sortino_ratio,
    calculate_correlation_matrix, stress_test_portfolio, calculate_beta
)

# Sample returns data
returns = [0.02, -0.01, 0.03, -0.02, 0.01, 0.015, -0.005, 0.025]

# Risk metrics
var_95 = calculate_var(returns, 0.95)
var_99 = calculate_var(returns, 0.99)
volatility = calculate_volatility(returns)
sharpe = calculate_sharpe_ratio(returns)
sortino = calculate_sortino_ratio(returns)

print(f"95% VaR: {var_95:.2%}")
print(f"99% VaR: {var_99:.2%}")
print(f"Volatility: {volatility:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Sortino Ratio: {sortino:.2f}")

# Drawdown analysis
prices = [100, 105, 110, 108, 95, 90, 92, 98, 105, 115]
drawdown = calculate_max_drawdown(prices)
print(f"Max Drawdown: {drawdown['max_drawdown']:.2%}")
print(f"Drawdown Duration: {drawdown['drawdown_duration']} periods")

# Correlation analysis
returns_data = {
    "AAPL": [0.02, -0.01, 0.03, -0.02],
    "GOOGL": [0.01, -0.02, 0.025, -0.015],
    "MSFT": [0.015, -0.005, 0.02, -0.01]
}
correlations = calculate_correlation_matrix(returns_data)

# Stress testing
holdings = {"AAPL": {"shares": 100}, "GOOGL": {"shares": 50}}
scenarios = {
    "Market Crash": {"AAPL": -0.30, "GOOGL": -0.25},
    "Tech Boom": {"AAPL": 0.20, "GOOGL": 0.15}
}
stress_results = stress_test_portfolio(holdings, scenarios)
```

## Installation

Requires numpy for statistical calculations:

```bash
pip install numpy
```

## Testing

Run the module directly to see demonstrations:

```bash
python risk_utils.py
```

## Common Use Cases

- **Risk Management**: Calculate VaR, maximum drawdown, and volatility
- **Performance Analysis**: Evaluate risk-adjusted returns (Sharpe, Sortino)
- **Portfolio Construction**: Analyze correlations and diversification benefits
- **Stress Testing**: Test portfolio resilience under adverse scenarios
- **Regulatory Reporting**: Generate risk metrics for compliance
- **Risk Budgeting**: Allocate risk across portfolio positions
- **Scenario Analysis**: Model portfolio behavior under different market conditions

## Notes

- VaR calculation uses historical simulation method
- Maximum drawdown includes peak-to-trough analysis
- Volatility is annualized assuming daily returns by default
- Correlation matrix returns values between -1 and 1
- Stress testing applies percentage shocks to current holdings
- Beta calculation uses linear regression against market returns
- All calculations assume returns are in decimal format (0.02 = 2%)

## Risk Metrics Interpretation

- **VaR**: Maximum expected loss over given time horizon at confidence level
- **Max Drawdown**: Largest peak-to-trough decline in portfolio value
- **Volatility**: Standard deviation of returns, measure of risk
- **Sharpe Ratio**: Risk-adjusted return, higher is better
- **Sortino Ratio**: Downside risk-adjusted return, higher is better
- **Beta**: Sensitivity to market movements, >1 = more volatile
- **Correlation**: Relationship between assets, diversification benefit
