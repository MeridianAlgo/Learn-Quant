# Portfolio Management Utilities

This folder contains utilities for portfolio management, risk analysis, and investment optimization.

## Available Utilities

### Portfolio Management (`portfolio_utils.py`)
- Portfolio valuation and allocation analysis
- Rebalancing calculations
- Diversification metrics
- Position sizing
- Portfolio turnover analysis

### Risk Analysis (`risk_utils.py`)
- Value at Risk (VaR) calculations
- Maximum drawdown analysis
- Volatility calculations
- Sharpe and Sortino ratios
- Correlation analysis
- Stress testing

## Usage

```python
# Portfolio operations
from portfolio_utils import calculate_portfolio_value, rebalance_portfolio
from risk_utils import calculate_var, calculate_sharpe_ratio, calculate_max_drawdown

# Portfolio analysis
portfolio_value = calculate_portfolio_value(holdings, prices)
allocation = calculate_portfolio_allocation(holdings, prices)

# Risk management
var_95 = calculate_var(returns, 0.95)
sharpe = calculate_sharpe_ratio(returns)
max_dd = calculate_max_drawdown(prices)

# Rebalancing
trades = rebalance_portfolio(target_allocation, holdings, prices, portfolio_value)
```

## Installation

Requires numpy for statistical calculations:

```bash
pip install numpy
```

## Testing

Run each utility directly to see demonstrations:

```bash
python portfolio_utils.py
python risk_utils.py
```

## Common Use Cases

- **Portfolio Analysis**: Evaluate portfolio performance and risk
- **Risk Management**: Calculate risk metrics and set risk limits
- **Rebalancing**: Maintain target asset allocation
- **Performance Tracking**: Monitor portfolio returns and metrics
- **Investment Optimization**: Optimize portfolio composition
- **Compliance**: Generate risk reports for regulatory requirements
