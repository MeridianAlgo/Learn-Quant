# Portfolio Management Utilities

<p class="lq-meta">Advanced <span>/</span> Portfolio Management <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Portfolio%20Management">source</a></p>


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

---

## Running it

```bash
cd "Portfolio Management"
python "portfolio_utils.py"
```

## More in Portfolio Management

- [Monte Carlo Portfolio Simulator](Monte Carlo Portfolio Simulator.md): This utility helps you forecast possible futures for a portfolio using random simulations—a key idea in finance, risk management, and statistics!
- [Portfolio Analytics - JavaScript](Portfolio Analytics - JavaScript.md): A portfolio is not the sum of its holdings, and the whole of portfolio theory follows from that one fact.
- [Portfolio Management - Black Litterman](Portfolio Management - Black Litterman.md): The Black-Litterman (1990) model addresses the instability of mean-variance optimization by blending market equilibrium returns with investor views using Bayesian updating.
- [Portfolio Management - Hierarchical Risk Parity](Portfolio Management - Hierarchical Risk Parity.md): Markowitz optimisation needs the inverse of the covariance matrix, and the inverse of a noisy matrix is mostly noise.
- [Portfolio Management - Risk Parity](Portfolio Management - Risk Parity.md): Risk parity builds a portfolio where every asset contributes the same amount of risk to the total — not the same amount of capital.
- [Portfolio Optimizer](Portfolio Optimizer.md): This utility helps you find the best mix of assets for a portfolio, balancing risk and return using the foundation of Modern Portfolio Theory (MPT).

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
