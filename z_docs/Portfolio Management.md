<p class="lq-badges"><span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-cat">Portfolio Management</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Portfolio Management"
    python "portfolio_utils.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Portfolio%20Management)

---
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


---

## Continue in Portfolio Management

<div class="grid cards" markdown>

-   :material-briefcase-outline: __[Monte Carlo Portfolio Simulator](Monte Carlo Portfolio Simulator.md)__

    This utility helps you forecast possible futures for a portfolio using random simulations—a key idea in finance, risk management, and statistics!

-   :material-briefcase-outline: __[Portfolio Management - Black Litterman](Portfolio Management - Black Litterman.md)__

    The Black-Litterman (1990) model addresses the instability of mean-variance optimization by blending **market equilibrium returns** with **investor views** using Bayesian updating.

-   :material-briefcase-outline: __[Portfolio Management - Risk Parity](Portfolio Management - Risk Parity.md)__

    Risk parity builds a portfolio where **every asset contributes the same amount of risk** to the total — not the same amount of capital. A naive 60/40 stock/bond portfolio is ~90% *equity risk* despite being only 60% equity *capital*; risk parity fixes that imbalance.

-   :material-briefcase-outline: __[Portfolio Optimizer](Portfolio Optimizer.md)__

    This utility helps you find the best mix of assets for a portfolio, balancing risk and return using the foundation of Modern Portfolio Theory (MPT).

-   :material-briefcase-outline: __[Portfolio Tracker](Portfolio Tracker.md)__

    **This utility uses the yfinance API to fetch current prices automatically.** All other calculations and data are managed locally for learning and experimentation.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
