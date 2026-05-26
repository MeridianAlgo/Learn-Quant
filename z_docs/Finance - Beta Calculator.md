# Beta Calculator

Calculate beta and related systematic risk metrics for portfolio analysis.

## Features

- Standard Beta Calculation
- Rolling Beta
- Levered/Unlevered Beta
- Downside Beta
- Upside Beta
- Beta Decomposition
- Adjusted Beta (Blume)

## Usage

```python
from beta_calculator import calculate_beta, beta_decomposition

asset_returns = [0.01, -0.02, 0.015, -0.01, 0.02]
market_returns = [0.008, -0.015, 0.012, -0.008, 0.018]

beta = calculate_beta(asset_returns, market_returns)
print(f"Beta: {beta:.4f}")

decomp = beta_decomposition(asset_returns, market_returns)
```

## Applications

- CAPM cost of equity estimation
- Portfolio risk attribution
- Leverage analysis
- Market timing strategies
- Risk-adjusted performance measurement
