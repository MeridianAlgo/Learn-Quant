# Risk Parity Portfolio Construction

<p class="lq-meta">Advanced <span>/</span> Portfolio Management <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Portfolio%20Management%20-%20Risk%20Parity">source</a></p>


Risk parity builds a portfolio where **every asset contributes the same amount of risk** to the total — not the same amount of capital. A naive 60/40 stock/bond portfolio is ~90% *equity risk* despite being only 60% equity *capital*; risk parity fixes that imbalance.

## Functions

| Function | Description |
|---|---|
| `portfolio_volatility(weights, cov)` | Portfolio standard deviation `sqrt(wᵀΣw)` |
| `risk_contributions(weights, cov)` | Component risk contribution per asset (sums to total vol) |
| `inverse_volatility_weights(cov)` | Naive risk parity — closed-form, ignores correlations |
| `risk_parity_weights(cov, budget=None)` | Equal Risk Contribution (ERC) or custom risk budget, solved numerically |

## Key Concepts

- **Marginal risk contribution**: `MRC = (Σw) / σ_p` — how much portfolio volatility changes per unit of weight.
- **Component risk contribution**: `w ⊙ MRC`. By Euler's theorem these sum *exactly* to the portfolio volatility.
- **Equal Risk Contribution (ERC)**: choose weights so every component contribution is equal. No closed form in general → solved by minimising the squared deviation of fractional contributions from the target budget.
- **Risk budgeting**: ERC generalised — set an arbitrary target risk share per asset (e.g. 50% equity / 30% bond / 20% cash).

## Example

```python
import numpy as np
from risk_parity import risk_parity_weights, risk_contributions

vols = np.array([0.20, 0.10, 0.04])
corr = np.array([[1.0, 0.3, 0.05], [0.3, 1.0, 0.15], [0.05, 0.15, 1.0]])
cov = np.outer(vols, vols) * corr

w = risk_parity_weights(cov)              # ERC weights
rc = risk_contributions(w, cov)
print(rc / rc.sum())                      # ≈ equal risk shares
```

## Practical Notes

- Inverse-volatility weighting is a fast, robust approximation and is exact only when assets are uncorrelated.
- Risk parity portfolios are often **levered up** to reach a target volatility, since they tend to be bond-heavy and low-vol.
- The optimiser starts from inverse-vol weights, which are close to the ERC solution, so SLSQP converges quickly and reliably.

---

## Running it

```bash
cd "Portfolio Management - Risk Parity"
python "risk_parity.py"
```

## More in Portfolio Management

- [Monte Carlo Portfolio Simulator](Monte Carlo Portfolio Simulator.md): This utility helps you forecast possible futures for a portfolio using random simulations—a key idea in finance, risk management, and statistics!
- [Portfolio Analytics - JavaScript](Portfolio Analytics - JavaScript.md): A portfolio is not the sum of its holdings, and the whole of portfolio theory follows from that one fact.
- [Portfolio Management](Portfolio Management.md): This folder contains utilities for portfolio management, risk analysis, and investment optimization.
- [Portfolio Management - Black Litterman](Portfolio Management - Black Litterman.md): The Black-Litterman (1990) model addresses the instability of mean-variance optimization by blending market equilibrium returns with investor views using Bayesian updating.
- [Portfolio Management - Hierarchical Risk Parity](Portfolio Management - Hierarchical Risk Parity.md): Markowitz optimisation needs the inverse of the covariance matrix, and the inverse of a noisy matrix is mostly noise.
- [Portfolio Optimizer](Portfolio Optimizer.md): This utility helps you find the best mix of assets for a portfolio, balancing risk and return using the foundation of Modern Portfolio Theory (MPT).

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
