<p class="lq-badges"><span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-cat">Options, Derivatives & Finance</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Finance - Covariance Estimation"
    python "covariance_estimation.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Finance%20-%20Covariance%20Estimation)

---
# Robust Covariance Estimation

Sample covariance is noisy and often poorly conditioned with many assets. Shrinkage estimators blend sample covariance with a structured target for more stable portfolio optimization.

## Functions

| Function | Description |
|---|---|
| `sample_covariance(returns)` | Standard MLE covariance |
| `ledoit_wolf_shrinkage(returns)` | Analytical LW shrinkage toward scaled identity |
| `constant_correlation_shrinkage(returns)` | Shrink toward equal-correlation matrix |
| `ewma_covariance(returns, lambda_)` | RiskMetrics exponentially weighted covariance |
| `condition_number(cov)` | Ratio of max/min eigenvalue (lower = better) |

## Key Concepts

- **Curse of dimensionality**: With N assets and T observations, sample covariance has N(N+1)/2 parameters. When T < N, it's singular.
- **Ledoit-Wolf**: Finds optimal alpha blending `S` toward `mu*I`. Minimizes expected Frobenius distance to true covariance.
- **Constant Correlation**: Preserves sample variances but equalizes all correlations to the cross-sectional mean. Works well for equity portfolios.
- **EWMA**: Downweights old data — lambda=0.94 (daily) makes the half-life ~11 days. Captures volatility clustering.

## Example

```python
import numpy as np
from covariance_estimation import ledoit_wolf_shrinkage, condition_number

T, N = 252, 50
returns = np.random.randn(T, N) * 0.01
S = np.cov(returns.T)

lw = ledoit_wolf_shrinkage(returns)
print(f"Sample cov condition number: {condition_number(S):.0f}")
print(f"LW shrunk condition number:  {condition_number(lw['shrunk_cov']):.0f}")
print(f"Shrinkage intensity alpha:   {lw['alpha']:.3f}")
```

## Rule of Thumb

- T > 10N: Sample covariance is fine
- T ~ 3N: Use Ledoit-Wolf or constant correlation
- T < 2N: Consider factor-based covariance models


---

## Continue in Options, Derivatives & Finance

<div class="grid cards" markdown>

-   :material-chart-bell-curve: __[Advanced Options Pricing](Advanced Options Pricing.md)__

    This module covers advanced mathematical techniques for pricing financial derivatives. The focus is on models beyond the standard assumptions. Rather than assuming constant volatility, we explore dynamic and local volatility models. These models are crucial for correctly valuing exotic options and managing the risks of complex derivatives portfolios.

-   :material-chart-bell-curve: __[Black-Scholes Option Pricing](Black-Scholes Option Pricing.md)__

    This module lets you price basic stock options (calls and puts) using the Black-Scholes formula, a foundation of modern financial analysis.

-   :material-chart-bell-curve: __[Bond Price and Yield](Bond Price and Yield.md)__

    This utility lets you calculate the fair price of a bond or estimate its yield to maturity (YTM), two of the most basic (and important!) ideas in investing.

-   :material-chart-bell-curve: __[CAPM](CAPM.md)__

    CAPM is the idea that won a Nobel Prize and still anchors how the industry

-   :material-chart-bell-curve: __[Discounted Cash Flow (DCF)](Discounted Cash Flow (DCF).md)__

    This tool calculates the present value of a series of future cash flows—the basic principle behind valuing businesses, real estate, projects, and stocks!

-   :material-chart-bell-curve: __[Dividend Tracker](Dividend Tracker.md)__

    **This utility does NOT use any external APIs.** All data is managed locally for learning and experimentation.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
