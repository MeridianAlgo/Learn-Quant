<p class="lq-badges"><span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-cat">Options, Derivatives & Finance</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Finance - Expected Shortfall"
    python "expected_shortfall.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Finance%20-%20Expected%20Shortfall)

---
# Expected Shortfall (CVaR)

Expected Shortfall (ES), also called Conditional Value at Risk (CVaR), measures the **expected loss given that losses exceed the VaR threshold**. It is a coherent risk measure — unlike VaR, it captures tail severity, not just frequency.

## Functions

| Function | Description |
|---|---|
| `historical_es(returns, confidence_level)` | Non-parametric ES from actual distribution |
| `parametric_es(returns, confidence_level)` | Normal-assumption ES |
| `cornish_fisher_es(returns, confidence_level)` | Skewness/kurtosis-adjusted ES |
| `es_summary(returns, confidence_level)` | All three estimates in one dict |

## Key Concepts

- **VaR vs ES**: VaR says "you won't lose more than X with 95% probability." ES says "given you exceed VaR, your average loss is Y."
- **Coherence**: ES satisfies subadditivity — diversification always reduces risk. VaR does not.
- **Cornish-Fisher**: Adjusts the normal quantile using higher moments. Better for fat-tailed (leptokurtic) returns.

## Example

```python
from expected_shortfall import es_summary
import numpy as np

returns = np.random.normal(0.001, 0.02, 252)
summary = es_summary(returns, confidence_level=0.95)
print(summary)
# {'historical_es': 0.0412, 'parametric_es': 0.0398, 'cornish_fisher_es': 0.0405, ...}
```

## When to Use

- Portfolio risk reporting (ES is required under Basel III / FRTB)
- Comparing risk across strategies with different tail behaviors
- Stress testing alongside VaR


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
