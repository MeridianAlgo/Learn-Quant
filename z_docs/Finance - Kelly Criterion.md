<p class="lq-badges"><span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-cat">Options, Derivatives & Finance</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Finance - Kelly Criterion"
    python "kelly_criterion.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Finance%20-%20Kelly%20Criterion)

---
# Kelly Criterion Position Sizing

The Kelly Criterion determines the **optimal fraction of capital to allocate** to maximize the long-run geometric growth rate of wealth.

## Functions

| Function | Description |
|---|---|
| `kelly_fraction(win_prob, win_loss_ratio)` | Discrete Kelly for binary bets |
| `kelly_continuous(mu, sigma)` | Kelly for continuous return distributions |
| `fractional_kelly(win_prob, ratio, fraction)` | Scaled-down Kelly for risk control |
| `multi_asset_kelly(expected_returns, cov)` | Portfolio-level optimal allocation |
| `kelly_growth_rate(win_prob, ratio, fraction)` | Expected log growth at given fraction |

## Key Concepts

- **Discrete Kelly**: `f* = p - q/b` where p=win prob, q=1-p, b=win/loss ratio.
- **Continuous Kelly**: `f* = mu / sigma²`. Optimal leverage for log-normal returns.
- **Overbetting kills compounding**: Full Kelly maximizes growth but has huge variance. Half-Kelly roughly preserves 75% of growth with half the variance.
- **Multi-asset**: `f* = Σ⁻¹μ`. Accounts for correlations — a diversified Kelly portfolio.

## Example

```python
from kelly_criterion import kelly_fraction, fractional_kelly, kelly_growth_rate

# 60% win rate, 2:1 payoff
full_kelly = kelly_fraction(0.60, 2.0)   # 0.20 = 20% of capital
half_kelly = fractional_kelly(0.60, 2.0, 0.5)  # 0.10

# Growth rates
print(kelly_growth_rate(0.60, 2.0, 0.5))   # ~0.02 per bet
print(kelly_growth_rate(0.60, 2.0, 1.25))  # Lower! Overbetting hurts
```

## Practical Notes

- Full Kelly is theoretically optimal but practically dangerous (high variance, large drawdowns)
- **Half-Kelly is the most common real-world choice** among professional gamblers and traders
- Kelly assumes i.i.d. bets — serial correlation in returns changes the optimal fraction


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
