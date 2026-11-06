# Kelly Criterion Position Sizing

<p class="lq-meta">Intermediate <span>/</span> Options, Derivatives & Finance <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Finance%20-%20Kelly%20Criterion">source</a></p>


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

## Running it

```bash
cd "Finance - Kelly Criterion"
python "kelly_criterion.py"
```

## More in Options, Derivatives & Finance

- [Advanced Options Pricing](Advanced Options Pricing.md): This module covers advanced mathematical techniques for pricing financial derivatives.
- [Black-Scholes Option Pricing](Black-Scholes Option Pricing.md): This module lets you price basic stock options (calls and puts) using the Black-Scholes formula, a foundation of modern financial analysis.
- [Bond Price and Yield](Bond Price and Yield.md): This utility lets you calculate the fair price of a bond or estimate its yield to maturity (YTM), two of the most basic (and important!) ideas in investing.
- [CAPM](CAPM.md): CAPM is the idea that won a Nobel Prize and still anchors how the industry thinks about return and risk.
- [Discounted Cash Flow (DCF)](Discounted Cash Flow (DCF).md): This tool calculates the present value of a series of future cash flows—the basic principle behind valuing businesses, real estate, projects, and stocks!
- [Dividend Tracker](Dividend Tracker.md): This tool lets you track upcoming dividends for a list of stocks, calculate expected income, and view a dividend calendar.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
