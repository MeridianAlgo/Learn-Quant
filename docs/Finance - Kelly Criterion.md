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
