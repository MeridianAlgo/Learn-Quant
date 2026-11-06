# Value at Risk (VaR) & Conditional VaR

<p class="lq-meta">Intermediate <span>/</span> Risk & Performance <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Value%20at%20Risk%20%28VaR%29">source</a></p>


**Value at Risk** is the single most widely quoted number in financial risk
management. It compresses "how bad could it get?" into one figure:

> *There is a 5% chance of losing more than this amount over the next day.*

Banks size capital around it, desks set limits with it, and regulators demand
it. This module implements the three standard ways to estimate VaR, the tail
measure that fixes VaR's biggest flaw, and a backtest to check your number is
actually honest.

## Functions

| Function | Description |
|---|---|
| `parametric_var(returns, confidence_level)` | Variance–covariance VaR (assumes normal returns) |
| `historical_var(returns, confidence_level)` | Empirical-quantile VaR — no distribution assumed |
| `monte_carlo_var(returns, confidence_level, n_sims)` | Simulation-based VaR from a fitted model |
| `conditional_var(returns, confidence_level)` | Conditional VaR / Expected Shortfall (mean tail loss) |
| `kupiec_pof_test(returns, var_level, confidence_level)` | Proportion-of-failures backtest |
| `value_at_risk(...)` | Backwards-compatible alias for `parametric_var` |

## The three methods, and when to trust them

| Method | Assumption | Strength | Weakness |
|---|---|---|---|
| **Parametric** | Returns ~ Normal | Fast, closed-form | Underestimates fat tails |
| **Historical** | History repeats | Captures real skew/kurtosis | Bounded by worst observed day |
| **Monte Carlo** | A chosen model | Flexible, forward-looking | Only as good as the model |

```
Parametric:  VaR = -(mu + sigma * Phi^-1(1 - c))
Historical:  VaR = -quantile(returns, 1 - c)
```

## Why Conditional VaR (Expected Shortfall)?

VaR tells you the *threshold* of the tail but says nothing about how bad things
get *beyond* it — and it is not sub-additive, so it can punish diversification.
**Conditional VaR** averages the losses past the VaR point, is a *coherent* risk
measure, and is what Basel III now favours.

```python
import numpy as np
from var_calculator import parametric_var, historical_var, conditional_var, kupiec_pof_test

returns = np.random.normal(0.0005, 0.02, 1000)

print(parametric_var(returns, 0.95))
print(historical_var(returns, 0.99))
print(conditional_var(returns, 0.99))   # Expected Shortfall

# Is the model adequate? Backtest it.
var95 = parametric_var(returns, 0.95)
print(kupiec_pof_test(returns, var95, 0.95))
```

## Backtesting with the Kupiec test

A VaR model is only credible if reality agrees with it. The **Kupiec
proportion-of-failures test** counts how often losses breached your VaR and runs
a likelihood-ratio test against the expected breach rate. A trustworthy model is
one you *fail to reject* — too many exceptions means the model understates risk;
too few means it wastes capital.

## Practical notes

- VaR scales with horizon roughly as `sqrt(t)` under the normal assumption — a
  1-day VaR becomes a 10-day VaR by multiplying by `sqrt(10)`.
- The parametric and historical numbers diverge most in crises — that gap is the
  fat-tail risk. For the *deep* tail (99%+) prefer
  `Quantitative Methods - Extreme Value Theory`.
- Pair this with `Finance - Expected Shortfall`, `Risk Metrics` and
  `Risk Metrics - Stress Testing` for a complete risk picture.
- VaR is a *probabilistic* statement, not a worst case. The maximum loss is
  always larger than VaR — that is the whole point of also tracking CVaR.

---

## Running it

```bash
cd "Value at Risk (VaR)"
python "var_calculator.py"
```

## More in Risk & Performance

- [Finance - Calmar Ratio](Finance - Calmar Ratio.md): The Sharpe ratio judges a strategy by how much its returns wobble around their average.
- [Finance - Information Ratio](Finance - Information Ratio.md): When a portfolio is judged against a benchmark, what matters is how much it beat the benchmark by — and how *reliably*.
- [Finance - Jensens Alpha](Finance - Jensens Alpha.md): Beta tells you how much of a portfolio's return came from simply riding the market.
- [Finance - Omega Ratio](Finance - Omega Ratio.md): The Sharpe ratio squeezes a whole return distribution into a mean and a standard deviation, which throws away everything interesting about its shape.
- [Finance - Performance Attribution](Finance - Performance Attribution.md): Brinson decomposition splits portfolio active return into allocation and selection effects — answering *"did we beat the benchmark by picking the right sectors or the right stocks?"*
- [Finance - Treynor Ratio](Finance - Treynor Ratio.md): The Sharpe ratio divides excess return by total volatility, which mixes two very different kinds of risk.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
