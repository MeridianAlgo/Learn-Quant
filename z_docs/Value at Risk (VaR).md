<p class="lq-badges"><span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-cat">Risk & Performance</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Value at Risk (VaR)"
    python "var_calculator.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Value%20at%20Risk%20%28VaR%29)

---
# Value at Risk (VaR) & Conditional VaR

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

## Continue in Risk & Performance

<div class="grid cards" markdown>

-   :material-shield-alert-outline: __[Finance - Information Ratio](Finance - Information Ratio.md)__

    When a portfolio is judged **against a benchmark**, what matters is how much it beat the benchmark by — and how *reliably*. These are the core metrics of active management: active return, tracking error, Information Ratio, and the appraisal ratio.

-   :material-shield-alert-outline: __[Finance - Performance Attribution](Finance - Performance Attribution.md)__

    Brinson decomposition splits portfolio active return into **allocation** and **selection** effects — answering *"did we beat the benchmark by picking the right sectors or the right stocks?"*

-   :material-shield-alert-outline: __[Risk Metrics](Risk Metrics.md)__

    This module gives you quick, professional stats about risk in any list or array of investment returns. It's used by investors, analysts, and students everywhere!

-   :material-shield-alert-outline: __[Risk Metrics - Drawdown Analysis](Risk Metrics - Drawdown Analysis.md)__

    Comprehensive drawdown metrics for quantifying portfolio loss risk over time. Drawdown measures capture both the **depth** and **duration** of losses — dimensions VaR ignores.

-   :material-shield-alert-outline: __[Risk Metrics - Stress Testing](Risk Metrics - Stress Testing.md)__

    Stress tests answer: *"What happens if 2008 repeats?"* or *"How big a shock kills the portfolio?"* Required by Basel III, CCAR, and most institutional risk frameworks.

-   :material-shield-alert-outline: __[Sharpe and Sortino Ratio](Sharpe and Sortino Ratio.md)__

    This utility offers easy-to-use Python functions to calculate Sharpe and Sortino ratios for financial returns. These ratios help you understand whether a series of investment returns is attractive on a risk-adjusted basis.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
