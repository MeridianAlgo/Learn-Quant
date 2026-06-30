<p class="lq-badges"><span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-cat">Risk & Performance</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Risk Metrics - Stress Testing"
    python "stress_testing.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Risk%20Metrics%20-%20Stress%20Testing)

---
# Portfolio Stress Testing

Stress tests answer: *"What happens if 2008 repeats?"* or *"How big a shock kills the portfolio?"* Required by Basel III, CCAR, and most institutional risk frameworks.

## Functions

| Function | Description |
|---|---|
| `apply_scenario(weights, shocks, V)` | Apply per-asset return shocks |
| `historical_scenario(weights, name, V)` | Replay 2008 GFC, 2020 COVID, etc. |
| `sensitivity_analysis(weights, range, idx)` | Univariate shock sweep |
| `reverse_stress_test(weights, direction, loss)` | Find shock magnitude that breaches loss |
| `worst_case(weights, scenarios, V)` | Worst P&L across scenario set |

## Built-in Historical Scenarios

| Name | Equity | Credit | Rates | Commodity | FX |
|---|---|---|---|---|---|
| `2008_GFC` | -50% | -30% | -2% | -40% | -15% |
| `2020_COVID` | -34% | -20% | -1.5% | -55% | -10% |
| `1987_Black_Monday` | -22.5% | -5% | +0.5% | -10% | -5% |
| `2000_DotCom` | -49% | -10% | -3% | +5% | -8% |
| `2022_Inflation` | -20% | -13% | +4% | +30% | +12% |

## Example

```python
from stress_testing import historical_scenario, reverse_stress_test

res = historical_scenario(
    {"equity": 0.6, "credit": 0.3, "rates": 0.1},
    "2008_GFC",
    portfolio_value=10_000_000,
)
print(res["portfolio_pnl"])

# How big an equity-only shock to lose 25%?
k = reverse_stress_test([1.0], [-1.0], 0.25, 1.0)  # -> 0.25
```

## Practical Notes

- Shocks are **return shocks**, not price shocks (use `-0.50`, not `0.50`).
- Combine with VaR/ES for a complete tail-risk picture.
- Reverse stress tests reveal what scenario is *plausible enough* to breach risk limits.


---

## Continue in Risk & Performance

<div class="grid cards" markdown>

-   :material-shield-alert-outline: __[Finance - Calmar Ratio](Finance - Calmar Ratio.md)__

    The Sharpe ratio judges a strategy by how much its returns wobble around their

-   :material-shield-alert-outline: __[Finance - Information Ratio](Finance - Information Ratio.md)__

    When a portfolio is judged **against a benchmark**, what matters is how much it beat the benchmark by — and how *reliably*. These are the core metrics of active management: active return, tracking error, Information Ratio, and the appraisal ratio.

-   :material-shield-alert-outline: __[Finance - Performance Attribution](Finance - Performance Attribution.md)__

    Brinson decomposition splits portfolio active return into **allocation** and **selection** effects — answering *"did we beat the benchmark by picking the right sectors or the right stocks?"*

-   :material-shield-alert-outline: __[Risk Metrics](Risk Metrics.md)__

    This module gives you quick, professional stats about risk in any list or array of investment returns. It's used by investors, analysts, and students everywhere!

-   :material-shield-alert-outline: __[Risk Metrics - Drawdown Analysis](Risk Metrics - Drawdown Analysis.md)__

    Comprehensive drawdown metrics for quantifying portfolio loss risk over time. Drawdown measures capture both the **depth** and **duration** of losses — dimensions VaR ignores.

-   :material-shield-alert-outline: __[Sharpe and Sortino Ratio](Sharpe and Sortino Ratio.md)__

    This utility offers easy-to-use Python functions to calculate Sharpe and Sortino ratios for financial returns. These ratios help you understand whether a series of investment returns is attractive on a risk-adjusted basis.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
