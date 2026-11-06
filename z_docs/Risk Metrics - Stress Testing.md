# Portfolio Stress Testing

<p class="lq-meta">Intermediate <span>/</span> Risk & Performance <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Risk%20Metrics%20-%20Stress%20Testing">source</a></p>


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

## Running it

```bash
cd "Risk Metrics - Stress Testing"
python "stress_testing.py"
```

## More in Risk & Performance

- [Finance - Calmar Ratio](Finance - Calmar Ratio.md): The Sharpe ratio judges a strategy by how much its returns wobble around their average.
- [Finance - Information Ratio](Finance - Information Ratio.md): When a portfolio is judged against a benchmark, what matters is how much it beat the benchmark by — and how *reliably*.
- [Finance - Jensens Alpha](Finance - Jensens Alpha.md): Beta tells you how much of a portfolio's return came from simply riding the market.
- [Finance - Omega Ratio](Finance - Omega Ratio.md): The Sharpe ratio squeezes a whole return distribution into a mean and a standard deviation, which throws away everything interesting about its shape.
- [Finance - Performance Attribution](Finance - Performance Attribution.md): Brinson decomposition splits portfolio active return into allocation and selection effects — answering *"did we beat the benchmark by picking the right sectors or the right stocks?"*
- [Finance - Treynor Ratio](Finance - Treynor Ratio.md): The Sharpe ratio divides excess return by total volatility, which mixes two very different kinds of risk.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
