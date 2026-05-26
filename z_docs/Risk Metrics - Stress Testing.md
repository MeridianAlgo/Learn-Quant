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
