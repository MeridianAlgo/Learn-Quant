# Performance Attribution

Brinson decomposition splits portfolio active return into **allocation** and **selection** effects — answering *"did we beat the benchmark by picking the right sectors or the right stocks?"*

## Functions

| Function | Description |
|---|---|
| `brinson_attribution(...)` | Three-factor BHB: allocation + selection + interaction |
| `two_factor_brinson(...)` | Two-factor: allocation + (selection inc. interaction) |
| `information_ratio(rp, rb)` | Annualized IR = active return / tracking error |
| `tracking_error(rp, rb)` | Annualized std of active returns |

## Brinson-Hood-Beebower Decomposition

```
Allocation_i  = (w_p,i - w_b,i) * (r_b,i - r_b)
Selection_i   = w_b,i * (r_p,i - r_b,i)
Interaction_i = (w_p,i - w_b,i) * (r_p,i - r_b,i)

Active = sum(Allocation) + sum(Selection) + sum(Interaction)
```

## Example

```python
from performance_attribution import brinson_attribution, information_ratio

res = brinson_attribution(wp, rp, wb, rb)
print(res['total_allocation'], res['total_selection'])

ir = information_ratio(daily_port, daily_bench)
```

## Practical Notes

- **Allocation** > 0: overweighted sectors that beat the benchmark.
- **Selection** > 0: stocks within sectors outperformed sector index.
- **Interaction** is small and often combined with selection (two-factor model).
- IR > 0.5 is good; > 1.0 is exceptional.
