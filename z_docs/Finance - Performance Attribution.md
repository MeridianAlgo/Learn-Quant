<p class="lq-badges"><span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-cat">Risk & Performance</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Finance - Performance Attribution"
    python "performance_attribution.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Finance%20-%20Performance%20Attribution)

---
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


---

## Continue in Risk & Performance

<div class="grid cards" markdown>

-   :material-shield-alert-outline: __[Finance - Calmar Ratio](Finance - Calmar Ratio.md)__

    The Sharpe ratio judges a strategy by how much its returns wobble around their

-   :material-shield-alert-outline: __[Finance - Information Ratio](Finance - Information Ratio.md)__

    When a portfolio is judged **against a benchmark**, what matters is how much it beat the benchmark by — and how *reliably*. These are the core metrics of active management: active return, tracking error, Information Ratio, and the appraisal ratio.

-   :material-shield-alert-outline: __[Finance - Jensens Alpha](Finance - Jensens Alpha.md)__

    Beta tells you how much of a portfolio's return came from simply riding the

-   :material-shield-alert-outline: __[Finance - Omega Ratio](Finance - Omega Ratio.md)__

    The Sharpe ratio squeezes a whole return distribution into a mean and a

-   :material-shield-alert-outline: __[Finance - Treynor Ratio](Finance - Treynor Ratio.md)__

    The Sharpe ratio divides excess return by total volatility, which mixes two

-   :material-shield-alert-outline: __[Risk Metrics](Risk Metrics.md)__

    This module gives you quick, professional stats about risk in any list or array of investment returns. It's used by investors, analysts, and students everywhere!

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
