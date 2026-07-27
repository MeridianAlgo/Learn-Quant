<p class="lq-badges"><span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-cat">Risk & Performance</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Finance - Information Ratio"
    python "information_ratio.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Finance%20-%20Information%20Ratio)

---
# Information Ratio & Active Management Metrics

When a portfolio is judged **against a benchmark**, what matters is how much it beat the benchmark by — and how *reliably*. These are the core metrics of active management: active return, tracking error, Information Ratio, and the appraisal ratio.

## Functions

| Function | Description |
|---|---|
| `active_returns(portfolio, benchmark)` | Period-by-period excess over benchmark |
| `tracking_error(portfolio, benchmark, periods)` | Annualised volatility of active returns |
| `information_ratio(portfolio, benchmark, periods)` | Active return per unit of tracking error |
| `appraisal_ratio(portfolio, benchmark, periods)` | Jensen's alpha ÷ residual vol, via CAPM regression |

## Key Concepts

- **Active return**: `r_portfolio − r_benchmark`. The value the manager added (or lost) versus simply holding the benchmark.
- **Tracking error**: the standard deviation of active returns, annualised. It is the *risk* of deviating from the benchmark.
- **Information Ratio (IR)**: `annualised active return / tracking error`. The benchmark-relative cousin of the Sharpe ratio. IR > 0.5 is good; > 1.0 is excellent and rare over long horizons.
- **Appraisal ratio**: from `r_p = α + β·r_b + ε`, it is annualised `α / σ(ε)` — skill (alpha) per unit of idiosyncratic risk, isolating stock-picking from market exposure.

## Example

```python
import numpy as np
from information_ratio import information_ratio, tracking_error, appraisal_ratio

bench = np.random.default_rng(0).normal(0.0004, 0.011, 504)
port = 0.0002 + 1.05 * bench + np.random.default_rng(1).normal(0, 0.004, 504)

print(tracking_error(port, bench))     # annualised TE
print(information_ratio(port, bench))  # IR
print(appraisal_ratio(port, bench))    # alpha, beta, residual vol, appraisal ratio
```

## Practical Notes

- Annualisation uses `mean × periods` for return and `std × √periods` for volatility, so the IR scales by `√periods`.
- Use `periods=252` for daily data, `52` for weekly, `12` for monthly.
- A high IR from a tiny tracking error can be statistically fragile — pair it with the [[bootstrap]] module to put a confidence interval around it.


---

## Continue in Risk & Performance

<div class="grid cards" markdown>

-   :material-shield-alert-outline: __[Finance - Calmar Ratio](Finance - Calmar Ratio.md)__

    The Sharpe ratio judges a strategy by how much its returns wobble around their

-   :material-shield-alert-outline: __[Finance - Jensens Alpha](Finance - Jensens Alpha.md)__

    Beta tells you how much of a portfolio's return came from simply riding the

-   :material-shield-alert-outline: __[Finance - Omega Ratio](Finance - Omega Ratio.md)__

    The Sharpe ratio squeezes a whole return distribution into a mean and a

-   :material-shield-alert-outline: __[Finance - Performance Attribution](Finance - Performance Attribution.md)__

    Brinson decomposition splits portfolio active return into **allocation** and **selection** effects — answering *"did we beat the benchmark by picking the right sectors or the right stocks?"*

-   :material-shield-alert-outline: __[Finance - Treynor Ratio](Finance - Treynor Ratio.md)__

    The Sharpe ratio divides excess return by total volatility, which mixes two

-   :material-shield-alert-outline: __[Risk Metrics](Risk Metrics.md)__

    This module gives you quick, professional stats about risk in any list or array of investment returns. It's used by investors, analysts, and students everywhere!

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
