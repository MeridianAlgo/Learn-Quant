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
