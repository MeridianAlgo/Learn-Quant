# Bootstrap Resampling

The bootstrap estimates the sampling distribution of **any** statistic by resampling the observed data with replacement — no normality assumption required. It is the honest way to put confidence intervals around backtest metrics like Sharpe ratio, mean return, or maximum drawdown.

## Functions

| Function | Description |
|---|---|
| `iid_bootstrap(data, statistic, n_boot)` | Resample individual observations (assumes no serial dependence) |
| `block_bootstrap(data, statistic, block_size, n_boot)` | Resample contiguous blocks — preserves autocorrelation |
| `stationary_bootstrap(data, statistic, expected_block, n_boot)` | Politis–Romano: random geometric block lengths, circular wrap |
| `confidence_interval(estimates, alpha)` | Percentile confidence interval from bootstrap estimates |

## Key Concepts

- **Why bootstrap?** Financial returns are fat-tailed and serially correlated, so parametric (normal-theory) confidence intervals are usually wrong. Resampling makes no distributional assumption.
- **i.i.d. vs. block**: the i.i.d. bootstrap destroys time structure. For returns with autocorrelation or volatility clustering, use a **block** method so each resample preserves short-term dependence.
- **Stationary bootstrap**: blocks of *random* (geometric) length with circular wrap-around guarantee the resampled series is stationary, avoiding the fixed-block boundary artefacts.
- **Percentile CI**: take the 2.5th and 97.5th percentiles of the bootstrap estimates for a 95% interval.

## Example

```python
import numpy as np
from bootstrap import block_bootstrap, confidence_interval

returns = np.random.default_rng(0).normal(0.0005, 0.012, 504)
sharpe = lambda x: x.mean() / x.std(ddof=1) * np.sqrt(252)

est = block_bootstrap(returns, sharpe, block_size=20, n_boot=2000, seed=1)
print(confidence_interval(est))   # 95% CI for the annualised Sharpe
```

## Practical Notes

- A typical rule of thumb for the average block length is `n^(1/3)` for the stationary bootstrap.
- Use `n_boot >= 1000` for stable percentile intervals; `2000–10000` for tail statistics.
- The bootstrap quantifies *sampling* uncertainty — it cannot rescue a backtest that suffers from look-ahead bias or overfitting.
