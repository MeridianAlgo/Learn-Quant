<p class="lq-badges"><span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-cat">Quantitative Methods</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Quantitative Methods - Bootstrap"
    python "bootstrap.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Quantitative%20Methods%20-%20Bootstrap)

---
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


---

## Continue in Quantitative Methods

<div class="grid cards" markdown>

-   :material-function-variant: __[Quantitative Methods - Cointegration](Quantitative Methods - Cointegration.md)__

    Cointegration: two non-stationary series whose **linear combination is stationary**. Backbone of statistical arbitrage and pairs trading.

-   :material-function-variant: __[Quantitative Methods - Copulas](Quantitative Methods - Copulas.md)__

    This module demonstrates the concept of Copulas, specifically the Gaussian Copula, used in quantitative finance to model the dependency structure between multivariate random variables.

-   :material-function-variant: __[Quantitative Methods - Extreme Value Theory](Quantitative Methods - Extreme Value Theory.md)__

    Most risk models assume returns are normally distributed. They are not —

-   :material-function-variant: __[Quantitative Methods - Factor Models](Quantitative Methods - Factor Models.md)__

    Factor models explain asset returns as a linear combination of systematic **factors** plus a stock-specific residual. The **Fama-French 3-Factor Model (1992)** extended CAPM by adding two well-documented risk premia: the **Size premium** (SMB) and the **Value premium** (HML), dramatically improving the explanation of cross-sectional stock returns.

-   :material-function-variant: __[Quantitative Methods - GARCH](Quantitative Methods - GARCH.md)__

    GARCH (Generalized Autoregressive Conditional Heteroskedasticity) captures **volatility clustering** — high-volatility days tend to follow high-volatility days. Used for risk forecasting, option pricing, and VaR.

-   :material-function-variant: __[Quantitative Methods - Interest Rate Models](Quantitative Methods - Interest Rate Models.md)__

    Continuous-time models for the evolution of the short (instantaneous) interest rate. Used for bond pricing, interest rate derivatives, and yield curve modeling.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
