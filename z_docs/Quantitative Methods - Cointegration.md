<p class="lq-badges"><span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-cat">Quantitative Methods</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Quantitative Methods - Cointegration"
    python "cointegration.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Quantitative%20Methods%20-%20Cointegration)

---
# Cointegration & Pairs Trading Foundations

Cointegration: two non-stationary series whose **linear combination is stationary**. Backbone of statistical arbitrage and pairs trading.

## Functions

| Function | Description |
|---|---|
| `ols_hedge_ratio(y, x)` | OLS regression for hedge ratio + residuals |
| `adf_test(series, lags)` | Augmented Dickey-Fuller unit-root test |
| `engle_granger(y, x, lags)` | Two-step cointegration test |
| `half_life(spread)` | Mean-reversion half-life via OU fit |
| `zscore_spread(spread, window)` | Rolling z-score for entry/exit signals |

## ADF Critical Values (one-sided)

| Significance | Critical t-stat |
|---|---|
| 1% | -3.43 |
| 5% | -2.86 |
| 10% | -2.57 |

t-stat **more negative** than the threshold → reject unit root → stationary.

## Example

```python
from cointegration import engle_granger, half_life, zscore_spread

eg = engle_granger(prices_a, prices_b, lags=1)
if eg['cointegrated_5pct']:
    spread = eg['spread']
    z = zscore_spread(spread, window=60)
    hl = half_life(spread)
    # Enter when |z| > 2, exit when |z| < 0.5
```

## Practical Notes

- Half-life > 30 days → spread too slow, transaction costs eat profits.
- Half-life < 1 day → likely noise, not true mean reversion.
- Cointegration relationships **break** — re-test rolling windows.
- For >2 assets use Johansen's test (not implemented here).


---

## Continue in Quantitative Methods

<div class="grid cards" markdown>

-   :material-function-variant: __[Quantitative Methods - Bootstrap](Quantitative Methods - Bootstrap.md)__

    The bootstrap estimates the sampling distribution of **any** statistic by resampling the observed data with replacement — no normality assumption required. It is the honest way to put confidence intervals around backtest metrics like Sharpe ratio, mean return, or maximum drawdown.

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
