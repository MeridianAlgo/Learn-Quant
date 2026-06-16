<p class="lq-badges"><span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-cat">Quantitative Methods</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Quantitative Methods - GARCH"
    python "garch.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Quantitative%20Methods%20-%20GARCH)

---
# GARCH Volatility Models

GARCH (Generalized Autoregressive Conditional Heteroskedasticity) captures **volatility clustering** — high-volatility days tend to follow high-volatility days. Used for risk forecasting, option pricing, and VaR.

## Functions

| Function | Description |
|---|---|
| `ewma_volatility(returns, lambda_)` | RiskMetrics EWMA conditional volatility |
| `fit_garch(returns)` | MLE estimation of GARCH(1,1) parameters |
| `garch_forecast(fit, last_return, horizon)` | Multi-step variance forecast |
| `garch_log_likelihood(params, returns)` | Gaussian negative log-likelihood |

## Model

GARCH(1,1):

```
sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
```

- **alpha**: ARCH term — reaction to recent shocks.
- **beta**: GARCH term — persistence of past variance.
- **alpha + beta**: persistence (must be < 1 for stationarity).
- **omega / (1 - alpha - beta)**: unconditional variance.

## Example

```python
from garch import fit_garch, garch_forecast

fit = fit_garch(returns)
print(fit['alpha'], fit['beta'], fit['persistence'])

vol_5d = garch_forecast(fit, returns[-1], horizon=5)
```

## Practical Notes

- Most equity GARCH fits show **alpha ~ 0.05-0.15, beta ~ 0.80-0.92**.
- Persistence near 1 → integrated GARCH (IGARCH) — shocks have permanent effects.
- For thicker tails, use Student-t innovations (extension).
- EWMA is GARCH(1,1) with omega=0 and fixed alpha+beta=1.


---

## Continue in Quantitative Methods

<div class="grid cards" markdown>

-   :material-function-variant: __[Quantitative Methods - Bayesian Inference](Quantitative Methods - Bayesian Inference.md)__

    A strategy wins 7 of its first 10 trades. Is its true win rate 70%? Almost

-   :material-function-variant: __[Quantitative Methods - Bootstrap](Quantitative Methods - Bootstrap.md)__

    The bootstrap estimates the sampling distribution of **any** statistic by resampling the observed data with replacement — no normality assumption required. It is the honest way to put confidence intervals around backtest metrics like Sharpe ratio, mean return, or maximum drawdown.

-   :material-function-variant: __[Quantitative Methods - Cointegration](Quantitative Methods - Cointegration.md)__

    Cointegration: two non-stationary series whose **linear combination is stationary**. Backbone of statistical arbitrage and pairs trading.

-   :material-function-variant: __[Quantitative Methods - Copulas](Quantitative Methods - Copulas.md)__

    This module demonstrates the concept of Copulas, specifically the Gaussian Copula, used in quantitative finance to model the dependency structure between multivariate random variables.

-   :material-function-variant: __[Quantitative Methods - Extreme Value Theory](Quantitative Methods - Extreme Value Theory.md)__

    Most risk models assume returns are normally distributed. They are not —

-   :material-function-variant: __[Quantitative Methods - Factor Models](Quantitative Methods - Factor Models.md)__

    Factor models explain asset returns as a linear combination of systematic **factors** plus a stock-specific residual. The **Fama-French 3-Factor Model (1992)** extended CAPM by adding two well-documented risk premia: the **Size premium** (SMB) and the **Value premium** (HML), dramatically improving the explanation of cross-sectional stock returns.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
