<p class="lq-badges"><span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-cat">Quantitative Methods</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Quantitative Methods - Interest Rate Models"
    python "interest_rate_models.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Quantitative%20Methods%20-%20Interest%20Rate%20Models)

---
# Short Rate Interest Rate Models

Continuous-time models for the evolution of the short (instantaneous) interest rate. Used for bond pricing, interest rate derivatives, and yield curve modeling.

## Functions

| Function | Description |
|---|---|
| `vasicek_simulate(r0, kappa, theta, sigma, T, n_steps, n_paths)` | Simulate Vasicek paths |
| `vasicek_bond_price(r0, kappa, theta, sigma, T)` | Closed-form ZCB price |
| `vasicek_yield(r0, kappa, theta, sigma, T)` | Zero-coupon yield |
| `cir_simulate(r0, kappa, theta, sigma, T, n_steps, n_paths)` | Simulate CIR paths |
| `cir_bond_price(r0, kappa, theta, sigma, T)` | Closed-form ZCB price |
| `cir_yield(r0, kappa, theta, sigma, T)` | Zero-coupon yield |
| `term_structure(r0, kappa, theta, sigma, maturities, model)` | Full yield curve |

## Models

### Vasicek (1977)
`dr = kappa*(theta - r)*dt + sigma*dW`
- Mean-reverting: rate pulled toward theta at speed kappa
- Rates can go negative (unrealistic but analytically convenient)
- Closed-form bond prices

### Cox-Ingersoll-Ross (1985)
`dr = kappa*(theta - r)*dt + sigma*sqrt(r)*dW`
- Mean-reverting + non-negative rates (when `2*kappa*theta >= sigma²`)
- Volatility scales with rate level (more realistic)
- Closed-form bond prices

## Parameters

| Param | Typical Range | Meaning |
|---|---|---|
| kappa | 0.1–1.0 | Mean reversion speed (0.3 = ~3yr half-life) |
| theta | 0.03–0.07 | Long-run mean rate |
| sigma | 0.005–0.02 | Rate volatility |

## Example

```python
from interest_rate_models import term_structure

yields = term_structure(r0=0.03, kappa=0.3, theta=0.05, sigma=0.01,
                        maturities=[1, 2, 5, 10, 30], model="cir")
```


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
