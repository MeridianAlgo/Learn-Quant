# Short Rate Interest Rate Models

<p class="lq-meta">Advanced <span>/</span> Quantitative Methods <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Quantitative%20Methods%20-%20Interest%20Rate%20Models">source</a></p>


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

## Running it

```bash
cd "Quantitative Methods - Interest Rate Models"
python "interest_rate_models.py"
```

## More in Quantitative Methods

- [Quantitative Methods - Bayesian Inference](Quantitative Methods - Bayesian Inference.md): A strategy wins 7 of its first 10 trades.
- [Quantitative Methods - Bootstrap](Quantitative Methods - Bootstrap.md): The bootstrap estimates the sampling distribution of any statistic by resampling the observed data with replacement — no normality assumption required.
- [Quantitative Methods - Cointegration](Quantitative Methods - Cointegration.md): Cointegration: two non-stationary series whose linear combination is stationary.
- [Quantitative Methods - Copulas](Quantitative Methods - Copulas.md): This module demonstrates the concept of Copulas, specifically the Gaussian Copula, used in quantitative finance to model the dependency structure between multivariate random variables.
- [Quantitative Methods - Entropy](Quantitative Methods - Entropy.md): Entropy measures how surprised you should expect to be.
- [Quantitative Methods - Extreme Value Theory](Quantitative Methods - Extreme Value Theory.md): Most risk models assume returns are normally distributed.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
