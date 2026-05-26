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
