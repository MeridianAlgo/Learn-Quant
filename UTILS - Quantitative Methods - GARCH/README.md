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
