# Monte Carlo Simulation – JavaScript

## Overview

A pure JavaScript Monte Carlo engine for portfolio simulation and European option pricing via geometric Brownian motion (GBM). Implements correlated multi-asset paths using Cholesky decomposition, antithetic variates for variance reduction, and VaR/CVaR estimation from the simulated return distribution. No external dependencies — runs directly in Node.js.

## Concepts Covered
- Box-Muller transform for standard-normal random variate generation
- Single-asset GBM path simulation
- Cholesky decomposition to impose user-defined asset correlations
- Multi-asset correlated portfolio simulation with equity curve tracking
- European call option pricing via risk-neutral GBM with antithetic variates
- Value-at-Risk (VaR) and Conditional VaR (CVaR) from simulated return distributions

## Files
- `monteCarlo.js`: Self-contained module; exports `simulatePath`, `simulatePortfolio`, `mcOptionPrice`, `varCvar`, `randNormal`, `cholesky`

## How to Run
```bash
node monteCarlo.js
```
The demo runs three scenarios and prints results: a single GBM price path, a 10,000-path three-asset portfolio with risk metrics, and a 50,000-sim European call priced against the Black-Scholes analytical value.

## API

```js
const { simulatePath, simulatePortfolio, mcOptionPrice, varCvar } = require('./monteCarlo');

simulatePath(S0, mu, sigma, T, steps)
// Returns number[] — a single GBM price path of length steps + 1

simulatePortfolio(S0, mu, sigma, corr, weights, T, steps, nSims)
// Returns { finalReturns: number[], samplePaths: number[][] }
// samplePaths contains equity curves for the first 5 simulations

mcOptionPrice(S, K, T, r, sigma, nSims?)
// Returns { price: number, stdErr: number }

varCvar(returns, confidence?)
// Returns { var: number, cvar: number }
```

## Exported Functions

| Function | Description |
|---|---|
| `simulatePath` | Single GBM path (log-normal dynamics) |
| `simulatePortfolio` | Multi-asset correlated paths via Cholesky; tracks equity curves |
| `mcOptionPrice` | Risk-neutral European call pricing with antithetic variates |
| `varCvar` | VaR and CVaR from a sorted returns array |
| `randNormal` | Box-Muller standard-normal sampler |
| `cholesky` | Lower-triangular Cholesky factor of a correlation matrix |

## Practice Ideas
- Price a European put and verify put-call parity against `Options Pricing - JavaScript/`
- Extend `simulatePortfolio` to compute drawdown statistics across all paths
- Add an Asian option pricer by averaging the price path instead of using only the terminal value

## Next Steps
- Compare MC option prices to the analytical Black-Scholes formula in `Options Pricing - JavaScript/`
- See the Python Monte Carlo equivalent in `Quantitative Methods - Stochastic Processes/`
- Apply VaR/CVaR in a full risk pipeline with `Risk Metrics/` and `Value at Risk/`
