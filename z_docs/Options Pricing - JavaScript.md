# Options Pricing – JavaScript

## Overview

A pure JavaScript implementation of the Black-Scholes European options pricing model with all five Greeks and implied volatility via bisection. No external dependencies — runs directly in Node.js and can be imported as a module into any JS project.

## Concepts Covered
- Cumulative standard normal CDF (Abramowitz & Stegun approximation, error < 7.5e-8)
- Black-Scholes price formula for European calls and puts
- All five Greeks: delta, gamma, theta (daily decay), vega (per 1% vol move), rho (per 1% rate move)
- Implied volatility extraction via bisection search
- Put-call parity verification as a correctness check

## Files
- `blackScholes.js`: Self-contained module; exports `price`, `greeks`, `impliedVol`, `normCdf`, `normPdf`

## How to Run
```bash
node blackScholes.js
```
The demo runs with S=100, K=105, T=0.5yr, r=5%, σ=20% and prints prices, all Greeks, IV round-trip, and put-call parity residual for both call and put.

## API

```js
const { price, greeks, impliedVol } = require('./blackScholes');

price(S, K, T, r, sigma, type)      // 'call' | 'put' — returns premium
greeks(S, K, T, r, sigma, type)     // returns { delta, gamma, theta, vega, rho }
impliedVol(marketPrice, S, K, T, r, type)  // returns IV (decimal) or null
```

## Exported Functions

| Function | Description |
|---|---|
| `price` | Black-Scholes premium for a European call or put |
| `greeks` | All five Greeks in one call |
| `impliedVol` | Bisection IV solver; returns `null` if not found within tolerance |
| `normCdf` | Cumulative standard normal CDF |
| `normPdf` | Standard normal PDF |

## Practice Ideas
- Build an options chain by mapping `price` and `greeks` over a range of strikes
- Plot the IV smile by solving `impliedVol` against market quotes at different strikes
- Add a Newton-Raphson IV solver using vega and compare convergence speed

## Next Steps
- See the Python equivalent in `Black-Scholes Option Pricing/`
- Combine with `Monte Carlo Simulation - JavaScript/` to price path-dependent options
- Explore `Options Strategies/` for multi-leg payoff construction
