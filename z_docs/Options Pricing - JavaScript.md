# Options Pricing – JavaScript

<p class="lq-meta">Intermediate <span>/</span> Options, Derivatives & Finance <span>/</span> JavaScript <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Options%20Pricing%20-%20JavaScript">source</a></p>


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

---

## Running it

```bash
cd "Options Pricing - JavaScript"
node "blackScholes.js"
```

## More in Options, Derivatives & Finance

- [Advanced Options Pricing](Advanced Options Pricing.md): This module covers advanced mathematical techniques for pricing financial derivatives.
- [Black-Scholes Option Pricing](Black-Scholes Option Pricing.md): This module lets you price basic stock options (calls and puts) using the Black-Scholes formula, a foundation of modern financial analysis.
- [Bond Price and Yield](Bond Price and Yield.md): This utility lets you calculate the fair price of a bond or estimate its yield to maturity (YTM), two of the most basic (and important!) ideas in investing.
- [CAPM](CAPM.md): CAPM is the idea that won a Nobel Prize and still anchors how the industry thinks about return and risk.
- [Discounted Cash Flow (DCF)](Discounted Cash Flow (DCF).md): This tool calculates the present value of a series of future cash flows—the basic principle behind valuing businesses, real estate, projects, and stocks!
- [Dividend Tracker](Dividend Tracker.md): This tool lets you track upcoming dividends for a list of stocks, calculate expected income, and view a dividend calendar.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
