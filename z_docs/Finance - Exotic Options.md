# Exotic Options Pricing

<p class="lq-meta">Intermediate <span>/</span> Options, Derivatives & Finance <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Finance%20-%20Exotic%20Options">source</a></p>


Monte Carlo pricing for path-dependent options that have no simple closed-form solution (or where the path matters, not just the terminal price).

## Functions

| Function | Description |
|---|---|
| `barrier_option(S0, K, H, r, sigma, T, ...)` | Down/up knock-in/knock-out options |
| `asian_option(S0, K, r, sigma, T, ...)` | Arithmetic or geometric average price |
| `lookback_option(S0, r, sigma, T, ...)` | Floating or fixed strike lookback |

## Option Types

### Barrier Options
Price path must cross (or stay away from) a barrier level H.
- **Down-and-Out**: Expires worthless if S drops below H
- **Down-and-In**: Only activates if S drops below H
- **Up-and-Out / Up-and-In**: Symmetric for upside barriers
- Key identity: `Down-Out + Down-In = Vanilla` (in/out parity)

### Asian Options
Payoff based on average price, reducing volatility exposure and manipulation risk.
- **Arithmetic average**: `max(avg(S) - K, 0)` — no closed form
- **Geometric average**: Has a closed-form approximation (Kemna-Vorst)

### Lookback Options
Benefit from hindsight — payoff uses the best price during the option's life.
- **Floating strike call**: `S_T - min(S)` — "buy at the lowest price"
- **Fixed strike call**: `max(max(S) - K, 0)` — "use the highest price"

## Example

```python
from exotic_options import barrier_option, asian_option

# Down-and-out call: knocked out if price drops below 90
price = barrier_option(S0=100, K=100, H=90, r=0.05, sigma=0.20, T=1.0,
                       barrier_type="down-out")

# Asian call (arithmetic average)
asian_price = asian_option(S0=100, K=100, r=0.05, sigma=0.20, T=1.0)
```

---

## Running it

```bash
cd "Finance - Exotic Options"
python "exotic_options.py"
```

## More in Options, Derivatives & Finance

- [Advanced Options Pricing](Advanced Options Pricing.md): This module covers advanced mathematical techniques for pricing financial derivatives.
- [Black-Scholes Option Pricing](Black-Scholes Option Pricing.md): This module lets you price basic stock options (calls and puts) using the Black-Scholes formula, a foundation of modern financial analysis.
- [Bond Price and Yield](Bond Price and Yield.md): This utility lets you calculate the fair price of a bond or estimate its yield to maturity (YTM), two of the most basic (and important!) ideas in investing.
- [CAPM](CAPM.md): CAPM is the idea that won a Nobel Prize and still anchors how the industry thinks about return and risk.
- [Discounted Cash Flow (DCF)](Discounted Cash Flow (DCF).md): This tool calculates the present value of a series of future cash flows—the basic principle behind valuing businesses, real estate, projects, and stocks!
- [Dividend Tracker](Dividend Tracker.md): This tool lets you track upcoming dividends for a list of stocks, calculate expected income, and view a dividend calendar.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
