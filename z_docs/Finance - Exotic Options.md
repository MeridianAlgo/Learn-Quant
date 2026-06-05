<p class="lq-badges"><span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-cat">Options, Derivatives & Finance</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Finance - Exotic Options"
    python "exotic_options.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Finance%20-%20Exotic%20Options)

---
# Exotic Options Pricing

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

## Continue in Options, Derivatives & Finance

<div class="grid cards" markdown>

-   :material-chart-bell-curve: __[Advanced Options Pricing](Advanced Options Pricing.md)__

    This module covers advanced mathematical techniques for pricing financial derivatives. The focus is on models beyond the standard assumptions. Rather than assuming constant volatility, we explore dynamic and local volatility models. These models are crucial for correctly valuing exotic options and managing the risks of complex derivatives portfolios.

-   :material-chart-bell-curve: __[Black-Scholes Option Pricing](Black-Scholes Option Pricing.md)__

    This module lets you price basic stock options (calls and puts) using the Black-Scholes formula, a foundation of modern financial analysis.

-   :material-chart-bell-curve: __[Bond Price and Yield](Bond Price and Yield.md)__

    This utility lets you calculate the fair price of a bond or estimate its yield to maturity (YTM), two of the most basic (and important!) ideas in investing.

-   :material-chart-bell-curve: __[CAPM](CAPM.md)__

    CAPM is the idea that won a Nobel Prize and still anchors how the industry

-   :material-chart-bell-curve: __[Discounted Cash Flow (DCF)](Discounted Cash Flow (DCF).md)__

    This tool calculates the present value of a series of future cash flows—the basic principle behind valuing businesses, real estate, projects, and stocks!

-   :material-chart-bell-curve: __[Dividend Tracker](Dividend Tracker.md)__

    **This utility does NOT use any external APIs.** All data is managed locally for learning and experimentation.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
