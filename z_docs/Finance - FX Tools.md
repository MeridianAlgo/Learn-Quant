<p class="lq-badges"><span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-cat">Options, Derivatives & Finance</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Finance - FX Tools"
    python "fx_tools.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Finance%20-%20FX%20Tools)

---
# FX (Foreign Exchange) Tools

Core analytics for foreign exchange markets: no-arbitrage pricing, option valuation, and cross-rate calculations.

## Functions

| Function | Description |
|---|---|
| `forward_rate(spot, r_d, r_f, T)` | CIP-implied forward exchange rate |
| `forward_points(spot, r_d, r_f, T)` | Forward-spot differential in pips |
| `cip_deviation(spot, forward, r_d, r_f, T)` | Covered Interest Parity basis (bps) |
| `cross_rate(s_ab, s_ac)` | Derive B/C rate from two pairs |
| `triangular_arbitrage_profit(s_ab, s_bc, s_ca)` | Detect/quantify triangular arb |
| `garman_kohlhagen(S, K, r_d, r_f, sigma, T, type)` | European FX option pricing |

## Key Concepts

### Covered Interest Rate Parity (CIP)
`F = S × exp((r_d - r_f) × T)`
In theory, no arbitrage → forward rate is fully determined by spot + rate differential. In practice, CIP deviations (the "FX basis") are a significant source of hedge fund alpha.

### Forward Points
Market convention: quote forward as "pips" above/below spot. Forward points = `(F - S) / pip_size`. Positive when domestic rate > foreign rate.

### Garman-Kohlhagen
Black-Scholes extension for FX options. Foreign rate acts as a continuous dividend yield. Delta is expressed in domestic currency terms.

## Example

```python
from fx_tools import forward_rate, garman_kohlhagen, cip_deviation

# USD/EUR spot = 1.10, US rate 5%, EU rate 2%
F = forward_rate(1.10, r_domestic=0.05, r_foreign=0.02, T=1.0)  # ~1.1332

# FX call option
call = garman_kohlhagen(S=1.10, K=1.10, r_d=0.05, r_f=0.02, sigma=0.10, T=0.25)
```

## Pip Conventions

| Pair | Pip size |
|---|---|
| EUR/USD, GBP/USD | 0.0001 |
| USD/JPY | 0.01 |
| Most others | 0.0001 |


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
