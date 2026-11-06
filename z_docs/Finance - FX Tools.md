# FX (Foreign Exchange) Tools

<p class="lq-meta">Intermediate <span>/</span> Options, Derivatives & Finance <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Finance%20-%20FX%20Tools">source</a></p>


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

## Running it

```bash
cd "Finance - FX Tools"
python "fx_tools.py"
```

## More in Options, Derivatives & Finance

- [Advanced Options Pricing](Advanced Options Pricing.md): This module covers advanced mathematical techniques for pricing financial derivatives.
- [Black-Scholes Option Pricing](Black-Scholes Option Pricing.md): This module lets you price basic stock options (calls and puts) using the Black-Scholes formula, a foundation of modern financial analysis.
- [Bond Price and Yield](Bond Price and Yield.md): This utility lets you calculate the fair price of a bond or estimate its yield to maturity (YTM), two of the most basic (and important!) ideas in investing.
- [CAPM](CAPM.md): CAPM is the idea that won a Nobel Prize and still anchors how the industry thinks about return and risk.
- [Discounted Cash Flow (DCF)](Discounted Cash Flow (DCF).md): This tool calculates the present value of a series of future cash flows—the basic principle behind valuing businesses, real estate, projects, and stocks!
- [Dividend Tracker](Dividend Tracker.md): This tool lets you track upcoming dividends for a list of stocks, calculate expected income, and view a dividend calendar.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
