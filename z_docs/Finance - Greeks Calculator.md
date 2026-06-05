<p class="lq-badges"><span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-cat">Options, Derivatives & Finance</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Finance - Greeks Calculator"
    python "greeks_calculator.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Finance%20-%20Greeks%20Calculator)

---
# Finance – Greeks Calculator

## Overview

The Options Greeks measure the sensitivity of an option's price to changes in underlying market parameters. They are the primary tools used by options traders and risk managers to understand, hedge, and monitor options positions.

This module implements all five first-order Greeks using the closed-form Black-Scholes formulas, covering both call and put options.

## Key Concepts

### The Black-Scholes Framework
Under Black-Scholes assumptions (log-normal prices, constant volatility, no dividends, continuous trading), option prices have closed-form solutions. The Greeks are the partial derivatives of these prices with respect to each input.

The d1 and d2 terms appear in every Greek:

```
d1 = [ln(S/K) + (r + 0.5*sigma^2)*T] / (sigma * sqrt(T))
d2 = d1 - sigma * sqrt(T)
```

### The Five Greeks

| Greek | Partial derivative | What it measures |
|-------|--------------------|-----------------|
| Delta (Δ) | dV/dS | Price change per $1 move in underlying |
| Gamma (Γ) | d²V/dS² | Rate of change of Delta |
| Theta (Θ) | dV/dt | Price decay per day passing |
| Vega (V) | dV/dσ | Price change per 1% move in volatility |
| Rho (ρ) | dV/dr | Price change per 1% move in interest rate |

### Delta
- Call delta: N(d1), ranges from 0 to 1.
- Put delta: N(d1) - 1, ranges from -1 to 0.
- Interpretation: a delta of 0.60 means the option behaves like holding 0.60 shares.
- Used for delta hedging: hold -delta shares of stock to be instantaneously market-neutral.

### Gamma
- Same formula for calls and puts: N'(d1) / (S * sigma * sqrt(T)).
- Highest for at-the-money options near expiry.
- Long gamma means you profit from large moves; short gamma means you suffer from them.
- Gamma hedging requires dynamic rebalancing of the delta hedge.

### Theta
- Typically negative (options lose value as time passes, all else equal).
- Largest in magnitude for at-the-money options near expiry.
- Short options strategies (selling premium) profit from theta decay.

### Vega
- Always positive for both calls and puts (more volatility = higher option value).
- Highest for at-the-money options with more time to expiry.
- Vega trading is about taking views on future realised or implied volatility.

### Rho
- Calls have positive rho (higher rates increase call value via cost-of-carry).
- Puts have negative rho.
- Rho matters most for long-dated options; negligible for short-dated options.

## Files
- `greeks_calculator.py`: d1/d2 computation, normal CDF/PDF implementations, and all five Greeks for calls and puts with worked examples.

## How to Run
```bash
python greeks_calculator.py
```

## Financial Applications

### 1. Delta Hedging
- Maintain a delta-neutral portfolio by offsetting option delta with a position in the underlying.
- Rebalance dynamically as the underlying price moves (gamma causes delta to drift).

### 2. Portfolio Risk Reporting
- Aggregate delta, gamma, vega, and theta across all positions to understand total exposure.
- "Greeks P&L attribution": explain daily P&L in terms of each Greek's contribution.

### 3. Options Market Making
- Market makers quote bid-ask prices on options and immediately hedge the Greeks.
- Target: flat delta, long gamma (benefit from realised vol exceeding implied vol).

### 4. Volatility Trading
- Trade vega by going long options (long vega) when implied vol is cheap.
- Gamma scalping: dynamically delta-hedge a long gamma position to monetise realised volatility.

### 5. Risk Limits
- Risk management systems enforce limits on total portfolio delta (directional exposure), vega (vol exposure), and gamma (curvature risk).

## Best Practices

- **Greeks are local approximations**: Delta and Gamma are instantaneous measures; for large moves, higher-order Greeks (vanna, volga) become significant.
- **Rebalance frequency depends on gamma**: High-gamma positions (near-expiry ATM options) require frequent hedging; low-gamma positions tolerate less frequent rebalancing.
- **Implied vol, not realised vol**: When computing Greeks for trading decisions, use the market's implied volatility, not historical volatility.
- **Check sign conventions**: Theta is usually quoted as a negative number (daily decay); Vega is quoted per 1% or per 1-vol-point change depending on convention.


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
