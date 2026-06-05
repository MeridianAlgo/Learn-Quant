<p class="lq-badges"><span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-cat">Options, Derivatives & Finance</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Finance - Yield Curve"
    python "yield_curve_tutorial.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Finance%20-%20Yield%20Curve)

---
# Finance – Yield Curve

## Overview

The yield curve is the most closely watched chart in global finance. It plots interest rates (yields) across different maturities for bonds of equal credit quality — most commonly US Treasury bonds. Its shape and movements drive pricing for virtually every financial asset, from mortgages to corporate bonds to equity discount rates.

This utility demonstrates how to construct, smooth, and interpret a yield curve using the **Nelson-Siegel parametric model** and implied **forward rate extraction**.

## Key Concepts

### **Types of Rates**
| Rate Type | Description |
|-----------|-------------|
| **Par Yield** | Coupon rate making a bond price equal to face value ($100) |
| **Zero / Spot Rate** | Yield on a zero-coupon bond — the pure rate for that maturity |
| **Forward Rate** | Implied rate for a *future* period, derived from today's spot curve |

### **Yield Curve Shapes**
| Shape | Description | Economic Signal |
|-------|-------------|-----------------|
| Normal | Long rates > Short rates | Growth expected |
| Inverted | Short rates > Long rates | Recession warning |
| Flat | Short ≈ Long rates | Transition / uncertainty |
| Humped | Mid > both ends | Complex rate expectations |

> The 2s10s spread (10yr minus 2yr yield) turning negative has preceded every US recession since 1955.

### **Nelson-Siegel Model**
A 4-parameter model that fits a smooth curve through noisy market yields:

```
y(T) = beta0
 + beta1 × [(1 – e^(–T/τ)) / (T/τ)] ← slope loading
 + beta2 × [(1 – e^(–T/τ)) / (T/τ) – e^(–T/τ)] ← curvature loading
```

- **beta0** (Level): Long-run asymptotic yield
- **beta1** (Slope): Short-to-long rate difference
- **beta2** (Curvature): Hump or trough at intermediate maturities
- **tau** (Decay): Controls where the slope/curvature effects peak

## Logic Implemented

1. **Raw yield data** — Hardcoded Treasury par yields representing a high-rate environment
2. **Curve classification** — Categorises the shape (normal/inverted/flat/humped)
3. **Nelson-Siegel fitting** — Minimises squared errors via `scipy.optimize.minimize`
4. **Forward rate extraction** — Derives the implied path of future short rates

## Files
- `yield_curve_tutorial.py`: Sample data, Nelson-Siegel model, fitting engine, forward rate computation, and shape analysis.

## How to Run
```bash
python yield_curve_tutorial.py
```

## Financial Applications

### 1. Bond Pricing
- Zero-coupon (spot) rates are used to discount each cash flow from a coupon bond.
- The yield curve determines the "risk-free discount factors" for all maturities.

### 2. Interest Rate Derivatives
- Swap rates, caps, floors, and swaptions are all priced off the forward rate curve.
- The Nelson-Siegel model provides a smooth forward curve without oscillations.

### 3. Central Bank Policy Analysis
- The short end of the curve is heavily influenced by the Fed Funds rate.
- The long end reflects growth/inflation expectations and term premium.

### 4. Credit Spread Analysis
- Corporate bond yields = Treasury yield (risk-free) + credit spread.
- The yield curve provides the risk-free baseline for all credit analysis.

### 5. Mortgage-Backed Securities
- Mortgage rates are typically benchmarked off the 10-year Treasury yield.

## Best Practices

- **Use zero rates (not par rates) for discounting**: Par rates mix maturity effects; spot rates are "pure" rates for each specific maturity.
- **Forward rates are volatile**: Small changes in the spot curve create large swings in forward rates at long maturities.
- **Nelson-Siegel has limits**: It cannot fit bimodal or heavily distorted curves — use cubic splines for more complex shapes.
- **Data sources**: Free access to daily US Treasury par yields via [FRED (St. Louis Fed)](https://fred.stlouisfed.org/series/DGS10).

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
