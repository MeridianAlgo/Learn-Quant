<p class="lq-badges"><span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-cat">Options, Derivatives & Finance</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Finance - Duration Convexity"
    python "duration_convexity.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Finance%20-%20Duration%20Convexity)

---
# Bond Duration, Convexity, and DV01

Fixed income sensitivity measures that quantify how bond prices respond to changes in interest rates.

## Functions

| Function | Description |
|---|---|
| `bond_price(cashflows, times, ytm)` | PV of cash flows at given yield |
| `macaulay_duration(cashflows, times, ytm)` | Weighted average time to cash flows (years) |
| `modified_duration(cashflows, times, ytm)` | % price change per 1% yield change |
| `convexity(cashflows, times, ytm)` | Second-order yield sensitivity |
| `dv01(cashflows, times, ytm)` | Dollar value of 1 basis point |
| `price_change_approx(mod_dur, conv, price, dy)` | Taylor expansion price estimate |
| `build_cashflows(face, coupon_rate, maturity, freq)` | Generate coupon bond cash flows |

## Key Concepts

- **Macaulay Duration**: Measures the weighted average maturity of cash flows. Zero-coupon bond has duration = maturity.
- **Modified Duration**: `D_mod = D_mac / (1 + y)`. A bond with mod duration 7 loses ~7% in price per +100bp yield move.
- **Convexity**: The curve in the price-yield relationship. Positive convexity benefits investors — price rises more than duration predicts when yields fall.
- **DV01**: Practical measure for hedging. "My portfolio has DV01 of $5,000" means a +1bp move costs $5,000.

## Example

```python
from duration_convexity import build_cashflows, bond_price, modified_duration, dv01

cfs, ts = build_cashflows(face=1000, coupon_rate=0.05, maturity=10)
price = bond_price(cfs, ts, ytm=0.04)     # ~1081.11
mod_dur = modified_duration(cfs, ts, 0.04)  # ~7.99
dv = dv01(cfs, ts, 0.04)                    # ~0.086 per $1000
```

## Duration Approximation

```
ΔP ≈ -D_mod × P × Δy + 0.5 × Convexity × P × Δy²
```

For a +100bp shock: duration term dominates. For large moves, convexity correction matters.


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
