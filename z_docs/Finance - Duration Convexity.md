# Bond Duration, Convexity, and DV01

<p class="lq-meta">Intermediate <span>/</span> Options, Derivatives & Finance <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Finance%20-%20Duration%20Convexity">source</a></p>


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

## Running it

```bash
cd "Finance - Duration Convexity"
python "duration_convexity.py"
```

## More in Options, Derivatives & Finance

- [Advanced Options Pricing](Advanced Options Pricing.md): This module covers advanced mathematical techniques for pricing financial derivatives.
- [Black-Scholes Option Pricing](Black-Scholes Option Pricing.md): This module lets you price basic stock options (calls and puts) using the Black-Scholes formula, a foundation of modern financial analysis.
- [Bond Price and Yield](Bond Price and Yield.md): This utility lets you calculate the fair price of a bond or estimate its yield to maturity (YTM), two of the most basic (and important!) ideas in investing.
- [CAPM](CAPM.md): CAPM is the idea that won a Nobel Prize and still anchors how the industry thinks about return and risk.
- [Discounted Cash Flow (DCF)](Discounted Cash Flow (DCF).md): This tool calculates the present value of a series of future cash flows—the basic principle behind valuing businesses, real estate, projects, and stocks!
- [Dividend Tracker](Dividend Tracker.md): This tool lets you track upcoming dividends for a list of stocks, calculate expected income, and view a dividend calendar.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
