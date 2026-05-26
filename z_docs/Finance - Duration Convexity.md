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
