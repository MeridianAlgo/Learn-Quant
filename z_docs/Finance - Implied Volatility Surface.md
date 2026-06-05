<p class="lq-badges"><span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-cat">Options, Derivatives & Finance</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Finance - Implied Volatility Surface"
    python "implied_vol_surface.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Finance%20-%20Implied%20Volatility%20Surface)

---
# Implied Volatility & the Volatility Surface

Black-Scholes turns *volatility into a price*. The market runs the formula
backwards: it quotes a price and we solve for the volatility that reproduces it.
That number is the **implied volatility (IV)** — the market's forward-looking
estimate of how much the underlying will move.

Collect IV across every strike and maturity and you get the **volatility
surface**. It is never flat, and the *shape* of that non-flatness (the smile,
the skew, the term structure) is what options traders actually trade.

## Functions

| Function | Description |
|---|---|
| `bs_price(S, K, T, r, sigma, option, q)` | Black-Scholes-Merton European price |
| `bs_vega(S, K, T, r, sigma, q)` | Vega — ∂price/∂σ, the engine of the IV solver |
| `implied_vol(price, S, K, T, r, option, q)` | Invert BS for IV (Newton + bisection) |
| `VolSurface` | Build & bilinearly interpolate a surface from a price grid |
| `VolSurface.smile(T)` | The IV smile across strikes at one maturity |

## Inverting Black-Scholes

There is no closed form for IV, so we solve `BS(σ) = market_price` numerically.
Newton's method converges fast because we know the derivative (vega):

```
σ_{n+1} = σ_n - (BS(σ_n) - price) / vega(σ_n)
```

Pure Newton can overshoot for deep in/out-of-the-money options where vega is
tiny, so this implementation keeps a bracket `[lo, hi]` and falls back to
**bisection** whenever a Newton step would leave it. The result is a solver that
is both fast *and* can't diverge.

## The surface

```python
import numpy as np
from implied_vol_surface import VolSurface, bs_price

S, r = 100.0, 0.02
strikes = np.array([80, 90, 100, 110, 120], dtype=float)
maturities = np.array([0.25, 0.5, 1.0], dtype=float)

# price_grid[i, j] = market price of the option at maturities[i], strikes[j]
surf = VolSurface(S, r).fit(strikes, maturities, price_grid, option="call")

surf.iv(105, 0.4)   # interpolated IV anywhere on the surface
surf.smile(0.5)     # the smile at the 6-month maturity
```

## Reading the shape

- **Skew** (equities): IV rises as strike falls. Crash protection (low-strike
  puts) is in demand, so it is priced richer — a fatter left tail than
  log-normal assumes.
- **Smile** (FX, single stocks): IV is high at both wings and lowest near the
  money.
- **Term structure**: IV usually rises with maturity in calm markets and
  *inverts* (short-dated richer) during stress.

## Practical notes

- **Garbage in, garbage out.** IV inversion amplifies quote noise; always sanity
  check that prices respect no-arbitrage bounds (`intrinsic ≤ price ≤ S` for a
  call). `implied_vol` returns `nan` when they don't.
- Bilinear interpolation is fine for queries *inside* the quoted grid. For a
  production surface you would fit a smooth parametric form (e.g. **SVI**) and
  enforce no calendar/butterfly arbitrage — a natural next step from here.
- This pairs directly with `Finance - Greeks Calculator`,
  `Black-Scholes Option Pricing` and `Finance - Volatility Calculator`
  (realised vs. implied).
- IV is an *annualised* number; to get an expected daily move, scale by
  `σ * sqrt(1/252)`.


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
