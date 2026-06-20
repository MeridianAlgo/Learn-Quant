<p class="lq-badges"><span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-cat">Options, Derivatives & Finance</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Options Pricing - Binomial Tree"
    python "binomial_tree.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Options%20Pricing%20-%20Binomial%20Tree)

---
# Options Pricing — Binomial Tree

Black-Scholes hands you a price but hides the mechanics and **cannot value an
American option** — one you may exercise before expiry. The **binomial tree**
fixes both. It models the underlying as a lattice of up/down moves over `n`
discrete steps, then walks backward from expiry discounting risk-neutral
expected payoffs. American options drop out for free: at every node you simply
take the larger of *hold* and *exercise now*. And as `n` grows, the European
price **converges to Black-Scholes** — so you can watch the famous formula
emerge from a simple loop.

This module implements the **Cox-Ross-Rubinstein (CRR)** lattice from scratch.

## Functions

| Function | Description |
|---|---|
| `binomial_price(S, K, T, r, sigma, n, option, american, q)` | Price a vanilla call/put, European or American |
| `crr_parameters(sigma, T, n, r, q)` | The `(u, d, p)` up/down factors and risk-neutral probability |
| `implied_volatility(price, S, K, T, r, ...)` | Back out the volatility that reproduces an observed price (bisection) |

## The Cox-Ross-Rubinstein lattice

Split the life of the option into `n` steps of length `dt = T / n`. Over each
step the underlying multiplies by `u` (up) or `d` (down), chosen so the tree's
volatility matches the real one:

```
u = exp(sigma * sqrt(dt))        d = 1 / u
p = (exp((r - q) * dt) - d) / (u - d)
```

`p` is the **risk-neutral** probability of an up move — not a real-world
probability, but the one under which discounted prices are martingales, which is
all pricing needs. A valid (arbitrage-free) tree requires `d < exp((r-q)dt) < u`;
if that fails, `crr_parameters` raises, usually meaning `n` is too small for the
chosen `sigma`.

## Backward induction

1. Compute the payoff at every terminal node: `max(S_T - K, 0)` for a call.
2. Step backward: each node's value is the discounted average of its two
   children, `disc * (p * up + (1 - p) * down)`.
3. **For American options**, also compare against the immediate-exercise payoff
   at each node and keep the larger.

The implementation collapses one column of the lattice per step with NumPy, so
even thousands of steps are fast.

## Example

```python
from binomial_tree import binomial_price

# One-year at-the-money options, 5% rates, 20% vol.
euro_call = binomial_price(100, 100, 1.0, 0.05, 0.20, n=500, option="call")
amer_put  = binomial_price(100, 100, 1.0, 0.05, 0.20, n=500, option="put", american=True)

print(euro_call)   # ~10.45, matches Black-Scholes
print(amer_put)    # >= the European put — the early-exercise premium
```

## Why early exercise matters

A European put and an American put on a non-dividend payer differ because the
American holder can exercise a deep in-the-money put early and reinvest the
strike at the risk-free rate. The tree captures this automatically — the
American price is never below the European one, and the gap *is* the early
exercise premium.

## Practical notes

- **More steps, more accuracy.** The error shrinks like `1/n`, but the price
  oscillates as `n` alternates odd/even (the strike sits between nodes). For a
  smooth answer use a few hundred steps or average `n` and `n+1`.
- For the closed-form European benchmark, see
  [`Black-Scholes Option Pricing`](Black-Scholes Option Pricing.md).
- For the sensitivities (delta, gamma, vega…), see
  [`Finance - Greeks Calculator`](Finance - Greeks Calculator.md).
- For path-dependent payoffs the tree handles poorly (barriers, Asians), see
  [`Finance - Exotic Options`](Finance - Exotic Options.md).


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
