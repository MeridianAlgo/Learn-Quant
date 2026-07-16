<p class="lq-badges"><span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-cat">Options, Derivatives & Finance</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Finance - Treynor Ratio"
    python "treynor_ratio.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Finance%20-%20Treynor%20Ratio)

---
# Finance, Treynor Ratio

The Sharpe ratio divides excess return by total volatility, which mixes two
very different kinds of risk. Part of a portfolio's movement comes from the
market itself and part is its own private noise. Inside a well diversified
book the private noise mostly cancels out, so the only risk you are truly
paid to carry is the market exposure, measured by beta. The Treynor ratio
therefore divides excess return by beta instead of by standard deviation. It
asks how much reward the portfolio earns per unit of market risk it actually
adds to your book.

## The pieces

* `beta(asset_returns, market_returns)` measures how strongly the asset moves
  with the market, computed as the covariance with the market divided by the
  market's variance.
* `treynor_ratio(asset_returns, market_returns, risk_free_rate)` divides the
  mean excess return by that beta. The risk free rate is quoted per period,
  matching the return series.
* `annualized_treynor_ratio(...)` scales the per period figure by the number
  of periods per year. Beta is dimensionless, so only the numerator scales.

## Reading the number

A higher Treynor ratio means more excess return for each unit of beta
carried. A fund with a beta of half and a fund with a beta of two can both
beat the market, but the low beta fund doing so has generated its edge with
far less borrowed market exposure, and its Treynor ratio will say so. Two
funds with identical Sharpe ratios can have very different Treynor ratios
when one of them hides a lot of diversifiable noise inside its volatility.

## When to prefer it over Sharpe

Use Treynor when the portfolio being judged is one sleeve of a larger
diversified book, because there the idiosyncratic wiggle washes out and beta
is the risk that remains. Use Sharpe when the portfolio is the whole
investment, because then every unit of volatility is felt. The two ratios
agree on a perfectly diversified index fund and disagree most on
concentrated, idiosyncratic strategies.

## Edge cases

A beta near zero makes the ratio explode toward infinity, and a negative
beta flips its sign, so a market neutral book should not be judged on
Treynor at all. The implementation raises on zero beta rather than
returning a misleading number.

## Example

```python
from treynor_ratio import beta, annualized_treynor_ratio

fund = [0.004, -0.002, 0.006, -0.001, 0.003, 0.002, -0.004, 0.005]
market = [0.005, -0.003, 0.004, -0.002, 0.004, 0.001, -0.005, 0.006]

print(beta(fund, market))
print(annualized_treynor_ratio(fund, market, risk_free_rate=0.0001))
```

## Where to go next

* For the total volatility cousin see
  [`Sharpe and Sortino Ratio`](Sharpe and Sortino Ratio.md).
* For beta itself in more depth see
  [`Finance - Beta Calculator`](Finance - Beta Calculator.md).
* For the model that gives beta its meaning see [`CAPM`](CAPM.md).


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
