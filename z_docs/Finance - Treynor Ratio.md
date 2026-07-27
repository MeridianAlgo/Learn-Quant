<p class="lq-badges"><span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-cat">Risk & Performance</span><span class="lq-badge lq-lang">Python</span></p>

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

## Continue in Risk & Performance

<div class="grid cards" markdown>

-   :material-shield-alert-outline: __[Finance - Calmar Ratio](Finance - Calmar Ratio.md)__

    The Sharpe ratio judges a strategy by how much its returns wobble around their

-   :material-shield-alert-outline: __[Finance - Information Ratio](Finance - Information Ratio.md)__

    When a portfolio is judged **against a benchmark**, what matters is how much it beat the benchmark by — and how *reliably*. These are the core metrics of active management: active return, tracking error, Information Ratio, and the appraisal ratio.

-   :material-shield-alert-outline: __[Finance - Jensens Alpha](Finance - Jensens Alpha.md)__

    Beta tells you how much of a portfolio's return came from simply riding the

-   :material-shield-alert-outline: __[Finance - Omega Ratio](Finance - Omega Ratio.md)__

    The Sharpe ratio squeezes a whole return distribution into a mean and a

-   :material-shield-alert-outline: __[Finance - Performance Attribution](Finance - Performance Attribution.md)__

    Brinson decomposition splits portfolio active return into **allocation** and **selection** effects — answering *"did we beat the benchmark by picking the right sectors or the right stocks?"*

-   :material-shield-alert-outline: __[Risk Metrics](Risk Metrics.md)__

    This module gives you quick, professional stats about risk in any list or array of investment returns. It's used by investors, analysts, and students everywhere!

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
