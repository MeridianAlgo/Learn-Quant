<p class="lq-badges"><span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-cat">Risk & Performance</span><span class="lq-badge lq-lang">JavaScript</span></p>

!!! tip "Run this module"
    ```bash
    cd "Risk Metrics - JavaScript"
    node "riskMetrics.js"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Risk%20Metrics%20-%20JavaScript)

---
# Risk Metrics, JavaScript

Every number in this file answers the same question from a different angle.
How much pain does this strategy inflict, and is the return worth it. A return
figure on its own tells you almost nothing, because doubling the leverage
doubles it and nobody learned anything about the strategy in the process.

Pure Node.js, no dependencies. Run it with `node riskMetrics.js`.

## The pieces

* `toReturns(prices)` and `equityCurve(returns)` convert between the two forms
  everything else expects.
* `annualizedReturn(returns, periods)` and `annualizedVolatility(returns,
  periods)` scale a period figure up to a year.
* `maxDrawdown(returns)` returns the worst peak to trough fall along with
  where it started, where it bottomed and whether it ever recovered.
* `sharpeRatio(returns, riskFreeRate, periods)` is excess return per unit of
  total volatility.
* `sortinoRatio(returns, target, periods)` is the same idea counting only
  downside deviation.
* `calmarRatio(returns, periods)` is annual return divided by the worst
  drawdown.
* `historicalVaR(returns, confidence)` and `conditionalVaR(returns,
  confidence)` measure the tail.
* `ulcerIndex(returns)` measures how deep and how long the underwater periods
  were.
* `riskReport(returns, riskFreeRate, periods)` returns all of it in one object.

## Drawdown is not high minus low

The maximum drawdown is the worst fall from a running peak, and the running
part is what people get wrong. Subtracting the lowest point of a series from
its highest gives the right answer only when the high happens to come first.
Any series that dipped early and rallied later will report a drawdown it never
suffered. The implementation here walks the curve forward and tracks the peak
as it goes, which is the only way to get this right.

The function also returns the recovery point, or minus one when the strategy
never made it back. A strategy still underwater at the end of the sample is a
different proposition from one that fell just as far and climbed out in a
month, and a single drawdown percentage hides that completely.

## Sharpe, Sortino and the volatility that is not risk

Sharpe divides excess return by total volatility, which quietly treats a big
up day as being just as bad as a big down day. Nobody holding the position
agrees with that. Sortino fixes it by putting only the returns below the
target into the denominator, so upside moves stop counting against you.

Sortino is therefore always the friendlier of the two, and the gap between
them tells you something. A strategy whose two ratios are close has roughly
symmetric returns. One whose Sortino sits far above its Sharpe is making its
money in occasional large jumps, and that shape has its own problems, chiefly
that you cannot tell from the ratio whether the jumps continue.

Both ratios are silent on path. A strategy can post a fine Sharpe while
spending two years underwater, which is why Calmar and the ulcer index exist.
Calmar puts the return next to the worst hole. The ulcer index goes further
and measures every moment spent below the previous high, so a long shallow
grind scores worse than a sharp fall that recovered quickly, which matches how
it actually feels to hold.

## What historical VaR cannot do

Historical VaR is the empirical quantile of the sample, which is its great
virtue and its great limitation in the same breath. It assumes nothing about
the shape of the distribution, so fat tails and skew are included for free
provided they appeared in your sample. What it cannot do is imagine a loss
larger than any it has seen. Feed it a calm year and it will quietly tell you
that calm is all there is.

Conditional VaR at least reports the average of the losses beyond the cutoff
rather than stopping at the cutoff itself, so it says something about how bad
the bad days are instead of only where they begin. It is the better number for
sizing a position, and it inherits the same blindness to anything outside the
sample.

## Annualization and its assumption

Multiplying a daily standard deviation by the square root of two hundred and
fifty two assumes returns are independent from one day to the next. They are
not, quite. Trending series have positive autocorrelation and their true
annual volatility is higher than the scaled figure, mean reverting series run
the other way. The convention is fine for comparing strategies measured the
same way and it is not a measurement of anything on its own.

## Example

```javascript
const { toReturns, riskReport, maxDrawdown } = require('./riskMetrics');

const prices = [100, 103, 99, 105, 101, 110, 104, 112];
const returns = toReturns(prices);

console.log(maxDrawdown(returns));
console.log(riskReport(returns, 0.02));
```

## Where to go next

* For the Python versions of these ratios see
  [`Sharpe and Sortino Ratio`](Sharpe and Sortino Ratio.md).
* For a deeper treatment of underwater periods see
  [`Risk Metrics - Drawdown Analysis`](Risk Metrics - Drawdown Analysis.md).
* For the parametric and simulated alternatives to historical VaR see
  [`Value at Risk (VaR)`](Value at Risk (VaR).md).
* For the tail measure in full see
  [`Finance - Expected Shortfall`](Finance - Expected Shortfall.md).


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

-   :material-shield-alert-outline: __[Finance - Treynor Ratio](Finance - Treynor Ratio.md)__

    The Sharpe ratio divides excess return by total volatility, which mixes two

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
