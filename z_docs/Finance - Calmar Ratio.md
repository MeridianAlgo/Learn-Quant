<p class="lq-badges"><span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-cat">Risk & Performance</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Finance - Calmar Ratio"
    python "calmar_ratio.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Finance%20-%20Calmar%20Ratio)

---
# Finance, Calmar Ratio

The Sharpe ratio judges a strategy by how much its returns wobble around their
average. But an investor does not lie awake over wobble, they lie awake over
drawdown, the slow sick feeling of watching an account fall from its high water
mark and stay there. The Calmar ratio speaks to that fear directly. It divides
the annual growth rate by the worst peak to trough loss, so a strategy that
compounds nicely yet once halved your capital scores poorly no matter how smooth
the rest of the ride looked.

This lesson builds the pieces from a return series with plain arithmetic on an
equity curve, which makes it a natural companion to the Sharpe and Sortino
lessons that measure risk a different way.

## The pieces

* `equity_curve(returns)` turns a series of period returns into a growing
  account value, the running product of one plus each return.
* `max_drawdown(returns)` tracks the running high water mark and reports the
  deepest fall below it as a positive fraction. A result of 0.25 means a quarter
  of the capital was lost from a peak at the worst point.
* `cagr(returns, periods_per_year)` is the compound annual growth rate, the one
  smooth yearly rate that takes you from the start of the curve to the end.
* `calmar_ratio(returns)` divides the growth by the maximum drawdown.
* `mar_ratio(returns)` is the same calculation, traditionally measured over the
  full track record rather than a recent window.

## Reading the number

A higher Calmar means more compounding earned for each unit of worst case pain.
As a rough guide a value above three is often called excellent and below one is
weak, but the figure is sensitive to the window you measure it over, since a
single brutal year can dominate the drawdown for a long time afterward. Always
note how long a track record the ratio was computed on before comparing two of
them.

## Why drawdown and not volatility

Volatility treats an upside surprise and a downside surprise as equally bad,
because it squares every deviation from the mean. Drawdown only counts the
losses, and only the ones that compound on top of each other from a peak. That
matches how capital actually behaves and how investors actually feel, which is
why drawdown based ratios like Calmar sit next to the Sharpe rather than
replacing it. Each tells you something the other hides.

## Example

```python
from calmar_ratio import calmar_ratio, max_drawdown, cagr

returns = [0.004, -0.01, 0.006, -0.03, 0.012, 0.002, -0.05, 0.02]

print(cagr(returns))            # the smooth annual growth rate
print(max_drawdown(returns))    # the worst peak to trough fall
print(calmar_ratio(returns))    # growth earned per unit of that fall
```

## A note on the edges

A curve that only ever rises has no drawdown, so the Calmar ratio is infinite,
which the code reports as positive infinity rather than crashing. A strategy that
is fully wiped out returns a growth rate of minus one. Both are honest answers to
degenerate inputs rather than silent errors, but neither describes anything you
would actually trade.

## Where to go next

* For the volatility based cousin see
  [`Sharpe and Sortino Ratio`](Sharpe and Sortino Ratio.md).
* For a deeper look at the drawdown itself see
  [`Risk Metrics - Drawdown Analysis`](Risk Metrics - Drawdown Analysis.md).
* For the broader set of track record measures see
  [`Quantitative Methods - Performance Analysis`](Quantitative Methods - Performance Analysis.md).


---

## Continue in Risk & Performance

<div class="grid cards" markdown>

-   :material-shield-alert-outline: __[Finance - Information Ratio](Finance - Information Ratio.md)__

    When a portfolio is judged **against a benchmark**, what matters is how much it beat the benchmark by — and how *reliably*. These are the core metrics of active management: active return, tracking error, Information Ratio, and the appraisal ratio.

-   :material-shield-alert-outline: __[Finance - Performance Attribution](Finance - Performance Attribution.md)__

    Brinson decomposition splits portfolio active return into **allocation** and **selection** effects — answering *"did we beat the benchmark by picking the right sectors or the right stocks?"*

-   :material-shield-alert-outline: __[Risk Metrics](Risk Metrics.md)__

    This module gives you quick, professional stats about risk in any list or array of investment returns. It's used by investors, analysts, and students everywhere!

-   :material-shield-alert-outline: __[Risk Metrics - Drawdown Analysis](Risk Metrics - Drawdown Analysis.md)__

    Comprehensive drawdown metrics for quantifying portfolio loss risk over time. Drawdown measures capture both the **depth** and **duration** of losses — dimensions VaR ignores.

-   :material-shield-alert-outline: __[Risk Metrics - Stress Testing](Risk Metrics - Stress Testing.md)__

    Stress tests answer: *"What happens if 2008 repeats?"* or *"How big a shock kills the portfolio?"* Required by Basel III, CCAR, and most institutional risk frameworks.

-   :material-shield-alert-outline: __[Sharpe and Sortino Ratio](Sharpe and Sortino Ratio.md)__

    This utility offers easy-to-use Python functions to calculate Sharpe and Sortino ratios for financial returns. These ratios help you understand whether a series of investment returns is attractive on a risk-adjusted basis.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
