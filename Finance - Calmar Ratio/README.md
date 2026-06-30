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
  [`Sharpe and Sortino Ratio`](../Sharpe%20and%20Sortino%20Ratio/).
* For a deeper look at the drawdown itself see
  [`Risk Metrics - Drawdown Analysis`](../Risk%20Metrics%20-%20Drawdown%20Analysis/).
* For the broader set of track record measures see
  [`Quantitative Methods - Performance Analysis`](../Quantitative%20Methods%20-%20Performance%20Analysis/).
