# Finance, Omega Ratio

<p class="lq-meta">Intermediate <span>/</span> Risk & Performance <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Finance%20-%20Omega%20Ratio">source</a></p>


The Sharpe ratio squeezes a whole return distribution into a mean and a
standard deviation, which throws away everything interesting about its shape.
The Omega ratio keeps the shape. You pick a threshold return that matters to
you, then compare all the gains above it to all the shortfalls below it. The
answer uses every moment of the distribution, skew and fat tails included,
without ever having to estimate them.

An Omega of one means the gains above your bar exactly balance the losses
below it. Above one the strategy earned more than it gave up relative to that
bar. Because the threshold is yours to choose, one track record has a whole
Omega curve rather than a single score, and that curve is where the insight
lives.

## The pieces

* `excess_gains_and_losses(returns, threshold)` splits the series into the
  average gain above the bar and the average shortfall below it, both reported
  as positive numbers.
* `omega_ratio(returns, threshold)` divides the first by the second. This is
  the headline number.
* `omega_curve(returns, thresholds)` evaluates the ratio across a range of
  bars and returns the pairs, so you can see how quickly a strategy fades as
  you demand more of it.
* `annual_threshold_to_periodic(annual, periods_per_year)` converts a yearly
  bar like five percent into the per period rate the other functions expect.

## Reading the number

At a threshold of zero the Omega ratio simply asks whether the ups outweigh
the downs. As you raise the bar the ratio always falls, because more of the
distribution counts as shortfall. Two strategies can rank one way at a low bar
and the opposite way at a high one. A fund that grinds out small steady gains
beats a lottery ticket for a cautious investor, and the ordering can flip for
an aggressive one. The crossing point of two Omega curves is the bar at which
your preference should change.

## Why keep the whole distribution

Standard deviation punishes upside surprises exactly as hard as downside ones,
and it says nothing about asymmetry. Two funds with identical means and
volatilities can hide very different tail behaviour, one clipping small losses
while the other quietly accumulates crash risk. Omega sees the difference
because it works directly on the observed returns rather than on two summary
numbers. That makes it a natural companion to the Sharpe rather than a
replacement, each answers a question the other cannot.

## Example

```python
from omega_ratio import omega_ratio, omega_curve, annual_threshold_to_periodic

returns = [0.004, -0.01, 0.006, -0.03, 0.012, 0.002, -0.005, 0.02]

print(omega_ratio(returns))                # gains versus losses around zero
bar = annual_threshold_to_periodic(0.05)   # a five percent annual bar
print(omega_ratio(returns, bar))           # the same series judged harder
print(omega_curve(returns, [0.0, 0.001]))  # how the score fades as the bar rises
```

## A note on the edges

A series with no returns below the threshold has no shortfall to divide by, so
the ratio is infinite, reported as positive infinity rather than a crash. A
series sitting exactly on the threshold everywhere is reported as one, the
neutral score. Both are honest answers to degenerate inputs.

## Where to go next

* For the volatility based cousin see
  [`Sharpe and Sortino Ratio`](Sharpe and Sortino Ratio.md).
* For the drawdown based cousin see
  [`Finance - Calmar Ratio`](Finance - Calmar Ratio.md).
* For the broader toolkit of track record measures see
  [`Quantitative Methods - Performance Analysis`](Quantitative Methods - Performance Analysis.md).

---

## Running it

```bash
cd "Finance - Omega Ratio"
python "omega_ratio.py"
```

## More in Risk & Performance

- [Finance - Calmar Ratio](Finance - Calmar Ratio.md): The Sharpe ratio judges a strategy by how much its returns wobble around their average.
- [Finance - Information Ratio](Finance - Information Ratio.md): When a portfolio is judged against a benchmark, what matters is how much it beat the benchmark by — and how *reliably*.
- [Finance - Jensens Alpha](Finance - Jensens Alpha.md): Beta tells you how much of a portfolio's return came from simply riding the market.
- [Finance - Performance Attribution](Finance - Performance Attribution.md): Brinson decomposition splits portfolio active return into allocation and selection effects — answering *"did we beat the benchmark by picking the right sectors or the right stocks?"*
- [Finance - Treynor Ratio](Finance - Treynor Ratio.md): The Sharpe ratio divides excess return by total volatility, which mixes two very different kinds of risk.
- [Risk Metrics](Risk Metrics.md): This module gives you quick, professional stats about risk in any list or array of investment returns.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
