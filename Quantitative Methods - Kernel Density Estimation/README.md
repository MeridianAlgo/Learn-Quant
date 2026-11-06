# Quantitative Methods, Kernel Density Estimation

A histogram of returns looks different every time you move the bin edges.
A kernel density estimate gets rid of the bins. It places a small smooth bump
on top of every observation and adds them up, giving a curve you can
evaluate at any point, integrate for probabilities, and sample from.

For return data this sits in a useful middle ground. It does not assume the
normal shape that underestimates the tails, and it is not as ragged as the raw
empirical distribution, which makes quantiles like VaR less jumpy.

## The pieces

* `gaussian_kernel(u)` and `epanechnikov_kernel(u)` are the two bump shapes.
* `silverman_bandwidth(data)` and `scott_bandwidth(data)` are the standard
  rules for choosing the bump width.
* `kde(data, points, bandwidth, kernel)` evaluates the density.
* `kde_cdf(data, points, bandwidth)` gives cumulative probability under the
  Gaussian estimate.
* `kde_quantile(data, prob)` inverts that CDF by bisection.
* `kde_var(returns, confidence)` is VaR read off the smoothed distribution.
* `sample_kde(data, size)` draws new values, which is the smoothed bootstrap.

## The formula

```
f(x) = (1 / (n * h)) * sum over i of K((x - x_i) / h)
```

`K` is the kernel, any density that is symmetric around zero. `h` is the
bandwidth. Dividing by `n * h` keeps the total area at one.

## Bandwidth matters more than the kernel

Swap a Gaussian kernel for an Epanechnikov one and the curve hardly changes.
Halve the bandwidth and it changes completely. So most of the thinking goes
into `h`.

Scott's rule, `1.06 * std * n^(-1/5)`, is the bandwidth that minimises error
when the data really is normal. Return data is not, and one crash day can
inflate the standard deviation enough to smear the whole curve flat.
Silverman's version replaces the standard deviation with the smaller of the
standard deviation and the interquartile range divided by 1.34, and shrinks
the constant to 0.9. That makes it much less sensitive to a handful of
outliers, which is why it is the default here.

The `n^(-1/5)` part tells you that bandwidth shrinks slowly with more data.
Multiplying the sample by 32 only halves `h`.

## VaR from a smoothed distribution

Historical VaR at 99% on 500 days of data is decided by the fifth worst day.
Add or drop one bad day and it jumps. The KDE quantile blends in the
neighbouring observations, so it moves more smoothly as the window rolls. It
is still a historical method underneath. If the sample has no crash in it,
neither will the estimate.

The CDF of a Gaussian KDE has a closed form, the average of normal CDFs
centred on each point, so the quantile only needs a one dimensional root
search.

## Example

```python
import numpy as np
from kde import kde, kde_var, silverman_bandwidth

rng = np.random.default_rng(0)
returns = 0.01 * rng.standard_t(4, size=1000)

print(silverman_bandwidth(returns))
print(kde(returns, [-0.03, 0.0, 0.03]))
print(kde_var(returns, 0.99))
```

## Things to watch

* KDE spills probability past the edge of the data. Fine for returns, wrong
  for anything bounded such as volatility or a price that cannot go below
  zero. Fit on the log instead.
* A single global bandwidth is a compromise. It is often too wide in the
  peak and too narrow in the tails at the same time.
* Evaluating at `m` points costs `n * m` kernel calls. For very large
  samples use a binned or FFT based estimator.

## Where to go next

* For tail modelling that does not rely on the bulk of the data see
  [`Quantitative Methods - Extreme Value Theory`](../Quantitative%20Methods%20-%20Extreme%20Value%20Theory/).
* For the plain, unsmoothed resampling version see
  [`Quantitative Methods - Bootstrap`](../Quantitative%20Methods%20-%20Bootstrap/).
* For the VaR methods this is compared against see
  [`Value at Risk (VaR)`](../Value%20at%20Risk%20(VaR)/).
* For the distribution summaries this builds on see
  [`Quantitative Methods - Statistics`](../Quantitative%20Methods%20-%20Statistics/).
