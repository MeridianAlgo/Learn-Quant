# Quantitative Methods, Fourier Analysis

A price series is a sum of waves if you are willing to look at it that way.
The Fourier transform takes a series indexed by time and hands back the
strength of every frequency hiding inside it. Nothing is lost in the trip, the
same data is simply written in a different alphabet, and questions that are
awkward in the time domain often become easy in the frequency one.

Three of those questions come up constantly in market work. Is there a
repeating cycle in this series and how long is it. Can I strip the fast
wiggles out without the lag that a moving average forces on me. How long does
this series remember its own past.

## The pieces

* `detrend(series)` removes the least squares straight line, which has to
  happen before anything else in this file will tell you the truth.
* `power_spectrum(series, sample_spacing)` returns every frequency and how
  much of the series it accounts for.
* `dominant_period(series, sample_spacing, min_period)` returns the length of
  the strongest cycle, in whatever units the sample spacing was given in.
* `lowpass(series, cutoff_period)` deletes every cycle faster than the cutoff
  and rebuilds the series without them.
* `reconstruct(series, n_components)` rebuilds the series from only its
  loudest few frequencies, which is lossy compression for a price path.
* `autocorrelation(series, max_lag)` returns correlation at every lag at once,
  computed through the transform rather than lag by lag.

## Why detrending is the first step and not an optional one

A series that drifts upward looks, to the transform, like the opening half of
an extremely slow wave. The transform has no concept of trend, so it does the
only thing it can and spends the low end of the spectrum describing that drift
with enormous low frequency components. Those components then dominate
everything, and the genuine twenty day cycle sitting in the data shows up as a
rounding error next to them.

Run the demo in this folder to watch it happen. The raw series reports its
strongest cycle as being the entire length of the sample, which is the
transform's way of saying it found a trend. Remove the line first and the real
twenty day cycle appears immediately.

## The filter with no lag, and its catch

A moving average smooths by looking backward, so its output always arrives
late. A low pass filter built on the transform has no such lag, because it
smooths by deleting frequencies rather than by averaging neighbours. The
smoothed value at any point lines up exactly with the raw value at that point.

The catch is severe enough to state plainly. Computing the transform requires
the whole series, including every point that comes after the one being
smoothed. A filtered series is therefore contaminated by the future at every
single point, and a backtest that trades on one will look spectacular and mean
absolutely nothing. Use it to look at history and to reason about what kind of
cycles exist. Never use it to generate a live signal.

## The Nyquist limit

Nothing shorter than two samples per cycle can be recovered, ever. A cycle
faster than that does not simply get measured badly, it comes back disguised
as a slower cycle that is not there at all, which is called aliasing and is
the reason `dominant_period` refuses to report anything below its
`min_period`. If you suspect an intraday cycle, intraday data is the only
answer. No amount of cleverness will pull it out of a daily series.

## Example

```python
from fourier import autocorrelation, detrend, dominant_period, lowpass

prices = [100 + 5 * (i % 20) / 20 + 0.02 * i for i in range(200)]
clean = detrend(prices)

print(dominant_period(clean))
print(lowpass(clean, cutoff_period=10)[:5])
print(autocorrelation(clean, max_lag=5))
```

## A word on what this does not prove

Finding a cycle in a price series is easy. Finding one that survives into the
next sample is the hard part, and most do not. Noise alone will hand you a
strongest frequency every time you ask for one, because something always has
to be the largest. Before trading a cycle, check that it holds on data the
search never touched, and check that its amplitude is large enough to clear
your costs.

## Where to go next

* For the time domain view of memory in a series see
  [`Quantitative Methods - Time Series`](../Quantitative%20Methods%20-%20Time%20Series/).
* For the lagging smoothers this one is contrasted against see
  [`Technical Indicators`](../Technical%20Indicators/).
* For the state space approach to filtering see
  [`Quantitative Methods - Kalman Filter`](../Quantitative%20Methods%20-%20Kalman%20Filter/).
* For detecting cycle changes rather than cycle lengths see
  [`Quantitative Methods - Regime Detection`](../Quantitative%20Methods%20-%20Regime%20Detection/).
