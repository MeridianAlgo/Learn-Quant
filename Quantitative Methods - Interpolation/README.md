# Quantitative Methods, Interpolation

Markets hand you values at scattered points and questions at the points in
between. A yield curve quotes a handful of maturities but you need the rate
at seven and a half years. A volatility surface quotes round strikes but your
option sits between them. Interpolation is the craft of filling those gaps
with a curve that passes exactly through what you know.

## Two ways to connect the dots

Linear interpolation joins each pair of neighbouring points with a straight
line. It is simple, robust, and never invents a wiggle that is not in the
data, but the slope jumps at every knot, and anything you compute from the
slope, such as a forward rate, jumps with it.

A natural cubic spline threads a single smooth curve through all the points,
with continuous slope and curvature everywhere. Natural means the curvature
is set to zero at both ends, the standard choice when nothing is known about
the boundary behaviour. The price of the smoothness is a little machinery, a
tridiagonal system solved for the curvature at each knot, and the
possibility of gentle overshoot between widely spaced points.

## The pieces

* `linear_interpolate(xs, ys, x)` evaluates the straight line interpolant at
  x. Points outside the range are clamped to the nearest endpoint, the
  conservative choice for financial curves.
* `natural_cubic_spline(xs, ys)` returns the per interval cubic
  coefficients, one tuple of four numbers per gap between knots.
* `spline_interpolate(xs, ys, x)` evaluates the spline at x, with the same
  clamping behaviour outside the knots.

## Choosing between them

Prefer linear when the data may contain jumps or when you must guarantee the
interpolant stays between neighbouring values, as with probabilities or
discount factors. Prefer the spline when downstream quantities depend on
smooth derivatives, as with forward rates read off a yield curve, where the
kinks of linear interpolation would show up as artificial steps. Both agree
exactly at the knots, so the choice only matters in between.

## Example

```python
from interpolation import linear_interpolate, spline_interpolate

maturities = [1.0, 2.0, 5.0, 10.0, 30.0]
yields = [0.045, 0.043, 0.041, 0.042, 0.045]

print(linear_interpolate(maturities, yields, 7.5))
print(spline_interpolate(maturities, yields, 7.5))
```

## A note on extrapolation

Neither function extrapolates. Asking for a value outside the quoted range
returns the nearest endpoint instead of extending the last segment, because
a straight line or cubic continued beyond the data can wander anywhere and a
flat answer is at least honest about knowing nothing out there.

## Where to go next

* For the curve these tools most often fill in see
  [`Finance - Yield Curve`](../Finance%20-%20Yield%20Curve/).
* For root finding and integration on top of interpolated curves see
  [`Quantitative Methods - Numerical Methods`](../Quantitative%20Methods%20-%20Numerical%20Methods/).
* For the linear algebra behind the spline system see
  [`Quantitative Methods - Linear Algebra`](../Quantitative%20Methods%20-%20Linear%20Algebra/).
