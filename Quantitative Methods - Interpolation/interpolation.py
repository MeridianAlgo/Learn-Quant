"""
Interpolation
-------------
Markets hand you values at scattered points and questions at the points in
between. A yield curve quotes a handful of maturities but you need the rate
at seven and a half years. A volatility surface quotes round strikes but your
option sits between them. Interpolation is the craft of filling those gaps
with a curve that passes exactly through what you know.

Linear interpolation connects the dots with straight lines. It is simple,
robust, and never invents a wiggle that is not in the data, but its kinks
mean the slope jumps at every knot. A natural cubic spline instead threads a
smooth curve through the points, with continuous slope and curvature, at the
cost of a little more machinery and the possibility of gentle overshoot
between widely spaced points.
"""

from __future__ import annotations

import numpy as np


def linear_interpolate(xs: list[float], ys: list[float], x: float) -> float:
    """Return the straight line interpolation of the data at x.

    The x values must be strictly increasing. Points outside the range are
    clamped to the nearest endpoint rather than extrapolated, which is the
    conservative choice for financial curves.
    """
    xa = np.asarray(xs, dtype=float)
    ya = np.asarray(ys, dtype=float)
    if xa.size != ya.size:
        raise ValueError("xs and ys must be the same length")
    if xa.size < 2:
        raise ValueError("need at least two points")
    if np.any(np.diff(xa) <= 0):
        raise ValueError("xs must be strictly increasing")
    return float(np.interp(x, xa, ya))


def natural_cubic_spline(xs: list[float], ys: list[float]) -> list[tuple[float, float, float, float]]:
    """Return per interval cubic coefficients (a, b, c, d) for a natural spline.

    On interval i the curve is a + b*t + c*t**2 + d*t**3 with t = x - xs[i].
    Natural means the curvature is zero at both ends, the classic choice when
    you have no outside information about the boundary slopes. The
    coefficients come from solving the standard tridiagonal system for the
    second derivatives at the knots.
    """
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if x.size != y.size:
        raise ValueError("xs and ys must be the same length")
    if x.size < 3:
        raise ValueError("need at least three points for a cubic spline")
    if np.any(np.diff(x) <= 0):
        raise ValueError("xs must be strictly increasing")

    n = x.size
    h = np.diff(x)

    # Solve for second derivatives m at the interior knots, natural ends are zero.
    A = np.zeros((n, n))
    rhs = np.zeros(n)
    A[0, 0] = 1.0
    A[n - 1, n - 1] = 1.0
    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        rhs[i] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    m = np.linalg.solve(A, rhs)

    coeffs = []
    for i in range(n - 1):
        a = y[i]
        b = (y[i + 1] - y[i]) / h[i] - h[i] * (2 * m[i] + m[i + 1]) / 6
        c = m[i] / 2
        d = (m[i + 1] - m[i]) / (6 * h[i])
        coeffs.append((float(a), float(b), float(c), float(d)))
    return coeffs


def spline_interpolate(xs: list[float], ys: list[float], x: float) -> float:
    """Evaluate the natural cubic spline through the data at x.

    Points outside the knot range are clamped to the nearest endpoint value,
    matching the behaviour of linear_interpolate.
    """
    xa = np.asarray(xs, dtype=float)
    if x <= xa[0]:
        return float(ys[0])
    if x >= xa[-1]:
        return float(ys[-1])
    coeffs = natural_cubic_spline(xs, ys)
    i = int(np.searchsorted(xa, x) - 1)
    a, b, c, d = coeffs[i]
    t = x - xa[i]
    return a + b * t + c * t**2 + d * t**3


if __name__ == "__main__":
    print("Interpolation")
    print("=" * 40)

    maturities = [1.0, 2.0, 5.0, 10.0, 30.0]
    yields = [0.045, 0.043, 0.041, 0.042, 0.045]

    print("\nYield curve quoted at five maturities")
    for t in [3.0, 7.5, 20.0]:
        lin = linear_interpolate(maturities, yields, t)
        spl = spline_interpolate(maturities, yields, t)
        print(f"  {t:5.1f}y  linear {lin:.4%}  spline {spl:.4%}")

    print("\nBoth pass exactly through the quoted points")
    for t, y in zip(maturities, yields):
        assert abs(spline_interpolate(maturities, yields, t) - y) < 1e-12
        print(f"  {t:5.1f}y  quoted {y:.4%}  spline {spline_interpolate(maturities, yields, t):.4%}")
