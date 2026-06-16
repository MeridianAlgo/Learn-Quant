"""
Numerical Methods
-----------------
Most of the formulas in finance cannot be solved with algebra. There is no
closed form for a bond's yield-to-maturity or an option's implied volatility —
you have a function and a target, and you must *search* for the input that hits
it. Likewise, when a derivative or an integral has no neat antiderivative, you
approximate it numerically.

This module implements the small toolkit that quietly powers the rest of the
repository:

* **Root finding** — bisection (slow but bulletproof), Newton-Raphson (fast when
  you have a derivative) and the secant method (fast without one).
* **Differentiation** — central finite differences, the workhorse behind option
  Greeks computed "by bumping".
* **Integration** — the trapezoid and Simpson rules for areas under a curve,
  e.g. integrating a probability density.

Every routine is written from first principles so you can see exactly how the
black box behaves — and where it can fail to converge.
"""

from __future__ import annotations

from typing import Callable


def bisection(f: Callable[[float], float], a: float, b: float, tol: float = 1e-10, max_iter: int = 200) -> float:
    """Find a root of *f* in ``[a, b]`` by repeated bisection.

    Bisection only needs *f* to change sign across the bracket; it cannot
    diverge and halves the interval every step. The price of that safety is
    linear convergence — about 3.3 correct digits per ten iterations.

    Args:
        f: Continuous function to find a root of.
        a: Left end of a bracket where ``f(a)`` and ``f(b)`` differ in sign.
        b: Right end of the bracket.
        tol: Stop when the bracket is narrower than this.
        max_iter: Safety cap on iterations.

    Raises:
        ValueError: If ``f(a)`` and ``f(b)`` do not bracket a root.
    """
    fa, fb = f(a), f(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs to bracket a root")
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        fm = f(mid)
        if fm == 0.0 or 0.5 * (b - a) < tol:
            return mid
        if fa * fm < 0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    return 0.5 * (a + b)


def newton_raphson(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> float:
    """Find a root of *f* from a starting guess using its derivative *df*.

    Newton's method converges quadratically near a simple root — the number of
    correct digits roughly doubles each step — but it can shoot off to infinity
    if the derivative is near zero or the guess is poor.

    Raises:
        ValueError: If the derivative vanishes or the method fails to converge.
    """
    x = float(x0)
    for _ in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return x
        dfx = df(x)
        if dfx == 0.0:
            raise ValueError("Zero derivative encountered; Newton-Raphson cannot proceed")
        step = fx / dfx
        x -= step
        if abs(step) < tol:
            return x
    raise ValueError("Newton-Raphson did not converge within max_iter")


def secant(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> float:
    """Find a root of *f* without a derivative, using two starting points.

    The secant method replaces Newton's analytic derivative with a finite
    difference from the last two iterates. It converges almost as fast
    (superlinearly) and is the practical default when *df* is unavailable.
    """
    a, b = float(x0), float(x1)
    fa, fb = f(a), f(b)
    for _ in range(max_iter):
        if abs(fb) < tol:
            return b
        denom = fb - fa
        if denom == 0.0:
            raise ValueError("Flat secant (f(x0) == f(x1)); cannot continue")
        c = b - fb * (b - a) / denom
        a, fa = b, fb
        b, fb = c, f(c)
        if abs(b - a) < tol:
            return b
    raise ValueError("Secant method did not converge within max_iter")


def finite_difference(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """Approximate ``f'(x)`` with a central difference.

    The central scheme ``(f(x+h) - f(x-h)) / 2h`` has error of order ``h^2``,
    far better than a one-sided difference. This is exactly how option Greeks
    are computed "by bumping" the input and re-pricing.
    """
    return (f(x + h) - f(x - h)) / (2.0 * h)


def trapezoid(f: Callable[[float], float], a: float, b: float, n: int = 1000) -> float:
    """Integrate *f* over ``[a, b]`` with the composite trapezoid rule.

    Approximates the area under the curve with *n* straight-line segments.
    Simple and robust; error shrinks like ``1/n^2``.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    h = (b - a) / n
    total = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        total += f(a + i * h)
    return total * h


def simpson(f: Callable[[float], float], a: float, b: float, n: int = 1000) -> float:
    """Integrate *f* over ``[a, b]`` with composite Simpson's rule.

    Fits parabolas instead of lines, so error shrinks like ``1/n^4`` — far more
    accurate than the trapezoid rule for smooth integrands. *n* must be even.
    """
    if n < 2 or n % 2 != 0:
        raise ValueError("n must be a positive even integer")
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        total += (4 if i % 2 else 2) * f(a + i * h)
    return total * h / 3.0


if __name__ == "__main__":
    import math

    print("Numerical Methods")
    print("=" * 40)

    # Root finding: solve x^2 = 2 (the positive root is sqrt(2)).
    f = lambda x: x * x - 2.0
    df = lambda x: 2.0 * x
    print(f"Solving x^2 - 2 = 0  (true root = {math.sqrt(2):.10f})")
    print(f"  bisection      -> {bisection(f, 0, 2):.10f}")
    print(f"  newton-raphson -> {newton_raphson(f, df, 1.0):.10f}")
    print(f"  secant         -> {secant(f, 1.0, 2.0):.10f}")

    # Differentiation: d/dx sin(x) at x=1 should be cos(1).
    approx = finite_difference(math.sin, 1.0)
    print(f"\nd/dx sin(x) at x=1: {approx:.8f}  (true cos(1) = {math.cos(1.0):.8f})")

    # Integration: area under the standard normal density over [-3, 3].
    pdf = lambda x: math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    print("\nProbability mass of N(0,1) on [-3, 3] (true ~ 0.9973):")
    print(f"  trapezoid -> {trapezoid(pdf, -3, 3):.6f}")
    print(f"  simpson   -> {simpson(pdf, -3, 3):.6f}")
