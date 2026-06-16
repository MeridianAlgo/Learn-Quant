import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Quantitative Methods - Numerical Methods"))
from numerical_methods import (
    bisection,
    finite_difference,
    newton_raphson,
    secant,
    simpson,
    trapezoid,
)


def test_bisection_finds_sqrt2():
    root = bisection(lambda x: x * x - 2, 0, 2)
    assert abs(root - math.sqrt(2)) < 1e-8


def test_bisection_requires_sign_change():
    try:
        bisection(lambda x: x * x + 1, 0, 2)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_bisection_endpoint_root():
    assert bisection(lambda x: x - 1, 1, 5) == 1


def test_newton_finds_sqrt2():
    root = newton_raphson(lambda x: x * x - 2, lambda x: 2 * x, x0=1.0)
    assert abs(root - math.sqrt(2)) < 1e-10


def test_newton_zero_derivative_raises():
    # f(x) = x^2 + 1 has no real root; at x0=0 the derivative is zero.
    try:
        newton_raphson(lambda x: x * x + 1, lambda x: 2 * x, x0=0.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_secant_matches_newton():
    root = secant(lambda x: x * x - 2, 1.0, 2.0)
    assert abs(root - math.sqrt(2)) < 1e-10


def test_finite_difference_of_sin():
    assert abs(finite_difference(math.sin, 1.0) - math.cos(1.0)) < 1e-6


def test_finite_difference_polynomial():
    # d/dx x^3 at x=2 is 12.
    assert abs(finite_difference(lambda x: x**3, 2.0) - 12.0) < 1e-4


def test_trapezoid_integrates_quadratic():
    # integral of x^2 from 0 to 1 = 1/3.
    assert abs(trapezoid(lambda x: x * x, 0, 1, n=1000) - 1 / 3) < 1e-5


def test_simpson_integrates_normal_pdf():
    pdf = lambda x: math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    assert abs(simpson(pdf, -3, 3, n=1000) - 0.9973) < 1e-3


def test_simpson_more_accurate_than_trapezoid():
    f = lambda x: math.sin(x)
    true = 2.0  # integral of sin over [0, pi]
    err_trap = abs(trapezoid(f, 0, math.pi, n=20) - true)
    err_simp = abs(simpson(f, 0, math.pi, n=20) - true)
    assert err_simp < err_trap


def test_simpson_requires_even_n():
    try:
        simpson(lambda x: x, 0, 1, n=3)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
