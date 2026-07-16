import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Quantitative Methods - Interpolation"))
from interpolation import linear_interpolate, natural_cubic_spline, spline_interpolate

MATURITIES = [1.0, 2.0, 5.0, 10.0, 30.0]
YIELDS = [0.045, 0.043, 0.041, 0.042, 0.045]


def test_linear_hits_knots_exactly():
    for x, y in zip(MATURITIES, YIELDS):
        assert abs(linear_interpolate(MATURITIES, YIELDS, x) - y) < 1e-15


def test_linear_midpoint():
    assert abs(linear_interpolate([0.0, 2.0], [1.0, 3.0], 1.0) - 2.0) < 1e-15


def test_linear_clamps_outside_range():
    assert linear_interpolate(MATURITIES, YIELDS, 0.5) == YIELDS[0]
    assert linear_interpolate(MATURITIES, YIELDS, 50.0) == YIELDS[-1]


def test_spline_hits_knots_exactly():
    for x, y in zip(MATURITIES, YIELDS):
        assert abs(spline_interpolate(MATURITIES, YIELDS, x) - y) < 1e-12


def test_spline_clamps_outside_range():
    assert spline_interpolate(MATURITIES, YIELDS, 0.5) == YIELDS[0]
    assert spline_interpolate(MATURITIES, YIELDS, 50.0) == YIELDS[-1]


def test_spline_reproduces_straight_line():
    xs = [0.0, 1.0, 2.0, 3.0]
    ys = [1.0, 2.0, 3.0, 4.0]
    for x in [0.5, 1.25, 2.75]:
        assert abs(spline_interpolate(xs, ys, x) - (x + 1.0)) < 1e-10


def test_spline_is_continuous_across_knots():
    eps = 1e-8
    for knot in MATURITIES[1:-1]:
        left = spline_interpolate(MATURITIES, YIELDS, knot - eps)
        right = spline_interpolate(MATURITIES, YIELDS, knot + eps)
        assert abs(left - right) < 1e-5


def test_spline_smoother_than_linear_on_parabola():
    xs = np.linspace(0, 4, 5).tolist()
    ys = [x**2 for x in xs]
    x = 1.5
    spline_err = abs(spline_interpolate(xs, ys, x) - x**2)
    linear_err = abs(linear_interpolate(xs, ys, x) - x**2)
    assert spline_err < linear_err


def test_coefficient_count():
    coeffs = natural_cubic_spline(MATURITIES, YIELDS)
    assert len(coeffs) == len(MATURITIES) - 1
    assert all(len(c) == 4 for c in coeffs)


def test_non_increasing_xs_raise():
    with pytest.raises(ValueError):
        linear_interpolate([1.0, 1.0, 2.0], [0.0, 1.0, 2.0], 1.5)
    with pytest.raises(ValueError):
        natural_cubic_spline([2.0, 1.0, 3.0], [0.0, 1.0, 2.0])


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        linear_interpolate([1.0, 2.0], [1.0], 1.5)


def test_spline_needs_three_points():
    with pytest.raises(ValueError):
        natural_cubic_spline([1.0, 2.0], [1.0, 2.0])
