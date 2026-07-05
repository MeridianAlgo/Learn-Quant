import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(
    0,
    str(Path(__file__).parent.parent / "Quantitative Methods - Monte Carlo Integration"),
)
from monte_carlo_integration import (
    antithetic_integrate,
    black_scholes_call,
    mc_european_call,
    mc_integrate,
)


def test_integral_of_sine_is_near_two():
    estimate, stderr = mc_integrate(np.sin, 0, math.pi, n=100_000, seed=0)
    assert estimate == pytest.approx(2.0, abs=5 * stderr)


def test_integral_of_constant_is_exact():
    estimate, stderr = mc_integrate(lambda x: np.ones_like(x), 0, 3, n=1_000, seed=0)
    assert estimate == pytest.approx(3.0)
    assert stderr == pytest.approx(0.0)


def test_standard_error_shrinks_with_more_samples():
    _, err_small = mc_integrate(np.sin, 0, math.pi, n=1_000, seed=1)
    _, err_large = mc_integrate(np.sin, 0, math.pi, n=100_000, seed=1)
    assert err_large < err_small


def test_rejects_bad_interval():
    with pytest.raises(ValueError):
        mc_integrate(np.sin, 2, 1)


def test_rejects_too_few_samples():
    with pytest.raises(ValueError):
        mc_integrate(np.sin, 0, 1, n=1)


def test_antithetic_agrees_with_plain_estimate():
    estimate, stderr = antithetic_integrate(np.sin, 0, math.pi, n=100_000, seed=2)
    assert estimate == pytest.approx(2.0, abs=5 * stderr)


def test_antithetic_reduces_error_for_monotone_function():
    _, plain_err = mc_integrate(np.exp, 0, 1, n=20_000, seed=3)
    _, anti_err = antithetic_integrate(np.exp, 0, 1, n=20_000, seed=3)
    assert anti_err < plain_err


def test_black_scholes_known_value():
    # A widely tabulated case, spot 100 strike 100 r 5% vol 20% one year.
    assert black_scholes_call(100, 100, 0.05, 0.2, 1.0) == pytest.approx(10.4506, abs=1e-3)


def test_mc_price_converges_to_black_scholes():
    exact = black_scholes_call(100, 105, 0.05, 0.2, 1.0)
    price, stderr = mc_european_call(100, 105, 0.05, 0.2, 1.0, n=200_000, seed=4)
    assert price == pytest.approx(exact, abs=5 * stderr)


def test_deep_out_of_the_money_call_is_nearly_worthless():
    price, _ = mc_european_call(100, 400, 0.01, 0.1, 0.5, n=50_000, seed=5)
    assert price == pytest.approx(0.0, abs=1e-6)


def test_seed_makes_results_reproducible():
    a = mc_integrate(np.sin, 0, math.pi, n=5_000, seed=42)
    b = mc_integrate(np.sin, 0, math.pi, n=5_000, seed=42)
    assert a == b
