import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "UTILS - Quantitative Methods - Interest Rate Models"))
from interest_rate_models import (
    cir_bond_price,
    cir_simulate,
    cir_yield,
    term_structure,
    vasicek_bond_price,
    vasicek_simulate,
    vasicek_yield,
)

PARAMS = dict(r0=0.03, kappa=0.3, theta=0.05, sigma=0.01)


def test_vasicek_bond_price_between_zero_one():
    price = vasicek_bond_price(**PARAMS, T=1.0)
    assert 0 < price < 1


def test_vasicek_bond_price_decreases_with_maturity():
    prices = [vasicek_bond_price(**PARAMS, T=T) for T in [1, 5, 10, 30]]
    assert all(prices[i] > prices[i + 1] for i in range(len(prices) - 1))


def test_vasicek_yield_positive():
    y = vasicek_yield(**PARAMS, T=5.0)
    assert y > 0


def test_vasicek_simulate_shape():
    paths = vasicek_simulate(**PARAMS, T=1.0, n_steps=252, n_paths=5, seed=42)
    assert paths.shape == (5, 253)


def test_vasicek_simulate_starts_at_r0():
    paths = vasicek_simulate(**PARAMS, T=1.0, n_steps=100, n_paths=3, seed=42)
    assert np.all(paths[:, 0] == PARAMS["r0"])


def test_cir_bond_price_between_zero_one():
    price = cir_bond_price(**PARAMS, T=1.0)
    assert 0 < price < 1


def test_cir_non_negative_paths():
    paths = cir_simulate(**PARAMS, T=1.0, n_steps=252, n_paths=10, seed=42)
    assert np.all(paths >= 0)


def test_cir_simulate_shape():
    paths = cir_simulate(**PARAMS, T=1.0, n_steps=100, n_paths=4, seed=42)
    assert paths.shape == (4, 101)


def test_term_structure_vasicek_upward_sloping():
    """With r0 < theta, yield curve should slope upward."""
    maturities = [1, 5, 10, 30]
    ts = term_structure(**PARAMS, maturities=maturities, model="vasicek")
    yields = [ts[T] for T in maturities]
    assert yields[-1] > yields[0]


def test_term_structure_cir_keys():
    maturities = [1, 2, 5, 10]
    ts = term_structure(**PARAMS, maturities=maturities, model="cir")
    assert set(ts.keys()) == set(maturities)
