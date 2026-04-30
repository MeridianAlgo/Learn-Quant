import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "UTILS - Finance - Kelly Criterion"))
from kelly_criterion import (
    fractional_kelly,
    kelly_continuous,
    kelly_fraction,
    kelly_growth_rate,
    multi_asset_kelly,
)


def test_kelly_fraction_basic():
    # f = p - q/b = 0.6 - 0.4/2.0 = 0.4
    f = kelly_fraction(0.6, 2.0)
    assert abs(f - 0.40) < 1e-10


def test_kelly_fraction_edge_zero():
    # p=0.5, b=1 → f = 0.5 - 0.5/1 = 0
    assert kelly_fraction(0.5, 1.0) == 0.0


def test_kelly_fraction_negative_edge():
    # Bad odds → negative Kelly (don't bet)
    f = kelly_fraction(0.3, 1.0)
    assert f < 0


def test_kelly_fraction_invalid_prob():
    with pytest.raises(ValueError):
        kelly_fraction(0.0, 2.0)
    with pytest.raises(ValueError):
        kelly_fraction(1.0, 2.0)


def test_kelly_fraction_invalid_ratio():
    with pytest.raises(ValueError):
        kelly_fraction(0.6, 0.0)


def test_kelly_continuous_basic():
    f = kelly_continuous(0.001, 0.02)
    assert f > 0
    assert abs(f - 0.001 / 0.02**2) < 1e-10


def test_kelly_continuous_invalid():
    with pytest.raises(ValueError):
        kelly_continuous(0.001, 0.0)


def test_fractional_kelly_half():
    full = kelly_fraction(0.6, 2.0)
    half = fractional_kelly(0.6, 2.0, 0.5)
    assert abs(half - full * 0.5) < 1e-10


def test_kelly_growth_rate_max_at_full():
    """Full Kelly should give max growth."""
    g_full = kelly_growth_rate(0.6, 2.0, 1.0)
    g_over = kelly_growth_rate(0.6, 2.0, 1.25)  # overbetting
    g_under = kelly_growth_rate(0.6, 2.0, 0.5)
    assert g_full > g_over
    assert g_full > g_under


def test_multi_asset_kelly_shape():
    mu = np.array([0.10, 0.15, 0.08])
    cov = np.array([[0.04, 0.01, 0.005],
                    [0.01, 0.09, 0.008],
                    [0.005, 0.008, 0.02]])
    weights = multi_asset_kelly(mu, cov)
    assert weights.shape == (3,)
    assert np.all(weights > 0)  # All positive expected returns should give long positions
