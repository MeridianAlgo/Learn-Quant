import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Finance - Omega Ratio"))
from omega_ratio import (
    annual_threshold_to_periodic,
    excess_gains_and_losses,
    omega_curve,
    omega_ratio,
)


def test_gains_and_losses_split():
    gains, losses = excess_gains_and_losses([0.02, -0.01], threshold=0.0)
    assert gains == pytest.approx(0.01)  # mean of [0.02, 0]
    assert losses == pytest.approx(0.005)  # mean of [0, 0.01]


def test_gains_and_losses_are_non_negative():
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.02, size=300).tolist()
    gains, losses = excess_gains_and_losses(returns)
    assert gains >= 0
    assert losses >= 0


def test_empty_series_raises():
    with pytest.raises(ValueError):
        omega_ratio([])


def test_omega_of_symmetric_series_is_one():
    assert omega_ratio([0.01, -0.01, 0.02, -0.02]) == pytest.approx(1.0)


def test_omega_above_one_when_gains_dominate():
    assert omega_ratio([0.02, 0.03, -0.01]) > 1.0


def test_omega_infinite_without_shortfall():
    assert math.isinf(omega_ratio([0.01, 0.02, 0.03]))


def test_omega_is_one_when_flat_at_threshold():
    assert omega_ratio([0.0, 0.0, 0.0], threshold=0.0) == pytest.approx(1.0)


def test_omega_falls_as_threshold_rises():
    rng = np.random.default_rng(5)
    returns = rng.normal(0.001, 0.01, size=500).tolist()
    curve = omega_curve(returns, [0.0, 0.001, 0.002])
    values = [om for _, om in curve]
    assert values[0] > values[1] > values[2]


def test_omega_matches_hand_computation():
    returns = [0.05, -0.02, 0.01, -0.04]
    gains = (0.05 + 0.01) / 4
    losses = (0.02 + 0.04) / 4
    assert omega_ratio(returns) == pytest.approx(gains / losses)


def test_annual_threshold_conversion_round_trips():
    per_day = annual_threshold_to_periodic(0.05, periods_per_year=252)
    assert (1 + per_day) ** 252 - 1 == pytest.approx(0.05)
