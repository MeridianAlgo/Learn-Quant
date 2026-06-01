import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Finance - Information Ratio"))
from information_ratio import (
    active_returns,
    appraisal_ratio,
    information_ratio,
    tracking_error,
)


def test_active_returns_difference():
    port = np.array([0.02, 0.01, -0.01])
    bench = np.array([0.01, 0.01, 0.00])
    assert np.allclose(active_returns(port, bench), [0.01, 0.0, -0.01])


def test_active_returns_length_mismatch():
    with pytest.raises(ValueError):
        active_returns(np.array([0.01, 0.02]), np.array([0.01]))


def test_zero_tracking_error_gives_zero_ir():
    # Identical series → no active risk → IR defined as 0.
    series = np.array([0.01, 0.02, -0.01, 0.005])
    assert tracking_error(series, series) == 0.0
    assert information_ratio(series, series) == 0.0


def test_information_ratio_positive_when_outperforming():
    rng = np.random.default_rng(0)
    bench = rng.normal(0.0004, 0.011, 504)
    port = bench + 0.0003  # consistent constant outperformance
    assert information_ratio(port, bench) > 0


def test_appraisal_ratio_recovers_beta():
    rng = np.random.default_rng(1)
    bench = rng.normal(0.0004, 0.011, 1000)
    port = 0.0002 + 1.2 * bench + rng.normal(0, 0.003, 1000)
    result = appraisal_ratio(port, bench)
    assert abs(result["beta"] - 1.2) < 0.05
    assert result["appraisal_ratio"] > 0
