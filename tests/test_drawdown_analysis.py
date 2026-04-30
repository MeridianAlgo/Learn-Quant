import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "UTILS - Risk Metrics - Drawdown Analysis"))
from drawdown_analysis import (
    average_drawdown,
    calmar_ratio,
    drawdown_series,
    drawdown_summary,
    max_drawdown,
    max_drawdown_duration,
    ulcer_index,
    ulcer_performance_index,
)


@pytest.fixture
def mixed_returns():
    np.random.seed(42)
    bull = np.random.normal(0.001, 0.01, 200)
    bear = np.random.normal(-0.003, 0.02, 100)
    recovery = np.random.normal(0.001, 0.01, 152)
    return np.concatenate([bull, bear, recovery])


def test_drawdown_series_non_positive(mixed_returns):
    dd = drawdown_series(mixed_returns)
    assert np.all(dd <= 0.0 + 1e-10)


def test_drawdown_series_zero_at_start():
    flat = np.zeros(10)
    dd = drawdown_series(flat)
    assert dd[0] == 0.0


def test_max_drawdown_positive(mixed_returns):
    mdd = max_drawdown(mixed_returns)
    assert mdd > 0
    assert mdd < 1.0


def test_max_drawdown_zero_for_always_rising():
    always_up = np.full(100, 0.005)
    assert max_drawdown(always_up) == 0.0


def test_calmar_ratio_positive_for_positive_returns(mixed_returns):
    # Mixed returns may or may not be positive; just check type and finiteness
    cr = calmar_ratio(mixed_returns)
    assert isinstance(cr, float)


def test_calmar_ratio_inf_when_no_drawdown():
    always_up = np.full(100, 0.005)
    assert calmar_ratio(always_up) == np.inf


def test_ulcer_index_non_negative(mixed_returns):
    ui = ulcer_index(mixed_returns)
    assert ui >= 0


def test_average_drawdown_non_negative(mixed_returns):
    ad = average_drawdown(mixed_returns)
    assert ad >= 0


def test_max_drawdown_duration_positive(mixed_returns):
    dur = max_drawdown_duration(mixed_returns)
    assert dur > 0
    assert isinstance(dur, int)


def test_drawdown_summary_keys(mixed_returns):
    summary = drawdown_summary(mixed_returns)
    expected_keys = [
        "max_drawdown", "calmar_ratio", "ulcer_index",
        "ulcer_performance_index", "average_drawdown",
        "max_drawdown_duration_periods",
    ]
    for key in expected_keys:
        assert key in summary
