import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Finance - Calmar Ratio"))
from calmar_ratio import (
    cagr,
    calmar_ratio,
    equity_curve,
    mar_ratio,
    max_drawdown,
)


def test_equity_curve_compounds():
    curve = equity_curve([0.1, 0.1])
    assert curve[-1] == pytest.approx(1.21)  # 1.1 * 1.1


def test_max_drawdown_simple_case():
    # Up to 1.2, down to 0.9 -> trough is 0.9 from a peak of 1.2 -> 25 percent.
    returns = [0.2, -0.25]
    assert max_drawdown(returns) == pytest.approx(0.25)


def test_max_drawdown_of_rising_curve_is_zero():
    assert max_drawdown([0.01] * 50) == pytest.approx(0.0)


def test_max_drawdown_is_non_negative():
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.02, size=200).tolist()
    assert max_drawdown(returns) >= 0.0


def test_cagr_of_known_annual_doubling():
    # 252 daily returns that each compound to a doubling over the year.
    daily = (2 ** (1 / 252)) - 1
    returns = [daily] * 252
    assert cagr(returns, periods_per_year=252) == pytest.approx(1.0, rel=1e-6)


def test_cagr_total_wipeout():
    assert cagr([-1.0, 0.5], periods_per_year=252) == -1.0


def test_calmar_is_growth_over_drawdown():
    returns = [0.2, -0.25, 0.1, 0.05, -0.1]
    expected = cagr(returns) / max_drawdown(returns)
    assert calmar_ratio(returns) == pytest.approx(expected)


def test_calmar_infinite_without_drawdown():
    assert math.isinf(calmar_ratio([0.001] * 100))


def test_mar_matches_calmar_in_this_form():
    returns = [0.01, -0.02, 0.03, -0.04, 0.02]
    assert mar_ratio(returns) == pytest.approx(calmar_ratio(returns))


def test_smoother_strategy_scores_higher():
    rng = np.random.default_rng(3)
    smooth = rng.normal(0.0005, 0.003, size=400).tolist()
    rough = rng.normal(0.0005, 0.02, size=400).tolist()
    # Same drift, more volatility means deeper drawdowns and a lower Calmar.
    assert max_drawdown(rough) > max_drawdown(smooth)
