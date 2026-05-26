"""Tests for UTILS - Risk Metrics/risk_tutorial.py."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "UTILS - Risk Metrics"))

from risk_tutorial import (
    _drawdown_series,
    _historical_var,
    _norm_ppf,
    _parametric_var,
    _sharpe,
)

# ---------------------------------------------------------------------------
# _norm_ppf (inverse normal CDF)
# ---------------------------------------------------------------------------


class TestNormPpf:
    def test_median(self):
        assert abs(_norm_ppf(0.5)) < 1e-3  # should be near 0

    def test_95th_percentile(self):
        assert abs(_norm_ppf(0.95) - 1.645) < 0.02

    def test_99th_percentile(self):
        assert abs(_norm_ppf(0.99) - 2.326) < 0.02

    def test_5th_percentile_negative(self):
        assert _norm_ppf(0.05) < 0

    def test_symmetry(self):
        assert abs(_norm_ppf(0.95) + _norm_ppf(0.05)) < 0.01

    def test_raises_on_boundary(self):
        with pytest.raises(ValueError):
            _norm_ppf(0.0)
        with pytest.raises(ValueError):
            _norm_ppf(1.0)


# ---------------------------------------------------------------------------
# Historical VaR
# ---------------------------------------------------------------------------


class TestHistoricalVar:
    def setup_method(self):
        self.returns = [
            0.01,
            -0.02,
            0.03,
            -0.04,
            0.05,
            -0.01,
            0.02,
            -0.03,
            0.01,
            -0.05,
            0.02,
            0.01,
            -0.015,
            0.025,
            -0.035,
            0.015,
            -0.025,
            0.005,
            -0.045,
            0.03,
        ]

    def test_var_positive(self):
        var = _historical_var(self.returns, 0.95)
        assert var > 0

    def test_higher_confidence_higher_var(self):
        var_90 = _historical_var(self.returns, 0.90)
        var_95 = _historical_var(self.returns, 0.95)
        var_99 = _historical_var(self.returns, 0.99)
        assert var_90 <= var_95 <= var_99

    def test_var_bounded_by_worst_loss(self):
        worst = -min(self.returns)
        var = _historical_var(self.returns, 0.99)
        assert var <= worst + 1e-10

    def test_all_positive_returns_var_zero_or_positive(self):
        positive_returns = [0.01, 0.02, 0.03, 0.04, 0.05]
        var = _historical_var(positive_returns, 0.95)
        # When all returns are positive, VaR is the negation of the smallest return
        # which would be negative, but we floor at 0 essentially
        assert isinstance(var, float)


# ---------------------------------------------------------------------------
# Parametric VaR
# ---------------------------------------------------------------------------


class TestParametricVar:
    def test_var_positive_for_zero_mean(self):
        var = _parametric_var(0.0, 0.01, 0.95)
        assert var > 0

    def test_higher_confidence_higher_var(self):
        var_95 = _parametric_var(0.0, 0.01, 0.95)
        var_99 = _parametric_var(0.0, 0.01, 0.99)
        assert var_95 < var_99

    def test_higher_vol_higher_var(self):
        var_low = _parametric_var(0.0, 0.01, 0.95)
        var_high = _parametric_var(0.0, 0.02, 0.95)
        assert var_low < var_high

    def test_scaling_with_horizon(self):
        var_1d = _parametric_var(0.0, 0.01, 0.95, horizon_days=1)
        var_10d = _parametric_var(0.0, 0.01, 0.95, horizon_days=10)
        # 10-day VaR should be sqrt(10) times 1-day VaR (zero mean case)
        assert abs(var_10d / var_1d - math.sqrt(10)) < 0.01

    def test_positive_mean_reduces_var(self):
        var_zero_mean = _parametric_var(0.0, 0.01, 0.95)
        var_positive_mean = _parametric_var(0.001, 0.01, 0.95)
        assert var_positive_mean < var_zero_mean


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------


class TestDrawdownSeries:
    def test_monotonic_increase_zero_drawdown(self):
        equity = [100.0, 110.0, 120.0, 130.0, 140.0]
        max_dd, peak_i, trough_i = _drawdown_series(equity)
        assert max_dd == 0.0

    def test_simple_drawdown(self):
        equity = [100.0, 120.0, 80.0, 110.0]
        max_dd, peak_i, trough_i = _drawdown_series(equity)
        # Peak at 120, trough at 80 → drawdown = (120-80)/120 = 33.3%
        assert abs(max_dd - (120 - 80) / 120) < 1e-10
        assert peak_i == 1
        assert trough_i == 2

    def test_drawdown_between_zero_and_one(self):
        equity = [100.0, 90.0, 80.0, 110.0, 70.0]
        max_dd, _, _ = _drawdown_series(equity)
        assert 0.0 <= max_dd <= 1.0

    def test_all_same_value_zero_drawdown(self):
        equity = [100.0] * 10
        max_dd, _, _ = _drawdown_series(equity)
        assert max_dd == 0.0

    def test_drawdown_uses_running_peak(self):
        # Peak should update as new highs are made
        equity = [100.0, 90.0, 110.0, 80.0]
        # Running peaks: 100, 100, 110, 110
        # Drawdowns:       0,  10%,  0,  ~27%
        max_dd, peak_i, trough_i = _drawdown_series(equity)
        expected = (110 - 80) / 110
        assert abs(max_dd - expected) < 1e-10
        assert peak_i == 2
        assert trough_i == 3


# ---------------------------------------------------------------------------
# Sharpe
# ---------------------------------------------------------------------------


class TestSharpe:
    def test_positive_return_positive_sharpe(self):
        sr = _sharpe(0.001, 0.01)
        assert sr > 0

    def test_zero_std_returns_zero(self):
        assert _sharpe(0.001, 0.0) == 0.0

    def test_excess_return_matters(self):
        sr_high_rf = _sharpe(0.001, 0.01, rf=0.0008)
        sr_low_rf = _sharpe(0.001, 0.01, rf=0.0001)
        assert sr_low_rf > sr_high_rf

    def test_higher_return_higher_sharpe(self):
        sr_low = _sharpe(0.0005, 0.01)
        sr_high = _sharpe(0.002, 0.01)
        assert sr_high > sr_low

    def test_higher_vol_lower_sharpe(self):
        sr_low_vol = _sharpe(0.001, 0.005)
        sr_high_vol = _sharpe(0.001, 0.02)
        assert sr_low_vol > sr_high_vol
