"""Tests for UTILS - Portfolio Optimizer/portfolio_tutorial.py."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "UTILS - Portfolio Optimizer"))

from portfolio_tutorial import _dot, _portfolio_stats, _sharpe

# ---------------------------------------------------------------------------
# _dot
# ---------------------------------------------------------------------------


class TestDot:
    def test_basic(self):
        assert _dot([1, 2, 3], [4, 5, 6]) == 32.0

    def test_zero_vector(self):
        assert _dot([0, 0, 0], [1, 2, 3]) == 0.0

    def test_unit_vector(self):
        assert _dot([1, 0, 0], [5, 6, 7]) == 5.0

    def test_negative(self):
        assert _dot([1, -1], [3, 3]) == 0.0


# ---------------------------------------------------------------------------
# _portfolio_stats
# ---------------------------------------------------------------------------


class TestPortfolioStats:
    """Test portfolio return and volatility calculations."""

    def _make_cov(self, vols, corr_matrix):
        n = len(vols)
        return [[corr_matrix[i][j] * vols[i] * vols[j] for j in range(n)] for i in range(n)]

    def test_single_asset(self):
        weights = [1.0]
        means = [0.10]
        cov = [[0.04]]  # sigma=0.20
        ret, vol = _portfolio_stats(weights, means, cov)
        assert abs(ret - 0.10) < 1e-10
        assert abs(vol - 0.20) < 1e-10

    def test_two_equal_weights_uncorrelated(self):
        weights = [0.5, 0.5]
        means = [0.10, 0.10]
        vols = [0.20, 0.20]
        cov = self._make_cov(vols, [[1.0, 0.0], [0.0, 1.0]])
        ret, vol = _portfolio_stats(weights, means, cov)
        assert abs(ret - 0.10) < 1e-10
        # vol = sqrt(0.5^2 * 0.04 + 0.5^2 * 0.04) = sqrt(0.02) ≈ 0.1414
        expected_vol = math.sqrt(0.5**2 * 0.04 + 0.5**2 * 0.04)
        assert abs(vol - expected_vol) < 1e-10

    def test_perfect_positive_correlation_no_diversification(self):
        weights = [0.5, 0.5]
        means = [0.10, 0.10]
        vols = [0.20, 0.20]
        cov = self._make_cov(vols, [[1.0, 1.0], [1.0, 1.0]])
        _, vol = _portfolio_stats(weights, means, cov)
        # With corr=1, portfolio vol = weighted avg vol = 0.20
        assert abs(vol - 0.20) < 1e-10

    def test_diversification_reduces_vol(self):
        weights = [0.5, 0.5]
        means = [0.10, 0.10]
        vols = [0.20, 0.20]
        cov_corr1 = self._make_cov(vols, [[1.0, 1.0], [1.0, 1.0]])
        cov_low = self._make_cov(vols, [[1.0, 0.0], [0.0, 1.0]])
        _, vol_corr1 = _portfolio_stats(weights, means, cov_corr1)
        _, vol_low = _portfolio_stats(weights, means, cov_low)
        assert vol_low < vol_corr1

    def test_return_is_weighted_mean(self):
        weights = [0.3, 0.4, 0.3]
        means = [0.05, 0.10, 0.15]
        cov = [[0.01 if i == j else 0.0 for j in range(3)] for i in range(3)]
        ret, _ = _portfolio_stats(weights, means, cov)
        expected_ret = 0.3 * 0.05 + 0.4 * 0.10 + 0.3 * 0.15
        assert abs(ret - expected_ret) < 1e-10

    def test_vol_non_negative(self):
        import random

        random.seed(0)
        for _ in range(20):
            n = 3
            weights = [1 / n] * n
            means = [random.uniform(0.05, 0.15) for _ in range(n)]
            sigma = [random.uniform(0.05, 0.30) for _ in range(n)]
            cov = [[sigma[i] * sigma[j] * (1.0 if i == j else 0.2) for j in range(n)] for i in range(n)]
            _, vol = _portfolio_stats(weights, means, cov)
            assert vol >= 0.0


# ---------------------------------------------------------------------------
# _sharpe
# ---------------------------------------------------------------------------


class TestSharpePortfolio:
    def test_positive_excess_return(self):
        sr = _sharpe(0.10, 0.15)
        assert sr > 0

    def test_zero_std_returns_zero(self):
        assert _sharpe(0.10, 0.0) == 0.0

    def test_rf_reduces_sharpe(self):
        sr_no_rf = _sharpe(0.10, 0.15, rf=0.0)
        sr_with_rf = _sharpe(0.10, 0.15, rf=0.04)
        assert sr_with_rf < sr_no_rf

    def test_higher_return_better_sharpe(self):
        sr_low = _sharpe(0.06, 0.15, rf=0.04)
        sr_high = _sharpe(0.10, 0.15, rf=0.04)
        assert sr_high > sr_low

    def test_higher_vol_worse_sharpe(self):
        sr_low_vol = _sharpe(0.10, 0.10, rf=0.04)
        sr_high_vol = _sharpe(0.10, 0.25, rf=0.04)
        assert sr_low_vol > sr_high_vol

    def test_negative_excess_return(self):
        sr = _sharpe(0.02, 0.15, rf=0.05)
        assert sr < 0
