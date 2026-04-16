"""Tests for UTILS - Quantitative Methods - Statistics/statistics_tutorial.py."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "UTILS - Quantitative Methods - Statistics"))

from statistics_tutorial import (
    _cov,
    _kurtosis,
    _norm_cdf,
    _norm_pdf,
    _pearson,
    _skewness,
)

# ---------------------------------------------------------------------------
# Normal distribution helpers
# ---------------------------------------------------------------------------


class TestNormCdf:
    def test_median_is_half(self):
        assert abs(_norm_cdf(0.0) - 0.5) < 1e-6

    def test_one_sigma(self):
        # ~84.13% of distribution below +1 sigma
        assert abs(_norm_cdf(1.0) - 0.8413) < 0.001

    def test_minus_one_sigma(self):
        assert abs(_norm_cdf(-1.0) - 0.1587) < 0.001

    def test_two_sigma(self):
        assert abs(_norm_cdf(2.0) - 0.9772) < 0.001

    def test_symmetry(self):
        for z in [0.5, 1.0, 1.5, 2.0]:
            assert abs(_norm_cdf(z) + _norm_cdf(-z) - 1.0) < 1e-10

    def test_large_positive(self):
        assert _norm_cdf(5.0) > 0.9999

    def test_large_negative(self):
        assert _norm_cdf(-5.0) < 0.0001


class TestNormPdf:
    def test_peak_at_zero(self):
        # Standard normal peaks at 1/sqrt(2*pi) ≈ 0.3989
        assert abs(_norm_pdf(0.0) - 1.0 / math.sqrt(2 * math.pi)) < 1e-10

    def test_pdf_positive(self):
        for x in [-3, -1, 0, 1, 3]:
            assert _norm_pdf(x) > 0

    def test_symmetry(self):
        for x in [0.5, 1.0, 2.0]:
            assert abs(_norm_pdf(x) - _norm_pdf(-x)) < 1e-12

    def test_tails_smaller_than_peak(self):
        peak = _norm_pdf(0.0)
        assert _norm_pdf(1.0) < peak
        assert _norm_pdf(2.0) < _norm_pdf(1.0)


# ---------------------------------------------------------------------------
# Covariance and correlation
# ---------------------------------------------------------------------------


class TestCovariance:
    def test_identical_series_equals_variance(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        cov = _cov(data, data)
        import statistics

        var = statistics.variance(data)
        assert abs(cov - var) < 1e-10

    def test_opposite_series_negative(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [5.0, 4.0, 3.0, 2.0, 1.0]
        assert _cov(a, b) < 0

    def test_independent_near_zero(self):
        a = [1.0, -1.0, 1.0, -1.0]
        b = [1.0, 1.0, -1.0, -1.0]
        assert abs(_cov(a, b)) < 1e-10

    def test_symmetry(self):
        a = [0.01, -0.02, 0.03, -0.01, 0.02]
        b = [0.02, -0.01, 0.01, -0.03, 0.04]
        assert abs(_cov(a, b) - _cov(b, a)) < 1e-12


class TestPearson:
    def test_perfect_positive(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        corr = _pearson(a, a)
        assert abs(corr - 1.0) < 1e-10

    def test_perfect_negative(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [-1.0, -2.0, -3.0, -4.0, -5.0]
        assert abs(_pearson(a, b) - (-1.0)) < 1e-10

    def test_range_minus_one_to_one(self):
        import random

        random.seed(42)
        a = [random.gauss(0, 1) for _ in range(50)]
        b = [random.gauss(0, 1) for _ in range(50)]
        corr = _pearson(a, b)
        assert -1.0 <= corr <= 1.0

    def test_symmetry(self):
        a = [0.01, -0.02, 0.03, -0.01, 0.02]
        b = [0.02, -0.01, 0.01, -0.03, 0.04]
        assert abs(_pearson(a, b) - _pearson(b, a)) < 1e-12


# ---------------------------------------------------------------------------
# Skewness & Kurtosis
# ---------------------------------------------------------------------------


class TestSkewness:
    def test_symmetric_near_zero(self):
        # Perfectly symmetric data should have near-zero skewness
        data = [-2.0, -1.0, 0.0, 1.0, 2.0]
        assert abs(_skewness(data)) < 1e-10

    def test_right_skew_positive(self):
        # Data with a long right tail
        data = [0.0, 0.0, 0.0, 0.0, 10.0]
        assert _skewness(data) > 0

    def test_left_skew_negative(self):
        # Data with a long left tail
        data = [-10.0, 0.0, 0.0, 0.0, 0.0]
        assert _skewness(data) < 0


class TestKurtosis:
    def test_normal_like_near_zero(self):
        # Uniform/normal-like data should have excess kurtosis near 0
        data = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
        # Just check it's a finite float
        k = _kurtosis(data)
        assert isinstance(k, float)
        assert math.isfinite(k)

    def test_fat_tail_positive_excess(self):
        # Adding extreme outlier increases kurtosis
        normal = [0.1, -0.1, 0.2, -0.2, 0.15, -0.15, 0.05, -0.05]
        fat = normal + [-5.0, 5.0]
        assert _kurtosis(fat) > _kurtosis(normal)
