import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Quantitative Methods - Cointegration"))
from cointegration import (
    adf_test,
    engle_granger,
    half_life,
    ols_hedge_ratio,
    zscore_spread,
)


def test_ols_hedge_ratio_recovers_slope():
    np.random.seed(0)
    x = np.random.normal(0, 1, 200)
    y = 2.0 + 1.5 * x + np.random.normal(0, 0.1, 200)
    fit = ols_hedge_ratio(y, x)
    assert abs(fit["beta"] - 1.5) < 0.05
    assert abs(fit["alpha"] - 2.0) < 0.05
    assert fit["r_squared"] > 0.95


def test_ols_hedge_ratio_length_mismatch():
    with pytest.raises(ValueError):
        ols_hedge_ratio([1.0, 2.0, 3.0], [1.0, 2.0])


def test_adf_stationary_white_noise():
    np.random.seed(1)
    series = np.random.normal(0, 1, 500)
    result = adf_test(series, lags=1)
    assert result["t_stat"] < -2.86
    assert result["stationary_5pct"] is True


def test_adf_random_walk_nonstationary():
    np.random.seed(2)
    series = np.cumsum(np.random.normal(0, 1, 500))
    result = adf_test(series, lags=1)
    assert result["t_stat"] > -2.86
    assert result["stationary_5pct"] is False


def test_adf_too_short_raises():
    with pytest.raises(ValueError):
        adf_test([1.0, 2.0], lags=1)


def test_engle_granger_cointegrated():
    np.random.seed(3)
    n = 600
    x = np.cumsum(np.random.normal(0, 1, n))
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.6 * spread[t - 1] + np.random.normal(0, 0.5)
    y = 1.5 * x + spread + 5.0
    eg = engle_granger(y, x, lags=1)
    assert abs(eg["hedge_ratio"] - 1.5) < 0.1
    assert eg["cointegrated_5pct"] is True


def test_engle_granger_not_cointegrated():
    np.random.seed(4)
    n = 400
    x = np.cumsum(np.random.normal(0, 1, n))
    y = np.cumsum(np.random.normal(0, 1, n))
    eg = engle_granger(y, x, lags=1)
    assert eg["cointegrated_5pct"] is False


def test_half_life_mean_reverting():
    np.random.seed(5)
    n = 1000
    s = np.zeros(n)
    theta = 0.1
    for t in range(1, n):
        s[t] = (1 - theta) * s[t - 1] + np.random.normal(0, 0.5)
    hl = half_life(s)
    expected = np.log(2) / theta
    assert abs(hl - expected) / expected < 0.5


def test_half_life_explosive_returns_inf():
    s = np.zeros(200)
    for t in range(1, 200):
        s[t] = 1.05 * s[t - 1] + 1.0  # explosive (positive AR coefficient on lag)
    hl = half_life(s)
    assert hl == float("inf")


def test_zscore_spread_shape():
    s = np.random.normal(0, 1, 200)
    z = zscore_spread(s, window=30)
    assert len(z) == 200
    assert np.all(np.isnan(z[:29]))
    assert np.all(~np.isnan(z[29:]))


def test_zscore_spread_constant_zero():
    s = np.ones(50)
    z = zscore_spread(s, window=10)
    valid = z[~np.isnan(z)]
    assert np.all(valid == 0.0)
