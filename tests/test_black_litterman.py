import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "UTILS - Portfolio Management - Black Litterman"))
from black_litterman import black_litterman, bl_optimal_weights, market_implied_returns


@pytest.fixture
def market_setup():
    corr = np.array([
        [1.00, 0.75, -0.20, 0.30],
        [0.75, 1.00, -0.15, 0.35],
        [-0.20, -0.15, 1.00, -0.05],
        [0.30, 0.35, -0.05, 1.00],
    ])
    vols = np.array([0.16, 0.18, 0.05, 0.20])
    cov = np.outer(vols, vols) * corr
    weights = np.array([0.40, 0.30, 0.20, 0.10])
    return cov, weights


def test_implied_returns_shape(market_setup):
    cov, weights = market_setup
    pi = market_implied_returns(cov, weights)
    assert pi.shape == (4,)


def test_implied_returns_positive_for_equity_heavy(market_setup):
    cov, weights = market_setup
    pi = market_implied_returns(cov, weights)
    assert pi[0] > 0  # US equity should have positive implied return


def test_bl_output_keys(market_setup):
    cov, weights = market_setup
    P = np.array([[1, -1, 0, 0]])
    Q = np.array([0.02])
    result = black_litterman(cov, weights, P, Q)
    assert "posterior_returns" in result
    assert "posterior_covariance" in result
    assert "equilibrium_returns" in result


def test_bl_view_tilts_return(market_setup):
    cov, weights = market_setup
    eq = market_implied_returns(cov, weights)
    # View: asset 0 outperforms asset 1 by 5%
    P = np.array([[1, -1, 0, 0]])
    Q = np.array([0.05])
    result = black_litterman(cov, weights, P, Q)
    bl_spread = result["posterior_returns"][0] - result["posterior_returns"][1]
    eq_spread = eq[0] - eq[1]
    assert bl_spread > eq_spread  # View should tilt toward asset 0


def test_bl_optimal_weights_sum_abs_one(market_setup):
    cov, weights = market_setup
    P = np.array([[1, -1, 0, 0]])
    Q = np.array([0.02])
    result = black_litterman(cov, weights, P, Q)
    opt = bl_optimal_weights(result["posterior_returns"], result["posterior_covariance"])
    assert abs(np.sum(np.abs(opt)) - 1.0) < 1e-6


def test_bl_posterior_cov_symmetric(market_setup):
    cov, weights = market_setup
    P = np.array([[1, -1, 0, 0]])
    Q = np.array([0.02])
    result = black_litterman(cov, weights, P, Q)
    post_cov = result["posterior_covariance"]
    assert np.allclose(post_cov, post_cov.T, atol=1e-10)
