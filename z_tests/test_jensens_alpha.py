import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Finance - Jensens Alpha"))
from jensens_alpha import (
    alpha_t_statistic,
    annualized_alpha,
    beta,
    expected_return,
    jensens_alpha,
)


def test_beta_of_market_with_itself_is_one():
    rng = np.random.default_rng(0)
    market = rng.normal(0, 0.01, 300).tolist()
    assert abs(beta(market, market) - 1.0) < 1e-12


def test_market_has_zero_alpha_against_itself():
    rng = np.random.default_rng(1)
    market = rng.normal(0.0005, 0.01, 500).tolist()
    assert abs(jensens_alpha(market, market, 0.0002)) < 1e-12


def test_pure_leverage_produces_no_alpha():
    rng = np.random.default_rng(2)
    market = rng.normal(0.0005, 0.01, 800)
    levered = (2.0 * market).tolist()
    assert abs(jensens_alpha(levered, market.tolist(), 0.0)) < 1e-12


def test_constant_edge_shows_up_as_alpha():
    rng = np.random.default_rng(3)
    market = rng.normal(0.0004, 0.01, 800)
    skilled = (market + 0.001).tolist()
    assert abs(jensens_alpha(skilled, market.tolist()) - 0.001) < 1e-12


def test_underperformer_has_negative_alpha():
    rng = np.random.default_rng(4)
    market = rng.normal(0.0004, 0.01, 600)
    laggard = (market - 0.0008).tolist()
    assert jensens_alpha(laggard, market.tolist()) < 0


def test_expected_return_matches_capm_formula():
    rng = np.random.default_rng(5)
    market = rng.normal(0.0004, 0.01, 400)
    asset = (1.3 * market).tolist()
    m = market.tolist()
    rf = 0.0001
    manual = rf + beta(asset, m) * (float(np.mean(market)) - rf)
    assert abs(expected_return(asset, m, rf) - manual) < 1e-15


def test_alpha_is_actual_minus_expected():
    rng = np.random.default_rng(6)
    market = rng.normal(0.0004, 0.01, 400)
    asset = (0.8 * market + 0.0005).tolist()
    m = market.tolist()
    gap = float(np.mean(asset)) - expected_return(asset, m, 0.0001)
    assert abs(jensens_alpha(asset, m, 0.0001) - gap) < 1e-15


def test_annualized_scales_by_periods():
    rng = np.random.default_rng(7)
    market = rng.normal(0.0004, 0.01, 400)
    asset = (1.1 * market + 0.0003).tolist()
    m = market.tolist()
    assert abs(annualized_alpha(asset, m) - jensens_alpha(asset, m) * 252) < 1e-15


def test_t_statistic_large_for_a_clean_edge():
    rng = np.random.default_rng(8)
    market = rng.normal(0.0004, 0.01, 1000)
    skilled = (market + 0.001 + rng.normal(0, 0.0005, 1000)).tolist()
    assert alpha_t_statistic(skilled, market.tolist()) > 2.0


def test_t_statistic_small_for_noise():
    rng = np.random.default_rng(9)
    market = rng.normal(0.0004, 0.01, 1000)
    noisy = (market + rng.normal(0, 0.02, 1000)).tolist()
    assert abs(alpha_t_statistic(noisy, market.tolist())) < 2.0


def test_t_statistic_sign_follows_alpha():
    rng = np.random.default_rng(10)
    market = rng.normal(0.0004, 0.01, 600)
    laggard = (market - 0.001 + rng.normal(0, 0.001, 600)).tolist()
    m = market.tolist()
    assert jensens_alpha(laggard, m) < 0
    assert alpha_t_statistic(laggard, m) < 0


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        beta([0.01, 0.02], [0.01])


def test_zero_variance_market_raises():
    with pytest.raises(ValueError):
        jensens_alpha([0.01, 0.02, 0.03], [0.01, 0.01, 0.01])


def test_too_few_observations_for_t_statistic_raise():
    with pytest.raises(ValueError):
        alpha_t_statistic([0.01, 0.02], [0.01, 0.03])
