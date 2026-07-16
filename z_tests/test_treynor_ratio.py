import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Finance - Treynor Ratio"))
from treynor_ratio import annualized_treynor_ratio, beta, treynor_ratio


def test_beta_of_market_with_itself_is_one():
    rng = np.random.default_rng(0)
    market = rng.normal(0, 0.01, 300).tolist()
    assert abs(beta(market, market) - 1.0) < 1e-12


def test_beta_scales_with_leverage():
    rng = np.random.default_rng(1)
    market = rng.normal(0, 0.01, 500)
    levered = (2.0 * market).tolist()
    assert abs(beta(levered, market.tolist()) - 2.0) < 1e-10


def test_negative_beta_for_inverse_asset():
    rng = np.random.default_rng(2)
    market = rng.normal(0, 0.01, 500)
    inverse = (-market).tolist()
    assert abs(beta(inverse, market.tolist()) + 1.0) < 1e-10


def test_treynor_known_value():
    market = [0.01, -0.01, 0.02, -0.02, 0.015, -0.015]
    asset = [0.02, -0.02, 0.04, -0.04, 0.03, -0.03]  # beta 2, mean 0
    assert abs(treynor_ratio(asset, market) - 0.0) < 1e-12


def test_treynor_positive_for_outperformer():
    rng = np.random.default_rng(3)
    market = rng.normal(0.0, 0.01, 500)
    asset = (market + 0.001).tolist()
    assert treynor_ratio(asset, market.tolist()) > 0


def test_higher_beta_lowers_treynor_for_same_excess_return():
    rng = np.random.default_rng(4)
    market = rng.normal(0.0, 0.01, 1000)
    low = (0.5 * market + 0.001).tolist()
    high = (2.0 * market + 0.001).tolist()
    m = market.tolist()
    assert treynor_ratio(low, m) > treynor_ratio(high, m)


def test_annualized_scales_by_periods():
    rng = np.random.default_rng(5)
    market = rng.normal(0.0004, 0.01, 400).tolist()
    asset = (np.asarray(market) * 1.2 + 0.0002).tolist()
    per_period = treynor_ratio(asset, market)
    assert abs(annualized_treynor_ratio(asset, market) - per_period * 252) < 1e-12


def test_risk_free_rate_reduces_ratio():
    rng = np.random.default_rng(6)
    market = rng.normal(0.001, 0.01, 400).tolist()
    asset = (np.asarray(market) + 0.0005).tolist()
    assert treynor_ratio(asset, market, 0.0005) < treynor_ratio(asset, market, 0.0)


def test_zero_beta_raises():
    rng = np.random.default_rng(7)
    market = rng.normal(0, 0.01, 500)
    with pytest.raises(ValueError):
        treynor_ratio([0.001] * 500, market.tolist())


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        beta([0.01, 0.02], [0.01])


def test_too_few_observations_raise():
    with pytest.raises(ValueError):
        beta([0.01], [0.01])
