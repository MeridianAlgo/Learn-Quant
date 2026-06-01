import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "Portfolio Management - Risk Parity"))
from risk_parity import (
    inverse_volatility_weights,
    portfolio_volatility,
    risk_contributions,
    risk_parity_weights,
)


def _sample_cov():
    vols = np.array([0.20, 0.10, 0.04])
    corr = np.array([[1.0, 0.3, 0.05], [0.3, 1.0, 0.15], [0.05, 0.15, 1.0]])
    return np.outer(vols, vols) * corr


def test_portfolio_volatility_matches_quadratic_form():
    cov = _sample_cov()
    w = np.array([0.5, 0.3, 0.2])
    expected = np.sqrt(w @ cov @ w)
    assert abs(portfolio_volatility(w, cov) - expected) < 1e-12


def test_risk_contributions_sum_to_total_vol():
    cov = _sample_cov()
    w = np.array([0.4, 0.4, 0.2])
    rc = risk_contributions(w, cov)
    assert abs(rc.sum() - portfolio_volatility(w, cov)) < 1e-10


def test_inverse_vol_weights_sum_to_one():
    cov = _sample_cov()
    w = inverse_volatility_weights(cov)
    assert abs(w.sum() - 1.0) < 1e-12
    # Lowest-vol asset (cash) should get the largest weight.
    assert np.argmax(w) == 2


def test_erc_equalises_risk_contributions():
    cov = _sample_cov()
    w = risk_parity_weights(cov)
    rc = risk_contributions(w, cov)
    frac = rc / rc.sum()
    # All three fractional contributions should be ~1/3.
    assert np.allclose(frac, 1 / 3, atol=1e-3)
    assert abs(w.sum() - 1.0) < 1e-6


def test_risk_budget_respected():
    cov = _sample_cov()
    budget = np.array([0.5, 0.3, 0.2])
    w = risk_parity_weights(cov, budget=budget)
    frac = risk_contributions(w, cov)
    frac = frac / frac.sum()
    assert np.allclose(frac, budget, atol=5e-3)
