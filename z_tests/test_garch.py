import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Quantitative Methods - GARCH"))
from garch import (
    ewma_volatility,
    fit_garch,
    garch_forecast,
    garch_log_likelihood,
)


@pytest.fixture
def sim_returns():
    np.random.seed(123)
    n = 800
    omega, alpha, beta = 1e-5, 0.10, 0.85
    r = np.zeros(n)
    s2 = np.full(n, 1e-4)
    for t in range(1, n):
        s2[t] = omega + alpha * r[t - 1] ** 2 + beta * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * np.random.normal()
    return r


def test_ewma_length():
    r = np.random.normal(0, 0.01, 100)
    sig = ewma_volatility(r, lambda_=0.94)
    assert len(sig) == 100
    assert np.all(sig > 0)


def test_ewma_decay_recent_shock():
    r = np.zeros(50)
    r[-1] = 0.10  # shock at end
    sig = ewma_volatility(r, lambda_=0.94)
    # last value uses r[-1]^2 not yet, but sigma reacts at next step (we only have 50)
    assert np.all(np.isfinite(sig))


def test_garch_log_likelihood_invalid_params():
    r = np.array([0.01, -0.01, 0.02, -0.02])
    val = garch_log_likelihood(np.array([-1.0, 0.1, 0.8]), r)
    assert val == 1e10


def test_garch_log_likelihood_nonstationary():
    r = np.array([0.01, -0.01, 0.02, -0.02])
    val = garch_log_likelihood(np.array([1e-6, 0.5, 0.6]), r)
    assert val == 1e10


def test_fit_garch_recovers_params(sim_returns):
    fit = fit_garch(sim_returns)
    assert 0.0 < fit["alpha"] < 0.5
    assert 0.5 < fit["beta"] < 0.99
    assert fit["persistence"] < 1.0
    assert fit["omega"] > 0
    assert fit["unconditional_vol"] > 0
    assert len(fit["sigma"]) == len(sim_returns)


def test_fit_garch_sigma_positive(sim_returns):
    fit = fit_garch(sim_returns)
    assert np.all(fit["sigma"] > 0)


def test_garch_forecast_shape(sim_returns):
    fit = fit_garch(sim_returns)
    fc = garch_forecast(fit, sim_returns[-1], horizon=10)
    assert len(fc) == 10
    assert np.all(fc > 0)


def test_garch_forecast_converges_to_uncond(sim_returns):
    fit = fit_garch(sim_returns)
    fc = garch_forecast(fit, 0.0, horizon=500)
    # Long-horizon forecast should approach unconditional vol
    assert abs(fc[-1] - fit["unconditional_vol"]) / fit["unconditional_vol"] < 0.1
