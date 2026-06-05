import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "Value at Risk (VaR)"))
from var_calculator import (
    conditional_var,
    historical_var,
    kupiec_pof_test,
    monte_carlo_var,
    parametric_var,
    value_at_risk,
)


def _returns(seed=0, n=2000):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0005, 0.02, n)


def test_parametric_matches_alias():
    r = _returns()
    assert abs(parametric_var(r, 0.95) - value_at_risk(r, 0.95)) < 1e-12


def test_var_positive_and_increases_with_confidence():
    r = _returns()
    for fn in (parametric_var, historical_var):
        assert fn(r, 0.95) > 0
        assert fn(r, 0.99) > fn(r, 0.95)


def test_cvar_at_least_var():
    r = _returns(seed=1)
    assert conditional_var(r, 0.95) >= historical_var(r, 0.95)


def test_monte_carlo_close_to_parametric():
    r = _returns(seed=2)
    mc = monte_carlo_var(r, 0.95, n_sims=200_000, seed=7)
    pv = parametric_var(r, 0.95)
    # Both fit a normal model, so they should agree closely.
    assert abs(mc - pv) < 0.002


def test_kupiec_accepts_good_model():
    r = _returns(seed=3, n=3000)
    var95 = parametric_var(r, 0.95)
    res = kupiec_pof_test(r, var95, 0.95)
    assert 0.0 <= res["p_value"] <= 1.0
    assert "exceptions" in res
    # A normal VaR on normal data should not be rejected.
    assert not res["reject"]


def test_kupiec_flags_miscalibrated_var():
    r = _returns(seed=4, n=3000)
    # A wildly too-small VaR should produce far too many exceptions.
    res = kupiec_pof_test(r, 0.001, 0.95)
    assert res["exceptions"] > res["expected"]
    assert res["reject"]
