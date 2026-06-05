import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "CAPM"))
from capm_calculator import (
    capm_expected_return,
    estimate_beta,
    jensens_alpha,
    security_market_line,
)


def test_capm_known_value():
    # rf=3%, beta=1.2, rm=9% -> 0.03 + 1.2*0.06 = 0.102
    assert abs(capm_expected_return(0.03, 1.2, 0.09) - 0.102) < 1e-12


def test_beta_zero_returns_risk_free():
    assert abs(capm_expected_return(0.03, 0.0, 0.09) - 0.03) < 1e-12


def test_beta_one_returns_market():
    assert abs(capm_expected_return(0.03, 1.0, 0.09) - 0.09) < 1e-12


def test_capm_vectorised():
    out = capm_expected_return(0.03, np.array([0.5, 1.0, 1.5]), 0.09)
    assert np.allclose(out, [0.06, 0.09, 0.12])


def test_jensens_alpha_sign():
    # 14% actual vs 10.2% required -> positive alpha.
    a = jensens_alpha(0.14, 0.03, 1.2, 0.09)
    assert a > 0
    assert abs(a - (0.14 - 0.102)) < 1e-12


def test_estimate_beta_recovers_slope():
    rng = np.random.default_rng(0)
    mkt = rng.normal(0.0004, 0.01, 2000)
    asset = 1.3 * mkt + rng.normal(0, 0.003, 2000)
    assert abs(estimate_beta(asset, mkt) - 1.3) < 0.1


def test_security_market_line_monotonic():
    sml = security_market_line([0.0, 0.5, 1.0, 1.5], 0.03, 0.09)
    assert np.all(np.diff(sml) > 0)
    assert abs(sml[0] - 0.03) < 1e-12
