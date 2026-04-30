import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "UTILS - Finance - FX Tools"))
from fx_tools import (
    cip_deviation,
    cross_rate,
    forward_points,
    forward_rate,
    garman_kohlhagen,
    triangular_arbitrage_profit,
)


def test_forward_rate_higher_when_domestic_greater():
    """Higher domestic rate → forward > spot."""
    fwd = forward_rate(spot=1.10, r_domestic=0.05, r_foreign=0.02, T=1.0)
    assert fwd > 1.10


def test_forward_rate_lower_when_foreign_greater():
    """Higher foreign rate → forward < spot."""
    fwd = forward_rate(spot=1.10, r_domestic=0.02, r_foreign=0.05, T=1.0)
    assert fwd < 1.10


def test_forward_equals_spot_when_rates_equal():
    fwd = forward_rate(spot=1.10, r_domestic=0.04, r_foreign=0.04, T=1.0)
    assert abs(fwd - 1.10) < 1e-6


def test_forward_points_positive_when_domestic_higher():
    pts = forward_points(1.10, r_domestic=0.05, r_foreign=0.02, T=1.0)
    assert pts > 0


def test_cip_deviation_zero_for_fair_forward():
    """If forward matches CIP exactly, deviation should be 0."""
    spot = 1.10
    r_d, r_f = 0.05, 0.02
    fair_fwd = forward_rate(spot, r_d, r_f, 1.0)
    dev = cip_deviation(spot, fair_fwd, r_d, r_f, 1.0)
    assert abs(dev) < 1e-6


def test_cross_rate_basic():
    # USD/EUR = 1.10, USD/GBP = 1.30 → EUR/GBP = 1.30/1.10 ≈ 1.1818
    cr = cross_rate(s_ab=1.10, s_ac=1.30)
    assert abs(cr - 1.30 / 1.10) < 1e-10


def test_triangular_arb_no_profit_for_consistent_rates():
    # Consistent rates: s_ab=1.2, s_bc=1.0/1.2, s_ca=1.0 → product=1
    s_ab = 1.2
    s_bc = 1.0 / 1.2
    s_ca = 1.0
    profit = triangular_arbitrage_profit(s_ab, s_bc, s_ca, notional=1)
    assert abs(profit) < 1e-10


def test_gk_call_price_positive():
    result = garman_kohlhagen(S=1.10, K=1.10, r_d=0.05, r_f=0.02, sigma=0.10, T=0.25)
    assert result["price"] > 0


def test_gk_put_price_positive():
    result = garman_kohlhagen(S=1.10, K=1.10, r_d=0.05, r_f=0.02, sigma=0.10, T=0.25, option_type="put")
    assert result["price"] > 0


def test_gk_call_delta_between_zero_and_one():
    result = garman_kohlhagen(S=1.10, K=1.10, r_d=0.05, r_f=0.02, sigma=0.10, T=0.25)
    assert 0 < result["delta"] < 1


def test_gk_put_call_parity():
    """Put-call parity: C - P = S*exp(-r_f*T) - K*exp(-r_d*T)."""
    S, K, r_d, r_f, sigma, T = 1.10, 1.10, 0.05, 0.02, 0.10, 1.0
    call = garman_kohlhagen(S, K, r_d, r_f, sigma, T, "call")
    put = garman_kohlhagen(S, K, r_d, r_f, sigma, T, "put")
    parity = S * np.exp(-r_f * T) - K * np.exp(-r_d * T)
    assert abs((call["price"] - put["price"]) - parity) < 1e-6
