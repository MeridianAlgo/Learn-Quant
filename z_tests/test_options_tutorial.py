"""Tests for UTILS - Black-Scholes Option Pricing/options_tutorial.py."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "UTILS - Black-Scholes Option Pricing"))

from options_tutorial import _norm_cdf, _norm_pdf, black_scholes, greeks

# ---------------------------------------------------------------------------
# Black-Scholes pricing
# ---------------------------------------------------------------------------


class TestBlackScholes:
    def test_atm_call_positive(self):
        price = black_scholes(100, 100, 1.0, 0.05, 0.20, "call")
        assert price > 0

    def test_atm_put_positive(self):
        price = black_scholes(100, 100, 1.0, 0.05, 0.20, "put")
        assert price > 0

    def test_deep_itm_call_near_intrinsic(self):
        # Deep ITM call: S=200, K=100, T=0.01 (almost expired)
        price = black_scholes(200, 100, 0.01, 0.05, 0.20, "call")
        intrinsic = 200 - 100 * math.exp(-0.05 * 0.01)
        assert abs(price - intrinsic) < 2.0

    def test_deep_otm_call_near_zero(self):
        # Deep OTM, almost expired
        price = black_scholes(50, 100, 0.001, 0.05, 0.20, "call")
        assert price < 0.01

    def test_put_call_parity(self):
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        C = black_scholes(S, K, T, r, sigma, "call")
        P = black_scholes(S, K, T, r, sigma, "put")
        lhs = C - P
        rhs = S - K * math.exp(-r * T)
        assert abs(lhs - rhs) < 1e-8

    def test_call_increases_with_stock_price(self):
        c1 = black_scholes(90, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call")
        c2 = black_scholes(100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call")
        c3 = black_scholes(110, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call")
        assert c1 < c2 < c3

    def test_put_increases_as_stock_falls(self):
        p1 = black_scholes(110, K=100, T=1.0, r=0.05, sigma=0.20, option_type="put")
        p2 = black_scholes(100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="put")
        p3 = black_scholes(90, K=100, T=1.0, r=0.05, sigma=0.20, option_type="put")
        assert p1 < p2 < p3

    def test_call_increases_with_volatility(self):
        c_low = black_scholes(100, 100, 1.0, 0.05, 0.10, "call")
        c_high = black_scholes(100, 100, 1.0, 0.05, 0.40, "call")
        assert c_low < c_high

    def test_call_increases_with_time(self):
        c_short = black_scholes(100, 100, 0.25, 0.05, 0.20, "call")
        c_long = black_scholes(100, 100, 1.0, 0.05, 0.20, "call")
        assert c_short < c_long

    def test_expired_call_intrinsic_only(self):
        # T=0 → intrinsic value
        price = black_scholes(110, 100, 0.0, 0.05, 0.20, "call")
        assert abs(price - 10.0) < 1e-10

    def test_expired_put_intrinsic_only(self):
        price = black_scholes(90, 100, 0.0, 0.05, 0.20, "put")
        assert abs(price - 10.0) < 1e-10

    def test_expired_otm_call_zero(self):
        price = black_scholes(90, 100, 0.0, 0.05, 0.20, "call")
        assert price == 0.0

    def test_known_value(self):
        # Verified against standard BS calculator
        # S=100, K=100, T=1, r=0.05, sigma=0.2 → C ≈ 10.4506
        price = black_scholes(100, 100, 1.0, 0.05, 0.20, "call")
        assert abs(price - 10.4506) < 0.01


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------


class TestGreeks:
    def setup_method(self):
        self.g = greeks(100, 100, 1.0, 0.05, 0.20)

    def test_delta_call_range(self):
        # ATM call delta should be close to 0.5-0.6
        assert 0.5 < self.g["delta_call"] < 0.7

    def test_delta_put_negative(self):
        assert self.g["delta_put"] < 0

    def test_delta_put_call_relationship(self):
        # delta_call - delta_put = 1 (put-call parity for deltas)
        assert abs(self.g["delta_call"] - self.g["delta_put"] - 1.0) < 1e-10

    def test_gamma_positive(self):
        assert self.g["gamma"] > 0

    def test_theta_negative_for_call(self):
        # Time decay should always be negative for buyers
        assert self.g["theta_call"] < 0

    def test_vega_positive(self):
        assert self.g["vega"] > 0

    def test_rho_positive_for_call(self):
        assert self.g["rho_call"] > 0

    def test_deep_itm_delta_near_one(self):
        g_itm = greeks(200, 100, 1.0, 0.05, 0.20)
        assert g_itm["delta_call"] > 0.9

    def test_deep_otm_delta_near_zero(self):
        g_otm = greeks(50, 100, 1.0, 0.05, 0.20)
        assert g_otm["delta_call"] < 0.1


# ---------------------------------------------------------------------------
# Normal distribution helpers
# ---------------------------------------------------------------------------


class TestNormHelpers:
    def test_cdf_at_zero(self):
        assert abs(_norm_cdf(0.0) - 0.5) < 1e-6

    def test_cdf_symmetry(self):
        assert abs(_norm_cdf(1.0) + _norm_cdf(-1.0) - 1.0) < 1e-10

    def test_pdf_peak(self):
        assert abs(_norm_pdf(0.0) - 1.0 / math.sqrt(2 * math.pi)) < 1e-10

    def test_pdf_positive_everywhere(self):
        for x in [-3, -2, -1, 0, 1, 2, 3]:
            assert _norm_pdf(x) > 0
