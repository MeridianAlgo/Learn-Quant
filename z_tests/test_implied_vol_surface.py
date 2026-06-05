import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "Finance - Implied Volatility Surface"))
from implied_vol_surface import (
    VolSurface,
    bs_price,
    bs_vega,
    implied_vol,
)


def test_put_call_parity():
    S, K, T, r, sigma = 100, 100, 1.0, 0.03, 0.2
    call = bs_price(S, K, T, r, sigma, "call")
    put = bs_price(S, K, T, r, sigma, "put")
    # C - P = S - K e^{-rT}
    assert abs((call - put) - (S - K * np.exp(-r * T))) < 1e-9


def test_implied_vol_roundtrip():
    S, K, T, r, true_sigma = 100, 110, 0.75, 0.02, 0.27
    price = bs_price(S, K, T, r, true_sigma, "call")
    iv = implied_vol(price, S, K, T, r, "call")
    assert abs(iv - true_sigma) < 1e-4


def test_implied_vol_put_roundtrip():
    S, K, T, r, true_sigma = 100, 90, 0.5, 0.01, 0.35
    price = bs_price(S, K, T, r, true_sigma, "put")
    iv = implied_vol(price, S, K, T, r, "put")
    assert abs(iv - true_sigma) < 1e-4


def test_vega_positive():
    assert bs_vega(100, 100, 1.0, 0.02, 0.2) > 0


def test_arbitrage_violation_returns_nan():
    # Price below intrinsic value is impossible.
    iv = implied_vol(0.01, 100, 80, 1.0, 0.02, "call")
    assert np.isnan(iv)


def test_surface_recovers_and_interpolates():
    S, r = 100.0, 0.02
    strikes = np.array([80, 90, 100, 110, 120], dtype=float)
    maturities = np.array([0.25, 0.5, 1.0], dtype=float)

    def true_iv(K, T):
        return 0.20 - 0.3 * np.log(K / S) + 0.05 * np.sqrt(T)

    grid = np.array([[bs_price(S, K, T, r, true_iv(K, T), "call") for K in strikes] for T in maturities])
    surf = VolSurface(S, r).fit(strikes, maturities, grid, "call")

    # On-grid IV should match the truth closely.
    assert abs(surf.iv(100, 0.5) - true_iv(100, 0.5)) < 1e-3
    # Interpolated point lies within the surrounding values.
    val = surf.iv(95, 0.4)
    assert 0.0 < val < 1.0
    assert surf.smile(0.5).shape == (5,)
