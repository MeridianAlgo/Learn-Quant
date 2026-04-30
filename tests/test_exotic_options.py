import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "UTILS - Finance - Exotic Options"))
from exotic_options import asian_option, barrier_option, lookback_option

BASE = dict(S0=100, r=0.05, sigma=0.20, T=1.0)


def test_barrier_option_positive():
    price = barrier_option(**BASE, K=100, H=90, barrier_type="down-out")
    assert price >= 0


def test_barrier_in_out_parity():
    """Down-and-in + down-and-out should equal vanilla call (approximately)."""
    common = dict(**BASE, K=100, H=80, n_steps=100, n_paths=10_000)
    do = barrier_option(**common, barrier_type="down-out")
    di = barrier_option(**common, barrier_type="down-in")
    # Black-Scholes vanilla call for reference
    from scipy.stats import norm
    d1 = (np.log(BASE["S0"] / 100) + (BASE["r"] + 0.5 * BASE["sigma"]**2) * BASE["T"]) / (BASE["sigma"] * np.sqrt(BASE["T"]))
    d2 = d1 - BASE["sigma"] * np.sqrt(BASE["T"])
    vanilla = BASE["S0"] * norm.cdf(d1) - 100 * np.exp(-BASE["r"] * BASE["T"]) * norm.cdf(d2)
    assert abs((do + di) - vanilla) / vanilla < 0.15  # within 15% (MC noise)


def test_down_out_cheaper_than_vanilla():
    """Down-and-out call < vanilla call (it can knock out)."""
    from scipy.stats import norm
    d1 = (np.log(100 / 100) + (0.05 + 0.5 * 0.04)) / (0.20 * 1.0)
    d2 = d1 - 0.20
    vanilla = 100 * norm.cdf(d1) - 100 * np.exp(-0.05) * norm.cdf(d2)
    do = barrier_option(**BASE, K=100, H=90, barrier_type="down-out", n_paths=20_000)
    assert do <= vanilla + 1.0  # Small tolerance for MC noise


def test_asian_option_positive():
    price = asian_option(**BASE, K=100, n_paths=10_000)
    assert price >= 0


def test_asian_cheaper_than_vanilla():
    """Asian call < vanilla call (averaging reduces terminal uncertainty)."""
    from scipy.stats import norm
    d1 = (np.log(100 / 100) + (0.05 + 0.5 * 0.04)) / (0.20 * 1.0)
    d2 = d1 - 0.20
    vanilla = 100 * norm.cdf(d1) - 100 * np.exp(-0.05) * norm.cdf(d2)
    asian = asian_option(**BASE, K=100, n_paths=20_000)
    assert asian <= vanilla + 2.0


def test_geometric_asian_cheaper_than_arithmetic():
    arith = asian_option(**BASE, K=100, averaging="arithmetic", n_paths=20_000)
    geom = asian_option(**BASE, K=100, averaging="geometric", n_paths=20_000)
    # Geometric mean <= arithmetic mean, so geometric option <= arithmetic
    assert geom <= arith + 1.0


def test_lookback_option_positive():
    price = lookback_option(**BASE, n_paths=5_000)
    assert price >= 0


def test_lookback_more_expensive_than_vanilla():
    """Lookback call >= vanilla call (uses best price)."""
    from scipy.stats import norm
    d1 = (np.log(100 / 100) + (0.05 + 0.5 * 0.04)) / (0.20 * 1.0)
    d2 = d1 - 0.20
    vanilla = 100 * norm.cdf(d1) - 100 * np.exp(-0.05) * norm.cdf(d2)
    lb = lookback_option(**BASE, n_paths=20_000, option_type="call", fixed_strike=100)
    assert lb >= vanilla - 1.0  # Small tolerance for MC noise
