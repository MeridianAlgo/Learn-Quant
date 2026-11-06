import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Market Microstructure - Liquidity Measures"))
from liquidity import (
    _simulate_bounce,
    amihud_illiquidity,
    corwin_schultz_spread,
    effective_spread,
    kyle_lambda,
    realized_spread,
    roll_spread,
)


def test_roll_recovers_simulated_spread():
    prices = _simulate_bounce(40_000, 0.10, 0.01, seed=1)
    assert roll_spread(prices) == pytest.approx(0.10, rel=0.08)


def test_roll_nan_when_autocovariance_positive():
    trending = np.cumsum(np.repeat([1.0, 2.0, 3.0, 4.0, 5.0], 4))
    assert math.isnan(roll_spread(trending))


def test_roll_validation():
    with pytest.raises(ValueError):
        roll_spread([1.0, 2.0, 3.0])


def test_corwin_schultz_zero_when_no_range():
    h = np.full(20, 10.0)
    assert corwin_schultz_spread(h, h) == 0.0


def test_corwin_schultz_non_negative():
    rng = np.random.default_rng(2)
    close = 20 * np.exp(np.cumsum(rng.normal(0, 0.02, 100)))
    wiggle = np.abs(rng.normal(0, 0.01, 100))
    assert corwin_schultz_spread(close * (1 + wiggle), close * (1 - wiggle)) >= 0.0


def test_corwin_schultz_rises_with_spread():
    rng = np.random.default_rng(3)
    close = 30 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    vol = np.abs(rng.normal(0, 0.008, 300))
    tight = corwin_schultz_spread(close * (1 + vol + 0.0005), close * (1 - vol - 0.0005))
    wide = corwin_schultz_spread(close * (1 + vol + 0.005), close * (1 - vol - 0.005))
    assert wide > tight


def test_corwin_schultz_validation():
    with pytest.raises(ValueError):
        corwin_schultz_spread([10, 11], [10])
    with pytest.raises(ValueError):
        corwin_schultz_spread([10, 9], [11, 8])
    with pytest.raises(ValueError):
        corwin_schultz_spread([10, 11], [0, 9])


def test_amihud_exact_value():
    value = amihud_illiquidity([0.01, -0.02], [1e6, 2e6])
    assert value == pytest.approx(0.01)


def test_amihud_skips_zero_volume_days():
    assert amihud_illiquidity([0.01, 0.5], [1e6, 0.0]) == pytest.approx(0.01)


def test_amihud_thin_name_is_less_liquid():
    r = np.full(10, 0.01)
    assert amihud_illiquidity(r, np.full(10, 1e6)) > amihud_illiquidity(r, np.full(10, 1e8))


def test_amihud_validation():
    with pytest.raises(ValueError):
        amihud_illiquidity([0.01], [0.0])
    with pytest.raises(ValueError):
        amihud_illiquidity([0.01, 0.02], [1e6])


def test_kyle_lambda_exact_line():
    q = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert kyle_lambda(0.3 * q + 1.0, q) == pytest.approx(0.3)


def test_kyle_lambda_noisy_recovery():
    rng = np.random.default_rng(4)
    q = rng.normal(0, 500, 3000)
    dp = 0.002 * q + rng.normal(0, 0.5, 3000)
    assert kyle_lambda(dp, q) == pytest.approx(0.002, rel=0.05)


def test_kyle_lambda_validation():
    with pytest.raises(ValueError):
        kyle_lambda([1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        kyle_lambda([1.0, 2.0, 3.0], [1.0, 2.0])


def test_effective_spread_buy_and_sell():
    eff = effective_spread([100.05, 99.95], [100.0, 100.0], [1, -1])
    assert np.allclose(eff, [0.001, 0.001])


def test_effective_spread_negative_when_trade_improves():
    assert effective_spread([99.99], [100.0], [1])[0] < 0


def test_effective_spread_validation():
    with pytest.raises(ValueError):
        effective_spread([100.0], [100.0], [0])
    with pytest.raises(ValueError):
        effective_spread([100.0, 101.0], [100.0], [1])
    with pytest.raises(ValueError):
        effective_spread([100.0], [0.0], [1])


def test_realized_below_effective_when_price_moves_with_trade():
    eff = effective_spread([100.05], [100.0], [1])[0]
    real = realized_spread([100.05], [100.03], [1])[0]
    assert real < eff
