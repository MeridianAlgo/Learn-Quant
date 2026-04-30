import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "UTILS - Finance - Transaction Cost Analysis"))
from tca_utils import (
    almgren_chriss_impact,
    implementation_shortfall,
    sqrt_market_impact,
    twap,
    vwap,
    vwap_slippage,
)


def test_vwap_basic():
    prices = [10, 20, 30]
    volumes = [100, 100, 100]
    assert abs(vwap(prices, volumes) - 20.0) < 1e-10


def test_vwap_volume_weighted():
    prices = [10, 20]
    volumes = [300, 100]
    expected = (10 * 300 + 20 * 100) / 400  # = 12.5
    assert abs(vwap(prices, volumes) - expected) < 1e-10


def test_vwap_zero_volume_raises():
    with pytest.raises(ValueError):
        vwap([100, 200], [0, 0])


def test_twap_basic():
    prices = [10, 20, 30]
    assert abs(twap(prices) - 20.0) < 1e-10


def test_vwap_slippage_buy_positive_when_exec_above_vwap():
    slippage = vwap_slippage(execution_price=100.10, vwap_price=100.00, side="buy")
    assert slippage > 0


def test_vwap_slippage_buy_negative_when_exec_below_vwap():
    slippage = vwap_slippage(execution_price=99.90, vwap_price=100.00, side="buy")
    assert slippage < 0


def test_vwap_slippage_sell_positive_when_exec_below_vwap():
    slippage = vwap_slippage(execution_price=99.90, vwap_price=100.00, side="sell")
    assert slippage > 0


def test_is_keys():
    result = implementation_shortfall(
        decision_price=100.0,
        execution_prices=[100.05, 100.10],
        execution_quantities=[500, 500],
    )
    for key in ["average_execution_price", "implementation_shortfall_bps", "total_cost_bps"]:
        assert key in result


def test_is_positive_when_exec_above_decision():
    result = implementation_shortfall(
        decision_price=100.0,
        execution_prices=[100.10, 100.20],
        execution_quantities=[1000, 1000],
    )
    assert result["implementation_shortfall_bps"] > 0


def test_is_zero_volume_raises():
    with pytest.raises(ValueError):
        implementation_shortfall(100.0, [100.1], [0])


def test_almgren_chriss_keys():
    result = almgren_chriss_impact(100_000, 1_000_000, 0.015, T=5)
    for key in ["participation_rate", "temporary_impact_bps", "permanent_impact_bps",
                "expected_shortfall_bps", "timing_risk_variance"]:
        assert key in result


def test_almgren_chriss_impact_positive():
    result = almgren_chriss_impact(100_000, 1_000_000, 0.015, T=5)
    assert result["temporary_impact_bps"] > 0
    assert result["permanent_impact_bps"] > 0


def test_sqrt_impact_positive():
    impact = sqrt_market_impact(100_000, 1_000_000, 0.015)
    assert impact > 0


def test_sqrt_impact_larger_order():
    small = sqrt_market_impact(10_000, 1_000_000, 0.015)
    large = sqrt_market_impact(100_000, 1_000_000, 0.015)
    assert large > small
