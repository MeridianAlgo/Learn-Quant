import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Risk Metrics - Stress Testing"))
from stress_testing import (
    HISTORICAL_SCENARIOS,
    apply_scenario,
    historical_scenario,
    reverse_stress_test,
    sensitivity_analysis,
    worst_case,
)


def test_apply_scenario_basic():
    res = apply_scenario([0.5, 0.5], [-0.10, -0.20], portfolio_value=1000)
    assert abs(res["portfolio_return"] - (-0.15)) < 1e-12
    assert abs(res["portfolio_pnl"] - (-150.0)) < 1e-12


def test_apply_scenario_length_mismatch():
    with pytest.raises(ValueError):
        apply_scenario([0.5, 0.5], [-0.10])


def test_historical_scenario_known():
    res = historical_scenario({"equity": 1.0}, "2008_GFC", 1000)
    assert abs(res["portfolio_pnl"] - (-500.0)) < 1e-9


def test_historical_scenario_unknown_raises():
    with pytest.raises(ValueError):
        historical_scenario({"equity": 1.0}, "FAKE", 1000)


def test_historical_scenarios_keys_present():
    expected = {"2008_GFC", "2020_COVID", "1987_Black_Monday", "2000_DotCom", "2022_Inflation"}
    assert expected.issubset(HISTORICAL_SCENARIOS.keys())


def test_sensitivity_default_range():
    res = sensitivity_analysis([1.0, 0.0], asset_index=0)
    assert len(res["shocks"]) == 21
    assert len(res["pnls"]) == 21
    assert res["pnls"][0] < res["pnls"][-1]


def test_sensitivity_zero_weight_no_pnl():
    res = sensitivity_analysis([1.0, 0.0], asset_index=1, portfolio_value=100)
    assert np.allclose(res["pnls"], 0.0)


def test_reverse_stress_test_basic():
    k = reverse_stress_test([1.0], [-1.0], 0.20, 1.0)
    assert abs(k - 0.20) < 1e-12


def test_reverse_stress_test_no_loss_direction():
    k = reverse_stress_test([1.0], [1.0], 0.20, 1.0)
    assert np.isnan(k)


def test_worst_case():
    scenarios = [[-0.10, -0.10], [-0.30, 0.05], [0.10, 0.10]]
    res = worst_case([0.5, 0.5], scenarios, portfolio_value=1000)
    assert res["worst_scenario_index"] == 1
    assert abs(res["worst_pnl"] - (-125.0)) < 1e-9
