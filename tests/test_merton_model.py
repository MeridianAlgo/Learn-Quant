import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "UTILS - Finance - Credit Risk"))
from merton_model import implied_asset_value, merton_equity, merton_model

BASE = dict(V=100e6, F=80e6, r=0.05, sigma_V=0.20, T=1.0)


def test_equity_positive():
    eq = merton_equity(**BASE)
    assert eq > 0


def test_equity_less_than_assets():
    eq = merton_equity(**BASE)
    assert eq < BASE["V"]


def test_equity_increases_with_asset_value():
    eq_low = merton_equity(V=90e6, F=80e6, r=0.05, sigma_V=0.20, T=1.0)
    eq_high = merton_equity(V=120e6, F=80e6, r=0.05, sigma_V=0.20, T=1.0)
    assert eq_high > eq_low


def test_model_output_keys():
    result = merton_model(**BASE)
    for key in ["equity_value", "debt_value", "distance_to_default",
                "probability_of_default", "credit_spread_bps", "leverage"]:
        assert key in result


def test_equity_plus_debt_approx_assets():
    result = merton_model(**BASE)
    total = result["equity_value"] + result["debt_value"]
    assert abs(total - BASE["V"]) < 1.0  # Should sum to asset value


def test_pd_in_zero_one():
    result = merton_model(**BASE)
    assert 0 <= result["probability_of_default"] <= 1


def test_pd_higher_for_high_leverage():
    low_lev = merton_model(V=100e6, F=50e6, r=0.05, sigma_V=0.20, T=1.0)
    high_lev = merton_model(V=100e6, F=90e6, r=0.05, sigma_V=0.20, T=1.0)
    assert high_lev["probability_of_default"] > low_lev["probability_of_default"]


def test_credit_spread_non_negative():
    result = merton_model(**BASE)
    assert result["credit_spread_bps"] >= 0


def test_implied_asset_value_returns_dict():
    result = implied_asset_value(E=25e6, F=80e6, r=0.05, sigma_E=0.40, T=1.0)
    assert "asset_value" in result
    assert "asset_volatility" in result
    assert result["asset_value"] > 0
    assert result["asset_volatility"] > 0
