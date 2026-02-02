import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).parent.parent / "UTILS - Finance - Beta Calculator")
)

from beta_calculator import (adjusted_beta, beta_decomposition, calculate_beta,
                             downside_beta, levered_beta, rolling_beta,
                             unlevered_beta, upside_beta)


def test_calculate_beta():
    asset_returns = [0.01, -0.02, 0.015, -0.01, 0.02]
    market_returns = [0.008, -0.015, 0.012, -0.008, 0.018]
    beta = calculate_beta(asset_returns, market_returns)
    assert isinstance(beta, float)
    assert -5 < beta < 5


def test_rolling_beta():
    asset_returns = [0.01, -0.02, 0.015, -0.01, 0.02, 0.005, -0.008]
    market_returns = [0.008, -0.015, 0.012, -0.008, 0.018, 0.004, -0.006]
    betas = rolling_beta(asset_returns, market_returns, window=3)
    assert len(betas) == len(asset_returns) - 3 + 1
    assert all(isinstance(b, float) for b in betas)


def test_levered_unlevered_beta():
    unlevered = 1.0
    debt_to_equity = 0.5
    levered = levered_beta(unlevered, debt_to_equity)
    assert levered > unlevered
    recovered = unlevered_beta(levered, debt_to_equity)
    assert abs(recovered - unlevered) < 0.001


def test_downside_upside_beta():
    asset_returns = [0.01, -0.02, 0.015, -0.01, 0.02, -0.005, 0.008]
    market_returns = [0.008, -0.015, 0.012, -0.008, 0.018, -0.004, 0.006]
    down_beta = downside_beta(asset_returns, market_returns)
    up_beta = upside_beta(asset_returns, market_returns)
    assert isinstance(down_beta, float)
    assert isinstance(up_beta, float)


def test_beta_decomposition():
    asset_returns = [0.01, -0.02, 0.015, -0.01, 0.02, -0.005, 0.008]
    market_returns = [0.008, -0.015, 0.012, -0.008, 0.018, -0.004, 0.006]
    decomp = beta_decomposition(asset_returns, market_returns)
    assert "total_beta" in decomp
    assert "downside_beta" in decomp
    assert "upside_beta" in decomp
    assert "beta_asymmetry" in decomp


def test_adjusted_beta():
    historical_beta = 1.5
    adj_beta = adjusted_beta(historical_beta)
    assert 1.0 < adj_beta < historical_beta
