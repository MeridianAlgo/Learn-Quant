import sys
from pathlib import Path

import numpy as np

sys.path.insert(
    0, str(Path(__file__).parent.parent / "UTILS - Finance - Volatility Calculator")
)

from volatility_calculator import (ewma_volatility, garman_klass_volatility,
                                   historical_volatility, parkinson_volatility,
                                   realized_volatility, volatility_cone)


def test_historical_volatility():
    prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]
    vol = historical_volatility(prices, window=5)
    assert isinstance(vol, float)
    assert vol > 0


def test_parkinson_volatility():
    high = [102, 104, 103, 105, 107]
    low = [98, 100, 99, 101, 103]
    vol = parkinson_volatility(high, low)
    assert isinstance(vol, float)
    assert vol > 0


def test_garman_klass_volatility():
    open_prices = [100, 102, 101, 103, 105]
    high = [102, 104, 103, 105, 107]
    low = [98, 100, 99, 101, 103]
    close = [101, 103, 102, 104, 106]
    vol = garman_klass_volatility(open_prices, high, low, close)
    assert isinstance(vol, float)
    assert vol > 0


def test_ewma_volatility():
    returns = [0.01, -0.02, 0.015, -0.01, 0.02, 0.005, -0.008]
    vols = ewma_volatility(returns)
    assert len(vols) == len(returns)
    assert all(v > 0 for v in vols)


def test_realized_volatility():
    intraday_returns = np.random.normal(0, 0.001, 100).tolist()
    vol = realized_volatility(intraday_returns)
    assert isinstance(vol, float)
    assert vol > 0


def test_volatility_cone():
    np.random.seed(42)
    prices = (100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 200)))).tolist()
    cone = volatility_cone(prices, windows=[10, 20, 30])
    assert len(cone) > 0
    for _window, stats in cone.items():
        assert "min" in stats
        assert "max" in stats
        assert "mean" in stats
        assert "current" in stats
