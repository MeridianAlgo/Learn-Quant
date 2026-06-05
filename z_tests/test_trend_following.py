import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "Strategies - Trend Following"))
from trend_following import (
    atr,
    atr_position_size,
    donchian_breakout_signal,
    donchian_channel,
    ma_crossover_signal,
    trend_strength,
    tsmom_signal,
)


def test_donchian_channel_shape():
    h = np.linspace(1, 100, 100)
    l_ = h - 1
    res = donchian_channel(h, l_, window=20)
    assert len(res["upper"]) == 100
    assert np.all(np.isnan(res["upper"][:19]))
    assert np.all(~np.isnan(res["upper"][19:]))
    assert res["upper"][-1] == 100.0
    assert res["lower"][-1] == 80.0


def test_donchian_breakout_uptrend_long():
    prices = np.linspace(100, 200, 60)
    sig = donchian_breakout_signal(prices, prices, prices, 20, 10)
    assert sig[-1] == 1.0


def test_donchian_breakout_downtrend_flat():
    prices = np.concatenate([np.linspace(100, 200, 30), np.linspace(200, 100, 30)])
    sig = donchian_breakout_signal(prices, prices, prices, 20, 10)
    assert sig[-1] == 0.0


def test_ma_crossover_basic():
    prices = np.linspace(100, 200, 100)
    sig = ma_crossover_signal(prices, fast=10, slow=30)
    assert sig[-1] == 1.0


def test_ma_crossover_downtrend():
    prices = np.linspace(200, 100, 100)
    sig = ma_crossover_signal(prices, fast=10, slow=30)
    assert sig[-1] == -1.0


def test_tsmom_uptrend():
    prices = np.linspace(100, 200, 300)
    sig = tsmom_signal(prices, lookback=252)
    assert sig[-1] == 1.0


def test_tsmom_downtrend():
    prices = np.linspace(200, 100, 300)
    sig = tsmom_signal(prices, lookback=252)
    assert sig[-1] == -1.0


def test_atr_positive():
    np.random.seed(0)
    n = 100
    close = 100 + np.cumsum(np.random.normal(0, 1, n))
    high = close + 1.0
    low = close - 1.0
    a = atr(high, low, close, window=14)
    assert np.all(a[14:] > 0)
    assert np.all(np.isnan(a[:13]))


def test_atr_position_size_basic():
    size = atr_position_size(100_000, 0.01, 5.0, stop_atr_multiple=2.0)
    assert abs(size - 100.0) < 1e-9


def test_atr_position_size_zero_atr():
    assert atr_position_size(100_000, 0.01, 0.0, 2.0) == 0.0


def test_trend_strength_positive_uptrend():
    prices = np.linspace(100, 200, 100)
    s = trend_strength(prices, window=50)
    assert s[-1] > 0


def test_trend_strength_negative_downtrend():
    prices = np.linspace(200, 100, 100)
    s = trend_strength(prices, window=50)
    assert s[-1] < 0
