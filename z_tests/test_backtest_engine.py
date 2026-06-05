import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "Strategies - Backtesting Engine"))
from backtest_engine import (
    max_drawdown,
    performance_summary,
    run_backtest,
    sma_crossover_signal,
    to_returns,
)


def test_to_returns_alignment():
    prices = [100.0, 110.0, 99.0]
    r = to_returns(prices)
    assert r[0] == 0.0
    assert abs(r[1] - 0.10) < 1e-12
    assert abs(r[2] - (99.0 / 110.0 - 1.0)) < 1e-12


def test_no_lookahead_constant_long():
    # A constant long position should reproduce buy & hold (minus costs=0).
    prices = np.array([100.0, 101.0, 102.0, 100.0, 105.0])
    res = run_backtest(prices, np.ones_like(prices), fee_bps=0.0, slippage_bps=0.0)
    assert abs(res["equity"][-1] - prices[-1] / prices[0]) < 1e-9


def test_costs_charged_on_turnover():
    prices = np.array([100.0, 100.0, 100.0, 100.0])
    # Flat prices: any P&L must come from costs only.
    signal = np.array([0.0, 1.0, 0.0, 1.0])
    res = run_backtest(prices, signal, fee_bps=10.0, slippage_bps=0.0)
    assert res["equity"][-1] < 1.0  # costs eroded capital
    assert res["costs"].sum() > 0.0


def test_max_drawdown_basic():
    equity = np.array([1.0, 1.2, 0.9, 1.1, 0.6])
    dd = max_drawdown(equity)
    # Worst peak (1.2) to trough (0.6): 0.6/1.2 - 1 = -0.5
    assert abs(dd["max_drawdown"] - (-0.5)) < 1e-9
    assert dd["trough_index"] == 4


def test_performance_summary_keys_and_sign():
    rng = np.random.default_rng(0)
    prices = 100 * np.cumprod(1 + 0.001 + 0.01 * rng.standard_normal(500))
    res = run_backtest(prices, np.ones(len(prices)), fee_bps=0.0, slippage_bps=0.0)
    stats = performance_summary(res)
    for key in ["cagr", "sharpe", "max_drawdown", "sortino", "calmar", "hit_rate"]:
        assert key in stats
    assert stats["max_drawdown"] <= 0.0
    assert 0.0 <= stats["hit_rate"] <= 1.0


def test_sma_signal_is_binary():
    rng = np.random.default_rng(1)
    prices = 100 * np.cumprod(1 + 0.01 * rng.standard_normal(200))
    sig = sma_crossover_signal(prices, 10, 30)
    assert set(np.unique(sig)).issubset({0.0, 1.0})
    assert len(sig) == len(prices)


def test_length_mismatch_raises():
    try:
        run_backtest([1, 2, 3], [1, 0])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
