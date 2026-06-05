"""
Vectorised Backtesting Engine
-----------------------------
A small, dependency-light backtester for single-asset strategies. It turns a
price series and a target-position series into an equity curve, then reports the
performance statistics every strategy write-up quotes: CAGR, volatility, Sharpe,
Sortino, max drawdown, Calmar, hit rate and turnover.

Design choices that keep the results honest:

* **No look-ahead.** The position you decide at bar ``t`` only earns the return
  from bar ``t`` to ``t+1``. Signals are shifted by one bar internally.
* **Costs are charged on traded notional.** Every change in position pays
  ``fee_bps`` (commission) plus ``slippage_bps`` on the absolute size of the
  trade, so flipping +1 → -1 costs twice as much as 0 → +1.
* **Everything is an array.** No hidden state, no loops over time for the core
  P&L — easy to read, easy to test.
"""

from __future__ import annotations

from typing import Union

import numpy as np

ArrayLike = Union[list, np.ndarray]


def to_returns(prices: ArrayLike) -> np.ndarray:
    """Simple period-over-period returns ``r_t = P_t / P_{t-1} - 1``.

    The first element is ``0.0`` so the returns array lines up with ``prices``.
    """
    p = np.asarray(prices, dtype=float)
    r = np.zeros_like(p)
    r[1:] = p[1:] / p[:-1] - 1.0
    return r


def run_backtest(
    prices: ArrayLike,
    target_position: ArrayLike,
    fee_bps: float = 1.0,
    slippage_bps: float = 0.5,
    initial_capital: float = 1.0,
) -> dict:
    """Run a single-asset backtest.

    Args:
        prices: Price series of the traded asset.
        target_position: Desired exposure at each bar, e.g. in ``{-1, 0, 1}`` or
            any continuous weight. Interpreted as the position *decided* at that
            bar; it is shifted forward one bar before earning a return.
        fee_bps: Commission per unit traded notional, in basis points.
        slippage_bps: Execution slippage per unit traded notional, in basis points.
        initial_capital: Starting equity (the equity curve is scaled to this).

    Returns:
        dict with ``equity`` (curve), ``returns`` (net strategy returns),
        ``gross_returns``, ``position`` (the lagged, actually-held position),
        ``costs`` and ``turnover`` arrays.
    """
    p = np.asarray(prices, dtype=float)
    target = np.asarray(target_position, dtype=float)
    if p.shape != target.shape:
        raise ValueError("prices and target_position must have the same length")

    asset_ret = to_returns(p)

    # Hold today what we decided yesterday — this is what removes look-ahead.
    held = np.zeros_like(target)
    held[1:] = target[:-1]

    # Trades happen when the held position changes versus the prior bar.
    traded = np.zeros_like(held)
    traded[1:] = np.abs(held[1:] - held[:-1])
    cost_rate = (fee_bps + slippage_bps) / 1e4
    costs = traded * cost_rate

    gross = held * asset_ret
    net = gross - costs

    equity = initial_capital * np.cumprod(1.0 + net)

    return {
        "equity": equity,
        "returns": net,
        "gross_returns": gross,
        "position": held,
        "costs": costs,
        "turnover": traded,
    }


def max_drawdown(equity: ArrayLike) -> dict:
    """Largest peak-to-trough decline of an equity curve.

    Returns the (negative) max drawdown plus the peak/trough indices.
    """
    e = np.asarray(equity, dtype=float)
    running_max = np.maximum.accumulate(e)
    drawdown = e / running_max - 1.0
    trough = int(np.argmin(drawdown))
    peak = int(np.argmax(e[: trough + 1])) if trough > 0 else 0
    return {
        "max_drawdown": float(drawdown.min()),
        "peak_index": peak,
        "trough_index": trough,
        "drawdown_series": drawdown,
    }


def performance_summary(result: dict, periods_per_year: int = 252) -> dict:
    """Headline statistics for a backtest result.

    Args:
        result: Output of :func:`run_backtest`.
        periods_per_year: Bars per year (252 daily, 52 weekly, 12 monthly).
    """
    net = np.asarray(result["returns"], dtype=float)
    equity = np.asarray(result["equity"], dtype=float)
    n = len(net)
    if n < 2:
        raise ValueError("need at least two bars to summarise")

    total_return = float(equity[-1] / equity[0] - 1.0)
    years = n / periods_per_year
    cagr = float(equity[-1] / equity[0]) ** (1.0 / years) - 1.0 if years > 0 else 0.0

    vol = float(net.std(ddof=1) * np.sqrt(periods_per_year))
    mean_ann = float(net.mean() * periods_per_year)
    sharpe = mean_ann / vol if vol > 0 else 0.0

    downside = net[net < 0]
    downside_dev = float(downside.std(ddof=1) * np.sqrt(periods_per_year)) if downside.size > 1 else 0.0
    sortino = mean_ann / downside_dev if downside_dev > 0 else 0.0

    dd = max_drawdown(equity)
    calmar = cagr / abs(dd["max_drawdown"]) if dd["max_drawdown"] < 0 else 0.0

    hit_rate = float(np.mean(net > 0))
    avg_turnover = float(np.mean(result["turnover"]))

    return {
        "total_return": total_return,
        "cagr": cagr,
        "ann_volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": dd["max_drawdown"],
        "calmar": calmar,
        "hit_rate": hit_rate,
        "avg_turnover": avg_turnover,
        "total_cost": float(np.sum(result["costs"])),
        "n_bars": n,
    }


def sma_crossover_signal(prices: ArrayLike, fast: int = 20, slow: int = 50) -> np.ndarray:
    """A textbook long/flat signal: long when fast SMA > slow SMA, else flat."""
    p = np.asarray(prices, dtype=float)

    def sma(x, w):
        out = np.full_like(x, np.nan)
        if len(x) >= w:
            c = np.cumsum(np.insert(x, 0, 0.0))
            out[w - 1 :] = (c[w:] - c[:-w]) / w
        return out

    f, s = sma(p, fast), sma(p, slow)
    sig = np.where(f > s, 1.0, 0.0)
    sig[np.isnan(f) | np.isnan(s)] = 0.0
    return sig


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    n = 1500
    # A drifting random walk with occasional regime shifts.
    drift = 0.0003 + 0.0006 * np.sin(np.linspace(0, 6 * np.pi, n))
    rets = drift + 0.01 * rng.standard_normal(n)
    prices = 100 * np.cumprod(1 + rets)

    signal = sma_crossover_signal(prices, fast=20, slow=50)
    res = run_backtest(prices, signal, fee_bps=1.0, slippage_bps=0.5)
    stats = performance_summary(res)

    bh = run_backtest(prices, np.ones(n), fee_bps=0.0, slippage_bps=0.0)
    bh_stats = performance_summary(bh)

    print("SMA(20/50) Crossover vs. Buy & Hold")
    print("=" * 44)
    print(f"{'metric':<16}{'strategy':>14}{'buy&hold':>14}")
    for key, label in [
        ("total_return", "Total return"),
        ("cagr", "CAGR"),
        ("ann_volatility", "Volatility"),
        ("sharpe", "Sharpe"),
        ("sortino", "Sortino"),
        ("max_drawdown", "Max drawdown"),
        ("calmar", "Calmar"),
        ("hit_rate", "Hit rate"),
    ]:
        print(f"{label:<16}{stats[key]:>14.3f}{bh_stats[key]:>14.3f}")
    print(f"\nTotal costs paid: {stats['total_cost']:.4f}  ·  avg turnover: {stats['avg_turnover']:.4f}")
