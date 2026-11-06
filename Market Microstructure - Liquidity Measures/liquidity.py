"""
Liquidity Measures
------------------
Liquidity is the cost of getting in and out of a position, and nobody hands
you that number directly. Quote data with a real bid and ask is expensive and
often missing for older periods or smaller names. So a whole family of
estimators exists to back out trading costs from what is cheap and always
available: closing prices, daily highs and lows, and volume.

This file implements the ones that come up most.

* Roll (1984) reads the spread from the bounce between bid and ask.
* Corwin and Schultz (2012) read it from daily high and low prices.
* Amihud (2002) measures how far price moves per dollar traded.
* Kyle's lambda measures how far price moves per unit of signed order flow.
* Effective and realised spreads measure cost on actual trades when you do
  have quotes.
"""

from __future__ import annotations

import math

import numpy as np


def _as_array(values, name: str, min_len: int = 2) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size < min_len:
        raise ValueError(f"{name} needs at least {min_len} values")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def roll_spread(prices) -> float:
    """Roll's implied bid ask spread from a series of trade prices.

    When trades bounce between bid and ask, consecutive price changes are
    negatively correlated, and ``cov(dp_t, dp_t-1) = -s^2 / 4``. Solving for
    ``s`` gives the spread in price units.

    If the autocovariance is positive the model has nothing to say, because
    the square root of a negative number is not a spread. That happens often
    on daily data dominated by momentum, and the honest answer is NaN.
    """
    p = _as_array(prices, "prices", min_len=4)
    dp = np.diff(p)
    autocov = np.cov(dp[1:], dp[:-1])[0, 1]
    if autocov >= 0:
        return math.nan
    return 2.0 * math.sqrt(-autocov)


def corwin_schultz_spread(high, low) -> float:
    """Corwin and Schultz high low spread estimator, averaged over the sample.

    The daily high is usually a buy at the ask and the low a sell at the bid,
    so the high low range contains both volatility and the spread. Volatility
    grows with the length of the window and the spread does not, so comparing
    one day ranges with two day ranges separates the two.

    Negative daily estimates are set to zero, as in the original paper.
    """
    h = _as_array(high, "high")
    lo = _as_array(low, "low")
    if h.size != lo.size:
        raise ValueError("high and low must be the same length")
    if np.any(lo <= 0) or np.any(h < lo):
        raise ValueError("need positive lows and high >= low on every day")

    k = 3.0 - 2.0 * math.sqrt(2.0)
    log_range_sq = np.log(h / lo) ** 2
    beta = log_range_sq[1:] + log_range_sq[:-1]
    two_day_high = np.maximum(h[1:], h[:-1])
    two_day_low = np.minimum(lo[1:], lo[:-1])
    gamma = np.log(two_day_high / two_day_low) ** 2

    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return float(np.clip(spread, 0.0, None).mean())


def amihud_illiquidity(returns, dollar_volume, scale: float = 1e6) -> float:
    """Average absolute return per dollar traded, times ``scale``.

    A stock that moves 2% on one million dollars of volume is far less liquid
    than one that moves 2% on a hundred million. The default scale reports the
    move per million dollars so the numbers are readable.
    """
    r = _as_array(returns, "returns", min_len=1)
    dv = _as_array(dollar_volume, "dollar_volume", min_len=1)
    if r.size != dv.size:
        raise ValueError("returns and dollar_volume must be the same length")
    traded = dv > 0
    if not traded.any():
        raise ValueError("no days with positive dollar volume")
    return float(np.mean(np.abs(r[traded]) / dv[traded]) * scale)


def kyle_lambda(price_changes, signed_volume) -> float:
    """Price impact per unit of signed order flow, from an OLS slope.

    Regress price change on net buying volume. The slope is Kyle's lambda,
    how much the market maker moves the price to protect against informed
    traders. Larger lambda means a thinner, less liquid market.
    """
    dp = _as_array(price_changes, "price_changes", min_len=3)
    q = _as_array(signed_volume, "signed_volume", min_len=3)
    if dp.size != q.size:
        raise ValueError("price_changes and signed_volume must be the same length")
    q_centered = q - q.mean()
    denom = float(q_centered @ q_centered)
    if denom == 0:
        raise ValueError("signed volume has no variation")
    return float(q_centered @ (dp - dp.mean()) / denom)


def effective_spread(trade_prices, midquotes, sides) -> np.ndarray:
    """Effective spread of each trade as a fraction of the midquote.

    ``2 * side * (price - mid) / mid`` with side ``+1`` for a buy and ``-1``
    for a sell. This is what the trade actually cost against fair value.
    """
    p = _as_array(trade_prices, "trade_prices", min_len=1)
    m = _as_array(midquotes, "midquotes", min_len=1)
    d = _as_array(sides, "sides", min_len=1)
    if not (p.size == m.size == d.size):
        raise ValueError("inputs must be the same length")
    if not np.all(np.isin(d, (-1.0, 1.0))):
        raise ValueError("sides must be +1 (buy) or -1 (sell)")
    if np.any(m <= 0):
        raise ValueError("midquotes must be positive")
    return 2.0 * d * (p - m) / m


def realized_spread(trade_prices, future_midquotes, sides) -> np.ndarray:
    """Realised spread, the part of the effective spread the liquidity provider keeps.

    Same formula as the effective spread but measured against the midquote a
    few minutes later. The gap between the two is price impact, the part the
    provider lost to traders who knew something.
    """
    return effective_spread(trade_prices, future_midquotes, sides)


def _simulate_bounce(n: int, spread: float, vol: float, seed: int) -> np.ndarray:
    """Trade prices bouncing between bid and ask around a random walk."""
    rng = np.random.default_rng(seed)
    mid = 100.0 + np.cumsum(rng.normal(0.0, vol, n))
    side = rng.choice([-1.0, 1.0], size=n)
    return mid + 0.5 * spread * side


if __name__ == "__main__":
    true_spread = 0.05
    prices = _simulate_bounce(20_000, true_spread, 0.01, seed=3)
    print(f"True spread {true_spread:.4f}   Roll estimate {roll_spread(prices):.4f}")

    rng = np.random.default_rng(5)
    days = 250
    close = 50.0 * np.exp(np.cumsum(rng.normal(0, 0.015, days)))
    half = 0.002
    intraday = np.abs(rng.normal(0, 0.01, days))
    high = close * (1 + intraday + half)
    low = close * (1 - intraday - half)
    print(f"Corwin Schultz spread {corwin_schultz_spread(high, low):.4%}")

    rets = rng.normal(0, 0.02, days)
    liquid = amihud_illiquidity(rets, np.full(days, 5e8))
    thin = amihud_illiquidity(rets, np.full(days, 2e6))
    print(f"Amihud  liquid name {liquid:.5f}   thin name {thin:.5f}")

    flow = rng.normal(0, 1000, 500)
    moves = 0.0004 * flow + rng.normal(0, 0.2, 500)
    print(f"Kyle lambda (true 0.0004) {kyle_lambda(moves, flow):.6f}")

    eff = effective_spread([100.03, 99.98], [100.0, 100.0], [1, -1])
    print("Effective spreads (bps):", np.round(eff * 1e4, 2))
