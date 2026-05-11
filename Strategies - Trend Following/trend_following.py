"""
Trend Following Strategy Signals
---------------------------------
Classic trend-following building blocks used by CTAs and managed futures funds.

Implements:
  - Donchian channel breakout (Turtle Traders)
  - MA crossover signals
  - Time-series momentum (TSMOM)
  - ATR-based position sizing & stop placement
"""

from typing import Union

import numpy as np


def donchian_channel(
    high: Union[list, np.ndarray],
    low: Union[list, np.ndarray],
    window: int = 20,
) -> dict:
    """
    Donchian channel: rolling N-period highest high and lowest low.

    Args:
        high: High price series.
        low: Low price series.
        window: Lookback period.

    Returns:
        dict: upper, lower, middle channels.
    """
    h = np.array(high, dtype=float)
    l_ = np.array(low, dtype=float)
    n = len(h)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(window - 1, n):
        upper[i] = float(np.max(h[i - window + 1 : i + 1]))
        lower[i] = float(np.min(l_[i - window + 1 : i + 1]))
    middle = (upper + lower) / 2.0
    return {"upper": upper, "lower": lower, "middle": middle}


def donchian_breakout_signal(
    close: Union[list, np.ndarray],
    high: Union[list, np.ndarray],
    low: Union[list, np.ndarray],
    entry_window: int = 20,
    exit_window: int = 10,
) -> np.ndarray:
    """
    Turtle-style breakout: long when close > prior N-day high,
    flat when close < prior M-day low.

    Args:
        close: Close prices.
        high: Highs.
        low: Lows.
        entry_window: Long-entry breakout window.
        exit_window: Long-exit breakdown window.

    Returns:
        np.ndarray: Position (0 or 1).
    """
    c = np.array(close, dtype=float)
    h = np.array(high, dtype=float)
    l_ = np.array(low, dtype=float)
    n = len(c)
    pos = np.zeros(n)
    for i in range(max(entry_window, exit_window), n):
        prior_high = float(np.max(h[i - entry_window : i]))
        prior_low = float(np.min(l_[i - exit_window : i]))
        if pos[i - 1] == 0 and c[i] > prior_high:
            pos[i] = 1
        elif pos[i - 1] == 1 and c[i] < prior_low:
            pos[i] = 0
        else:
            pos[i] = pos[i - 1]
    return pos


def ma_crossover_signal(
    prices: Union[list, np.ndarray],
    fast: int = 20,
    slow: int = 50,
) -> np.ndarray:
    """
    Moving-average crossover: long when fast MA > slow MA, short otherwise.

    Returns:
        np.ndarray: +1, 0 (warmup), -1.
    """
    p = np.array(prices, dtype=float)
    n = len(p)
    sig = np.zeros(n)
    for i in range(slow - 1, n):
        fast_ma = float(np.mean(p[i - fast + 1 : i + 1]))
        slow_ma = float(np.mean(p[i - slow + 1 : i + 1]))
        sig[i] = 1.0 if fast_ma > slow_ma else -1.0
    return sig


def tsmom_signal(
    prices: Union[list, np.ndarray],
    lookback: int = 252,
) -> np.ndarray:
    """
    Time-Series Momentum: long if past N-period return > 0, short otherwise.
    Reference: Moskowitz, Ooi, Pedersen (2012).
    """
    p = np.array(prices, dtype=float)
    n = len(p)
    sig = np.zeros(n)
    for i in range(lookback, n):
        sig[i] = 1.0 if p[i] > p[i - lookback] else -1.0
    return sig


def atr(
    high: Union[list, np.ndarray],
    low: Union[list, np.ndarray],
    close: Union[list, np.ndarray],
    window: int = 14,
) -> np.ndarray:
    """
    Average True Range (Wilder smoothing).
    """
    h = np.array(high, dtype=float)
    l_ = np.array(low, dtype=float)
    c = np.array(close, dtype=float)
    n = len(c)
    tr = np.empty(n)
    tr[0] = h[0] - l_[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l_[i], abs(h[i] - c[i - 1]), abs(l_[i] - c[i - 1]))
    out = np.full(n, np.nan)
    if n >= window:
        out[window - 1] = float(np.mean(tr[:window]))
        for i in range(window, n):
            out[i] = (out[i - 1] * (window - 1) + tr[i]) / window
    return out


def atr_position_size(
    capital: float,
    risk_per_trade: float,
    atr_value: float,
    stop_atr_multiple: float = 2.0,
) -> float:
    """
    Volatility-targeted position size.

    contracts = (capital * risk_per_trade) / (atr * stop_multiple)

    Args:
        capital: Total capital.
        risk_per_trade: Fraction risked per trade (e.g., 0.01 = 1%).
        atr_value: Current ATR (in price units).
        stop_atr_multiple: Stop distance in ATRs.

    Returns:
        float: Contract/share count (continuous, floor at caller).
    """
    if atr_value <= 0 or stop_atr_multiple <= 0:
        return 0.0
    return float(capital * risk_per_trade / (atr_value * stop_atr_multiple))


def trend_strength(prices: Union[list, np.ndarray], window: int = 50) -> np.ndarray:
    """
    Trend strength indicator: slope of OLS regression of log price over window,
    normalized to annualized return units (assuming daily data).
    """
    p = np.log(np.array(prices, dtype=float))
    n = len(p)
    out = np.full(n, np.nan)
    x = np.arange(window, dtype=float)
    x_mean = float(np.mean(x))
    sxx = float(np.sum((x - x_mean) ** 2))
    for i in range(window - 1, n):
        y = p[i - window + 1 : i + 1]
        y_mean = float(np.mean(y))
        sxy = float(np.sum((x - x_mean) * (y - y_mean)))
        slope = sxy / sxx
        out[i] = slope * 252.0
    return out


if __name__ == "__main__":
    np.random.seed(42)
    n = 500
    drift = np.linspace(0, 0.4, n)
    noise = np.random.normal(0, 0.01, n).cumsum()
    prices = 100 * np.exp(drift + noise)
    highs = prices * 1.005
    lows = prices * 0.995

    print("Trend Following Signals")
    print("=" * 40)
    bo = donchian_breakout_signal(prices, highs, lows, 20, 10)
    print(f"Donchian (last 5):  {bo[-5:]}")

    ma = ma_crossover_signal(prices, 20, 50)
    print(f"MA crossover (last 5): {ma[-5:]}")

    ts = tsmom_signal(prices, 252)
    print(f"TSMOM (last 5): {ts[-5:]}")

    a = atr(highs, lows, prices, 14)
    size = atr_position_size(100_000, 0.01, a[-1], 2.0)
    print(f"ATR={a[-1]:.4f}, size={size:.2f} units")

    strength = trend_strength(prices, 50)
    print(f"Trend strength (annualized log return): {strength[-1]:.2%}")
