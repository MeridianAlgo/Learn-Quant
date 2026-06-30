"""
Calmar Ratio
------------
The Sharpe ratio judges a strategy by how much its returns wobble. But an
investor does not lose sleep over wobble, they lose sleep over drawdown, the
gut punch of watching an account fall from its high water mark. The Calmar ratio
speaks to that fear directly. It divides the annual growth rate by the worst
peak to trough loss, so a strategy that compounds nicely but once halved your
capital scores poorly no matter how smooth the rest looked.

This lesson computes the pieces from a return series, the compound annual growth
rate, the maximum drawdown, and then the Calmar and the closely related MAR
ratio. The maths is plain arithmetic on an equity curve, which makes it a good
companion to the Sharpe and Sortino lessons that measure risk a different way.
"""

from __future__ import annotations

import numpy as np


def equity_curve(returns: list[float], start: float = 1.0) -> np.ndarray:
    """Turn a series of period returns into a growing equity curve.

    Each point is the running product of one plus each return, so a value of 1.2
    means the capital has grown twenty percent since the start.
    """
    r = np.asarray(returns, dtype=float)
    return start * np.cumprod(1 + r)


def max_drawdown(returns: list[float]) -> float:
    """Return the worst peak to trough decline as a positive fraction.

    It tracks the running high water mark and measures how far below it the
    curve ever fell. A result of 0.25 means the strategy lost a quarter of its
    value from a peak at the deepest point. A flat or rising curve returns zero.
    """
    curve = equity_curve(returns)
    running_peak = np.maximum.accumulate(curve)
    drawdowns = (curve - running_peak) / running_peak
    return float(-drawdowns.min()) + 0.0  # the plus zero avoids a negative zero result


def cagr(returns: list[float], periods_per_year: int = 252) -> float:
    """Return the compound annual growth rate of the return series.

    This is the single smooth yearly rate that would take you from the start to
    the end of the equity curve. Daily data uses 252 trading days a year, monthly
    data uses 12. It answers what steady annual return the track record earned.
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    if n == 0:
        raise ValueError("need at least one return")
    total_growth = float(np.prod(1 + r))
    if total_growth <= 0:
        return -1.0  # the strategy was fully wiped out
    years = n / periods_per_year
    return total_growth ** (1 / years) - 1


def calmar_ratio(returns: list[float], periods_per_year: int = 252) -> float:
    """Return the Calmar ratio, annual growth divided by maximum drawdown.

    A higher number means more compounding earned per unit of worst case pain.
    Above three is often considered excellent, though the figure depends heavily
    on the window since one bad year can dominate the drawdown for a long time.
    With no drawdown at all the ratio is infinite, reported here as positive
    infinity.
    """
    mdd = max_drawdown(returns)
    growth = cagr(returns, periods_per_year)
    if mdd == 0:
        return float("inf") if growth > 0 else 0.0
    return growth / mdd


def mar_ratio(returns: list[float], periods_per_year: int = 252) -> float:
    """Return the MAR ratio, the same idea over the whole track record.

    MAR divides the compound annual growth rate by the maximum drawdown observed
    across the entire history rather than a recent window. In this self contained
    form it shares the formula with Calmar, the difference in practice is only
    the length of the window the two are measured over.
    """
    return calmar_ratio(returns, periods_per_year)


if __name__ == "__main__":
    print("Calmar Ratio")
    print("=" * 40)

    # Two years of daily returns from a steady but occasionally bruised strategy.
    rng = np.random.default_rng(7)
    good_days = rng.normal(0.0006, 0.01, size=500)
    # Splice in a rough patch so there is a real drawdown to measure.
    crash = np.full(20, -0.015)
    returns = np.concatenate([good_days[:250], crash, good_days[250:]]).tolist()

    growth = cagr(returns)
    mdd = max_drawdown(returns)
    print(f"\nCompound annual growth rate  {growth:.2%}")
    print(f"Maximum drawdown             {mdd:.2%}")
    print(f"Calmar ratio                 {calmar_ratio(returns):.2f}")

    print("\nA smoother curve scores higher")
    calm = rng.normal(0.0005, 0.004, size=500).tolist()
    print(f"  calm strategy  drawdown {max_drawdown(calm):.2%}  calmar {calmar_ratio(calm):.2f}")

    print("\nMaximum drawdown of a curve that only rises is zero")
    rising = [0.001] * 100
    print(f"  drawdown {max_drawdown(rising):.2%}  calmar {calmar_ratio(rising)}")
