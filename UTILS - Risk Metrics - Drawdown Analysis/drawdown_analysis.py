"""
Drawdown Analysis
------------------
Comprehensive drawdown metrics for evaluating portfolio risk and performance.

- Max Drawdown: Largest peak-to-trough decline
- Calmar Ratio: Annualized return / Max Drawdown
- Ulcer Index: RMS of drawdown depths (penalizes prolonged drawdowns)
- Average Drawdown: Mean of all drawdown depths
- Max Drawdown Duration: Longest time spent in drawdown
"""

import numpy as np
from typing import Union


def drawdown_series(returns: Union[list, np.ndarray]) -> np.ndarray:
    """
    Compute the drawdown series (current drawdown at each point).

    Args:
        returns: Return series.

    Returns:
        np.ndarray: Drawdown series (non-positive values; 0 at new peaks).
    """
    returns = np.array(returns)
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    return cumulative / running_max - 1


def max_drawdown(returns: Union[list, np.ndarray]) -> float:
    """
    Maximum drawdown: largest peak-to-trough percentage decline.

    Returns:
        float: Max drawdown as a positive number (e.g., 0.25 = 25%).
    """
    dd = drawdown_series(returns)
    return float(-np.min(dd))


def calmar_ratio(returns: Union[list, np.ndarray], periods: int = 252) -> float:
    """
    Calmar ratio: annualized return divided by max drawdown.

    Args:
        returns: Return series.
        periods: Trading periods per year.

    Returns:
        float: Calmar ratio (higher is better).
    """
    returns = np.array(returns)
    ann_return = np.mean(returns) * periods
    mdd = max_drawdown(returns)
    if mdd == 0:
        return np.inf
    return float(ann_return / mdd)


def ulcer_index(returns: Union[list, np.ndarray]) -> float:
    """
    Ulcer Index: RMS of drawdown depths.
    Penalizes both depth and duration of drawdowns.

    Returns:
        float: Ulcer index (lower is better).
    """
    dd = drawdown_series(returns)
    return float(np.sqrt(np.mean(dd**2)))


def ulcer_performance_index(
    returns: Union[list, np.ndarray],
    risk_free: float = 0.0,
) -> float:
    """
    Ulcer Performance Index (Martin Ratio): mean excess return / ulcer index.

    Returns:
        float: UPI (higher is better).
    """
    returns = np.array(returns)
    excess = np.mean(returns) - risk_free / 252
    ui = ulcer_index(returns)
    if ui == 0:
        return np.inf
    return float(excess / ui)


def average_drawdown(returns: Union[list, np.ndarray]) -> float:
    """
    Average depth of all drawdown periods.

    Returns:
        float: Mean drawdown depth (positive number).
    """
    dd = drawdown_series(returns)
    depths = dd[dd < 0]
    if len(depths) == 0:
        return 0.0
    return float(-np.mean(depths))


def max_drawdown_duration(returns: Union[list, np.ndarray]) -> int:
    """
    Maximum number of periods spent continuously in drawdown.

    Returns:
        int: Maximum drawdown duration in periods.
    """
    dd = drawdown_series(returns)
    in_drawdown = dd < 0
    max_dur = 0
    current_dur = 0
    for is_dd in in_drawdown:
        if is_dd:
            current_dur += 1
            max_dur = max(max_dur, current_dur)
        else:
            current_dur = 0
    return max_dur


def drawdown_summary(returns: Union[list, np.ndarray], periods: int = 252) -> dict:
    """
    Full drawdown analytics summary.

    Returns:
        dict: All drawdown metrics.
    """
    return {
        "max_drawdown": max_drawdown(returns),
        "calmar_ratio": calmar_ratio(returns, periods),
        "ulcer_index": ulcer_index(returns),
        "ulcer_performance_index": ulcer_performance_index(returns),
        "average_drawdown": average_drawdown(returns),
        "max_drawdown_duration_periods": max_drawdown_duration(returns),
    }


if __name__ == "__main__":
    np.random.seed(42)
    bull = np.random.normal(0.001, 0.01, 200)
    bear = np.random.normal(-0.002, 0.02, 100)
    recovery = np.random.normal(0.0015, 0.01, 152)
    returns = np.concatenate([bull, bear, recovery])

    print("Drawdown Analysis")
    print("=" * 40)
    summary = drawdown_summary(returns)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k.replace('_', ' ').title():40s}: {v:.4f}")
        else:
            print(f"{k.replace('_', ' ').title():40s}: {v}")
