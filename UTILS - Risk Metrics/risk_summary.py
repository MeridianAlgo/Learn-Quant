"""
Risk Metrics Summary Utility
----------------------------
Analyze risk in any return series: get stats used by pros and exam-takers alike!
"""

import numpy as np


def risk_metrics(returns):
    """
    Calculate common risk metrics for a given series of returns.
    Args:
        returns (list or np.array): Series of returns
    Returns:
        dict: Summary of risk statistics
    """
    returns = np.array(returns)
    volatility = returns.std()
    downside_std = np.sqrt(np.mean(np.minimum(0, returns) ** 2))
    cum_returns = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    skew = ((returns - returns.mean()) ** 3).mean() / returns.std() ** 3
    kurtosis = ((returns - returns.mean()) ** 4).mean() / returns.std() ** 4
    return {
        "volatility": volatility,
        "downside_volatility": downside_std,
        "max_drawdown": max_drawdown,
        "skew": skew,
        "kurtosis": kurtosis,
    }


if __name__ == "__main__":
    # Example: Summarize risk in a fake return series
    np.random.seed(0)
    daily = np.random.normal(0.0005, 0.01, 252)
    stats = risk_metrics(daily)
    for k, v in stats.items():
        print(f"{k}: {v:.4f}")
