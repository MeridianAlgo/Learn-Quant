"""
Sharpe and Sortino Ratio Calculator
-----------------------------------
Author: Your Name
This script provides beginner-friendly functions to calculate the Sharpe Ratio and Sortino Ratio for a set of asset returns.
Both ratios are used widely in finance to evaluate the risk-adjusted performance of investments.
"""
import numpy as np

def sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """
    Calculate the Sharpe Ratio.
    Args:
        returns (list or np.array): Series of returns (e.g., daily returns)
        risk_free_rate (float): The risk-free rate (annualized)
        periods_per_year (int): How many periods in a year (252 for daily, 12 for monthly)
    Returns:
        float: Annualized Sharpe Ratio
    """
    returns = np.array(returns)
    excess_returns = returns - (risk_free_rate / periods_per_year)
    mean_excess_return = excess_returns.mean()
    std_dev = excess_returns.std()
    sharpe = (mean_excess_return / std_dev) * np.sqrt(periods_per_year)
    return sharpe

def sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """
    Calculate the Sortino Ratio.
    Args:
        returns (list or np.array): Series of returns (e.g., daily returns)
        risk_free_rate (float): The risk-free rate (annualized)
        periods_per_year (int): How many periods in a year (252 for daily, 12 for monthly)
    Returns:
        float: Annualized Sortino Ratio
    """
    returns = np.array(returns)
    excess_returns = returns - (risk_free_rate / periods_per_year)
    mean_excess_return = excess_returns.mean()
    downside_std = np.sqrt(((np.minimum(0, excess_returns))**2).mean())
    sortino = (mean_excess_return / downside_std) * np.sqrt(periods_per_year)
    return sortino

if __name__ == "__main__":
    # Example usage with fake data
    daily_returns = np.random.normal(0.0005, 0.01, 252)  # Simulate 1 year of daily returns
    sharpe = sharpe_ratio(daily_returns)
    sortino = sortino_ratio(daily_returns)
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Sortino Ratio: {sortino:.2f}")
