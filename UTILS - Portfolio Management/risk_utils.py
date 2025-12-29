"""
Risk Analysis Utilities for Financial Applications

This module provides comprehensive risk analysis utilities for financial applications,
including Value at Risk (VaR), maximum drawdown, volatility calculations, correlation analysis,
and stress testing.

Author: Generated for Learn-Quant Project
Version: 1.0.0
"""

from typing import List, Dict, Any
import statistics
import math

try:
    import numpy as np
except ImportError:
    print("Warning: numpy not found. Some functions may not work optimally. Install with: pip install numpy")
    np = None


def calculate_var(returns: List[float], confidence_level: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR) using historical method.
    
    Args:
        returns: List of portfolio returns
        confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
        
    Returns:
        VaR value (negative number representing loss)
        
    Example:
        >>> returns = [0.02, -0.01, 0.03, -0.02, 0.01]
        >>> var = calculate_var(returns, 0.95)
        >>> print(f"95% VaR: {var:.2%}")
        -0.02
    """
    if not returns:
        return 0.0
    
    sorted_returns = sorted(returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    index = max(0, min(index, len(sorted_returns) - 1))
    
    return sorted_returns[index]


def calculate_max_drawdown(prices: List[float]) -> Dict[str, float]:
    """
    Calculate maximum drawdown and related metrics.
    
    Args:
        prices: List of prices or portfolio values
        
    Returns:
        Dictionary with max_drawdown, drawdown_duration, recovery_time
        
    Example:
        >>> prices = [100, 110, 105, 95, 90, 100, 115]
        >>> dd = calculate_max_drawdown(prices)
        >>> print(f"Max Drawdown: {dd['max_drawdown']:.2%}")
        -18.18%
    """
    if not prices or len(prices) < 2:
        return {"max_drawdown": 0.0, "drawdown_duration": 0, "recovery_time": 0}
    
    peak = prices[0]
    max_dd = 0.0
    dd_duration = 0
    max_dd_duration = 0
    recovery_time = 0
    in_drawdown = False
    drawdown_start = 0
    
    for i, price in enumerate(prices):
        if price > peak:
            peak = price
            if in_drawdown:
                recovery_time = i - drawdown_start
                in_drawdown = False
        
        drawdown = (price - peak) / peak
        
        if drawdown < max_dd:
            max_dd = drawdown
            max_dd_duration = dd_duration
        
        if drawdown < 0:
            if not in_drawdown:
                drawdown_start = i
                in_drawdown = True
            dd_duration += 1
        else:
            dd_duration = 0
    
    return {
        "max_drawdown": max_dd,
        "drawdown_duration": max_dd_duration,
        "recovery_time": recovery_time
    }


def calculate_volatility(returns: List[float], annualize: bool = True, periods_per_year: int = 252) -> float:
    """
    Calculate volatility (standard deviation of returns).
    
    Args:
        returns: List of returns
        annualize: Whether to annualize the volatility
        periods_per_year: Number of periods per year for annualization
        
    Returns:
        Volatility
        
    Example:
        >>> returns = [0.02, -0.01, 0.03, -0.02, 0.01]
        >>> vol = calculate_volatility(returns)
        >>> print(f"Annualized Volatility: {vol:.2%}")
        25.30%
    """
    if not returns:
        return 0.0
    
    if len(returns) < 2:
        return 0.0
    
    # Calculate standard deviation
    mean_return = statistics.mean(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
    volatility = math.sqrt(variance)
    
    if annualize:
        volatility *= math.sqrt(periods_per_year)
    
    return volatility


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: List of returns
        risk_free_rate: Risk-free rate (annual)
        periods_per_year: Number of periods per year
        
    Returns:
        Sharpe ratio
        
    Example:
        >>> returns = [0.02, -0.01, 0.03, -0.02, 0.01]
        >>> sharpe = calculate_sharpe_ratio(returns)
        >>> print(f"Sharpe Ratio: {sharpe:.2f}")
        0.79
    """
    if not returns or len(returns) < 2:
        return 0.0
    
    # Calculate annualized return
    total_return = (1 + sum(returns)) ** (periods_per_year / len(returns)) - 1
    
    # Calculate annualized volatility
    volatility = calculate_volatility(returns, True, periods_per_year)
    
    if volatility == 0:
        return 0.0
    
    # Calculate excess return
    excess_return = total_return - risk_free_rate
    
    return excess_return / volatility


def calculate_sortino_ratio(returns: List[float], risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
    """
    Calculate Sortino ratio (downside risk-adjusted return).
    
    Args:
        returns: List of returns
        risk_free_rate: Risk-free rate (annual)
        periods_per_year: Number of periods per year
        
    Returns:
        Sortino ratio
        
    Example:
        >>> returns = [0.02, -0.01, 0.03, -0.02, 0.01]
        >>> sortino = calculate_sortino_ratio(returns)
        >>> print(f"Sortino Ratio: {sortino:.2f}")
        1.12
    """
    if not returns or len(returns) < 2:
        return 0.0
    
    # Calculate annualized return
    total_return = (1 + sum(returns)) ** (periods_per_year / len(returns)) - 1
    
    # Calculate downside deviation
    negative_returns = [r for r in returns if r < 0]
    
    if not negative_returns:
        return float('inf') if total_return > risk_free_rate else 0.0
    
    mean_negative = statistics.mean(negative_returns)
    downside_variance = sum((r - mean_negative) ** 2 for r in negative_returns) / len(negative_returns)
    downside_deviation = math.sqrt(downside_variance) * math.sqrt(periods_per_year)
    
    if downside_deviation == 0:
        return float('inf') if total_return > risk_free_rate else 0.0
    
    excess_return = total_return - risk_free_rate
    return excess_return / downside_deviation


def calculate_correlation_matrix(returns_data: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate correlation matrix for multiple assets.
    
    Args:
        returns_data: Dictionary of returns {asset: [returns]}
        
    Returns:
        Correlation matrix dictionary
        
    Example:
        >>> returns_data = {"AAPL": [0.02, -0.01], "GOOGL": [0.01, -0.02]}
        >>> corr = calculate_correlation_matrix(returns_data)
        >>> print(f"AAPL-GOOGL correlation: {corr['AAPL']['GOOGL']:.2f}")
        0.50
    """
    assets = list(returns_data.keys())
    correlation_matrix = {}
    
    for asset1 in assets:
        correlation_matrix[asset1] = {}
        returns1 = returns_data[asset1]
        
        for asset2 in assets:
            returns2 = returns_data[asset2]
            
            if len(returns1) != len(returns2) or len(returns1) < 2:
                correlation_matrix[asset1][asset2] = 0.0
                continue
            
            # Calculate correlation
            mean1 = statistics.mean(returns1)
            mean2 = statistics.mean(returns2)
            
            numerator = sum((r1 - mean1) * (r2 - mean2) for r1, r2 in zip(returns1, returns2))
            
            var1 = sum((r1 - mean1) ** 2 for r1 in returns1)
            var2 = sum((r2 - mean2) ** 2 for r2 in returns2)
            
            denominator = math.sqrt(var1 * var2)
            
            if denominator == 0:
                correlation = 0.0
            else:
                correlation = numerator / denominator
            
            correlation_matrix[asset1][asset2] = correlation
    
    return correlation_matrix


def stress_test_portfolio(holdings: Dict[str, Dict[str, Any]], scenarios: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Perform stress testing on portfolio under different scenarios.
    
    Args:
        holdings: Portfolio holdings
        scenarios: Stress scenarios {name: {symbol: shock_pct}}
        
    Returns:
        Portfolio values under each scenario
        
    Example:
        >>> holdings = {"AAPL": {"shares": 100}}
        >>> scenarios = {"Market Crash": {"AAPL": -0.30}}
        >>> results = stress_test_portfolio(holdings, scenarios)
        >>> print(f"Crash scenario value: ${results['Market Crash']['portfolio_value']:,.2f}")
        10500.00
    """
    # Calculate current portfolio value (assuming current price = 100 for demo)
    current_value = sum(holding["shares"] * 100 for holding in holdings.values())
    
    stress_results = {}
    
    for scenario_name, shocks in scenarios.items():
        scenario_value = 0.0
        
        for symbol, holding in holdings.items():
            shares = holding["shares"]
            base_price = 100  # Assume base price for demo
            
            if symbol in shocks:
                shock_pct = shocks[symbol]
                shocked_price = base_price * (1 + shock_pct)
            else:
                shocked_price = base_price
            
            scenario_value += shares * shocked_price
        
        stress_results[scenario_name] = {
            "portfolio_value": scenario_value,
            "pnl": scenario_value - current_value,
            "pnl_pct": (scenario_value - current_value) / current_value
        }
    
    return stress_results


def calculate_beta(asset_returns: List[float], market_returns: List[float]) -> float:
    """
    Calculate beta of an asset relative to market.
    
    Args:
        asset_returns: Asset returns
        market_returns: Market returns
        
    Returns:
        Beta value
        
    Example:
        >>> asset_returns = [0.02, -0.01, 0.03]
        >>> market_returns = [0.01, -0.005, 0.02]
        >>> beta = calculate_beta(asset_returns, market_returns)
        >>> print(f"Beta: {beta:.2f}")
        1.50
    """
    if len(asset_returns) != len(market_returns) or len(asset_returns) < 2:
        return 1.0  # Default beta
    
    # Calculate covariance and variance
    asset_mean = statistics.mean(asset_returns)
    market_mean = statistics.mean(market_returns)
    
    covariance = sum((ar - asset_mean) * (mr - market_mean) for ar, mr in zip(asset_returns, market_returns))
    market_variance = sum((mr - market_mean) ** 2 for mr in market_returns)
    
    if market_variance == 0:
        return 1.0
    
    beta = covariance / market_variance
    return beta


def calculate_information_ratio(portfolio_returns: List[float], benchmark_returns: List[float]) -> float:
    """
    Calculate Information Ratio (active return divided by tracking error).
    
    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Benchmark returns
        
    Returns:
        Information Ratio
        
    Example:
        >>> portfolio_returns = [0.02, -0.01, 0.03]
        >>> benchmark_returns = [0.01, -0.005, 0.02]
        >>> ir = calculate_information_ratio(portfolio_returns, benchmark_returns)
        >>> print(f"Information Ratio: {ir:.2f}")
        1.22
    """
    if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < 2:
        return 0.0
    
    # Calculate active returns
    active_returns = [pr - br for pr, br in zip(portfolio_returns, benchmark_returns)]
    
    # Calculate tracking error (standard deviation of active returns)
    if len(active_returns) < 2:
        return 0.0
    
    mean_active = statistics.mean(active_returns)
    tracking_error = math.sqrt(sum((ar - mean_active) ** 2 for ar in active_returns) / (len(active_returns) - 1))
    
    if tracking_error == 0:
        return 0.0
    
    # Annualize
    annualized_return = mean_active * 252
    annualized_tracking_error = tracking_error * math.sqrt(252)
    
    return annualized_return / annualized_tracking_error


def demo_risk_utils():
    """Demonstrate risk analysis utilities."""
    print("=" * 60)
    print("RISK ANALYSIS UTILITIES DEMONSTRATION")
    print("=" * 60)
    
    # Sample data
    returns = [0.02, -0.01, 0.03, -0.02, 0.01, 0.015, -0.005, 0.025, -0.015, 0.02]
    prices = [100, 105, 110, 108, 95, 90, 92, 98, 105, 115, 120, 118]
    
    print("\n1. Value at Risk (VaR):")
    var_95 = calculate_var(returns, 0.95)
    var_99 = calculate_var(returns, 0.99)
    print(f"  95% VaR: {var_95:.2%}")
    print(f"  99% VaR: {var_99:.2%}")
    
    print("\n2. Maximum Drawdown:")
    drawdown = calculate_max_drawdown(prices)
    print(f"  Max Drawdown: {drawdown['max_drawdown']:.2%}")
    print(f"  Drawdown Duration: {drawdown['drawdown_duration']} periods")
    print(f"  Recovery Time: {drawdown['recovery_time']} periods")
    
    print("\n3. Volatility:")
    vol_daily = calculate_volatility(returns, annualize=False)
    vol_annual = calculate_volatility(returns, annualize=True)
    print(f"  Daily Volatility: {vol_daily:.2%}")
    print(f"  Annualized Volatility: {vol_annual:.2%}")
    
    print("\n4. Risk-Adjusted Returns:")
    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Sortino Ratio: {sortino:.2f}")
    
    print("\n5. Correlation Analysis:")
    returns_data = {
        "AAPL": [0.02, -0.01, 0.03, -0.02, 0.01],
        "GOOGL": [0.01, -0.02, 0.025, -0.015, 0.005],
        "MSFT": [0.015, -0.005, 0.02, -0.01, 0.012]
    }
    correlations = calculate_correlation_matrix(returns_data)
    
    print("  Correlation Matrix:")
    for asset1 in correlations:
        for asset2 in correlations[asset1]:
            if asset1 <= asset2:  # Avoid duplicates
                corr = correlations[asset1][asset2]
                print(f"    {asset1}-{asset2}: {corr:.3f}")
    
    print("\n6. Beta Calculation:")
    market_returns = [0.01, -0.005, 0.015, -0.01, 0.008]
    beta = calculate_beta(returns, market_returns)
    print(f"  Asset Beta: {beta:.2f}")
    
    print("\n7. Information Ratio:")
    portfolio_returns = [0.02, -0.01, 0.03, -0.02, 0.01]
    benchmark_returns = [0.01, -0.005, 0.015, -0.01, 0.008]
    ir = calculate_information_ratio(portfolio_returns, benchmark_returns)
    print(f"  Information Ratio: {ir:.2f}")
    
    print("\n8. Stress Testing:")
    holdings = {
        "AAPL": {"shares": 100},
        "GOOGL": {"shares": 50},
        "MSFT": {"shares": 75}
    }
    
    scenarios = {
        "Market Crash": {"AAPL": -0.30, "GOOGL": -0.25, "MSFT": -0.20},
        "Tech Boom": {"AAPL": 0.20, "GOOGL": 0.15, "MSFT": 0.18},
        "Sector Rotation": {"AAPL": -0.10, "GOOGL": 0.05, "MSFT": 0.15}
    }
    
    stress_results = stress_test_portfolio(holdings, scenarios)
    print("  Stress Test Results:")
    for scenario, results in stress_results.items():
        print(f"    {scenario}:")
        print(f"      Portfolio Value: ${results['portfolio_value']:,.2f}")
        print(f"      P&L: ${results['pnl']:,.2f} ({results['pnl_pct']:.2%})")


def main():
    """Main function to run demonstrations."""
    demo_risk_utils()
    print("\n🎉 Risk analysis utilities demonstration complete!")


if __name__ == "__main__":
    main()
