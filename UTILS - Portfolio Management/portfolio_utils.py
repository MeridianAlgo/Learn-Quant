"""
Portfolio Management Utilities for Financial Applications

This module provides comprehensive portfolio management utilities for financial applications,
including portfolio valuation, allocation analysis, rebalancing, performance tracking,
and diversification metrics.

Author: Generated for Learn-Quant Project
Version: 1.0.0
"""

from typing import Any, Dict


def calculate_portfolio_value(
    holdings: Dict[str, Dict[str, Any]], prices: Dict[str, float]
) -> float:
    """
    Calculate total portfolio value.

    Args:
        holdings: Dictionary of holdings {symbol: {"shares": int, "avg_cost": float}}
        prices: Dictionary of current prices {symbol: price}

    Returns:
        Total portfolio value

    Example:
        >>> holdings = {"AAPL": {"shares": 100, "avg_cost": 150.0}}
        >>> prices = {"AAPL": 155.25}
        >>> calculate_portfolio_value(holdings, prices)
        15525.0
    """
    total_value = 0.0

    for symbol, holding in holdings.items():
        if symbol in prices:
            shares = holding.get("shares", 0)
            price = prices[symbol]
            total_value += shares * price

    return total_value


def calculate_portfolio_allocation(
    holdings: Dict[str, Dict[str, Any]], prices: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate portfolio allocation percentages.

    Args:
        holdings: Dictionary of holdings
        prices: Dictionary of current prices

    Returns:
        Dictionary of allocation percentages {symbol: percentage}

    Example:
        >>> holdings = {"AAPL": {"shares": 100}, "GOOGL": {"shares": 50}}
        >>> prices = {"AAPL": 150, "GOOGL": 2800}
        >>> allocation = calculate_portfolio_allocation(holdings, prices)
        >>> print(allocation["AAPL"])
        51.72
    """
    total_value = calculate_portfolio_value(holdings, prices)
    allocation = {}

    if total_value == 0:
        return allocation

    for symbol, holding in holdings.items():
        if symbol in prices:
            shares = holding.get("shares", 0)
            price = prices[symbol]
            value = shares * price
            allocation[symbol] = (value / total_value) * 100

    return allocation


def calculate_portfolio_return(
    holdings: Dict[str, Dict[str, Any]], prices: Dict[str, float]
) -> float:
    """
    Calculate total portfolio return.

    Args:
        holdings: Dictionary of holdings with average costs
        prices: Dictionary of current prices

    Returns:
        Portfolio return as percentage

    Example:
        >>> holdings = {"AAPL": {"shares": 100, "avg_cost": 140.0}}
        >>> prices = {"AAPL": 150.0}
        >>> calculate_portfolio_return(holdings, prices)
        7.14
    """
    total_current_value = 0.0
    total_cost_basis = 0.0

    for symbol, holding in holdings.items():
        if symbol in prices:
            shares = holding.get("shares", 0)
            avg_cost = holding.get("avg_cost", 0)
            current_price = prices[symbol]

            total_current_value += shares * current_price
            total_cost_basis += shares * avg_cost

    if total_cost_basis == 0:
        return 0.0

    return ((total_current_value - total_cost_basis) / total_cost_basis) * 100


def rebalance_portfolio(
    target_allocation: Dict[str, float],
    holdings: Dict[str, Dict[str, Any]],
    prices: Dict[str, float],
    portfolio_value: float,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate trades needed to rebalance portfolio.

    Args:
        target_allocation: Target allocation percentages
        holdings: Current holdings
        prices: Current prices
        portfolio_value: Total portfolio value

    Returns:
        Dictionary of required trades {symbol: {"action": "BUY"/"SELL", "shares": float}}

    Example:
        >>> target = {"AAPL": 0.6, "GOOGL": 0.4}
        >>> holdings = {"AAPL": {"shares": 50}, "GOOGL": {"shares": 50}}
        >>> prices = {"AAPL": 150, "GOOGL": 2800}
        >>> trades = rebalance_portfolio(target, holdings, prices, 155000)
        >>> print(trades["AAPL"]["action"])
        "BUY"
    """
    trades = {}

    for symbol, target_pct in target_allocation.items():
        target_value = portfolio_value * target_pct

        if symbol in holdings and symbol in prices:
            current_value = holdings[symbol]["shares"] * prices[symbol]
        else:
            current_value = 0.0

        diff_value = target_value - current_value

        if abs(diff_value) > 0.01:  # Only trade if difference is significant
            if symbol in prices:
                shares_to_trade = diff_value / prices[symbol]
                action = "BUY" if shares_to_trade > 0 else "SELL"
                trades[symbol] = {
                    "action": action,
                    "shares": abs(shares_to_trade),
                    "value": abs(diff_value),
                }

    return trades


def calculate_diversification_metrics(
    holdings: Dict[str, Dict[str, Any]],
    prices: Dict[str, float],
    sectors: Dict[str, str],
) -> Dict[str, Any]:
    """
    Calculate portfolio diversification metrics.

    Args:
        holdings: Portfolio holdings
        prices: Current prices
        sectors: Sector mapping {symbol: sector}

    Returns:
        Diversification metrics dictionary

    Example:
        >>> holdings = {"AAPL": {"shares": 100}, "MSFT": {"shares": 50}}
        >>> prices = {"AAPL": 150, "MSFT": 250}
        >>> sectors = {"AAPL": "Technology", "MSFT": "Technology"}
        >>> metrics = calculate_diversification_metrics(holdings, prices, sectors)
        >>> print(metrics["sector_concentration"]["Technology"])
        100.0
    """
    allocation = calculate_portfolio_allocation(holdings, prices)

    # Calculate sector concentration
    sector_values = {}
    for symbol, pct in allocation.items():
        sector = sectors.get(symbol, "Unknown")
        sector_values[sector] = sector_values.get(sector, 0) + pct

    # Calculate concentration metrics
    sorted_allocations = sorted(allocation.values(), reverse=True)

    # Herfindahl-Hirschman Index (HHI)
    hhi = sum((pct / 100) ** 2 for pct in allocation.values())

    # Number of effective positions
    effective_positions = 1 / hhi if hhi > 0 else 0

    return {
        "sector_concentration": sector_values,
        "largest_position": max(allocation.values()) if allocation else 0,
        "smallest_position": min(allocation.values()) if allocation else 0,
        "number_of_positions": len(allocation),
        "hhi": hhi,
        "effective_positions": effective_positions,
        "top_3_concentration": (
            sum(sorted_allocations[:3])
            if len(sorted_allocations) >= 3
            else sum(sorted_allocations)
        ),
    }


def calculate_portfolio_beta(
    holdings: Dict[str, Dict[str, Any]],
    prices: Dict[str, float],
    betas: Dict[str, float],
) -> float:
    """
    Calculate portfolio beta.

    Args:
        holdings: Portfolio holdings
        prices: Current prices
        betas: Stock betas {symbol: beta}

    Returns:
        Portfolio beta

    Example:
        >>> holdings = {"AAPL": {"shares": 100}, "MSFT": {"shares": 100}}
        >>> prices = {"AAPL": 150, "MSFT": 250}
        >>> betas = {"AAPL": 1.2, "MSFT": 0.9}
        >>> portfolio_beta = calculate_portfolio_beta(holdings, prices, betas)
        >>> print(f"Portfolio beta: {portfolio_beta:.2f}")
        1.04
    """
    allocation = calculate_portfolio_allocation(holdings, prices)
    portfolio_beta = 0.0

    for symbol, pct in allocation.items():
        if symbol in betas:
            weight = pct / 100  # Convert percentage to weight
            portfolio_beta += weight * betas[symbol]

    return portfolio_beta


def calculate_position_size(
    portfolio_value: float,
    risk_per_trade: float,
    stop_loss_pct: float,
    stock_price: float,
) -> int:
    """
    Calculate optimal position size based on risk management.

    Args:
        portfolio_value: Total portfolio value
        risk_per_trade: Risk percentage per trade (e.g., 0.02 for 2%)
        stop_loss_pct: Stop loss percentage (e.g., 0.05 for 5%)
        stock_price: Current stock price

    Returns:
        Recommended number of shares

    Example:
        >>> calculate_position_size(100000, 0.02, 0.05, 50)
        800
    """
    max_risk_amount = portfolio_value * risk_per_trade
    risk_per_share = stock_price * stop_loss_pct
    max_shares = max_risk_amount / risk_per_share

    return int(max_shares)


def calculate_portfolio_turnover(
    holdings_before: Dict[str, Dict[str, Any]],
    holdings_after: Dict[str, Dict[str, Any]],
    prices: Dict[str, float],
) -> float:
    """
    Calculate portfolio turnover rate.

    Args:
        holdings_before: Holdings before period
        holdings_after: Holdings after period
        prices: Current prices

    Returns:
        Turnover rate as percentage

    Example:
        >>> before = {"AAPL": {"shares": 100}}
        >>> after = {"AAPL": {"shares": 150}}
        >>> prices = {"AAPL": 150}
        >>> turnover = calculate_portfolio_turnover(before, after, prices)
        >>> print(f"Turnover: {turnover:.2f}%")
        25.0
    """
    value_before = calculate_portfolio_value(holdings_before, prices)
    value_after = calculate_portfolio_value(holdings_after, prices)

    if value_before == 0:
        return 0.0

    # Calculate total trades (buys + sells)
    all_symbols = set(list(holdings_before.keys()) + list(holdings_after.keys()))
    total_trades = 0.0

    for symbol in all_symbols:
        shares_before = holdings_before.get(symbol, {}).get("shares", 0)
        shares_after = holdings_after.get(symbol, {}).get("shares", 0)

        if symbol in prices:
            trade_volume = abs(shares_after - shares_before) * prices[symbol]
            total_trades += trade_volume

    avg_portfolio_value = (value_before + value_after) / 2
    turnover = (total_trades / avg_portfolio_value) * 100

    return turnover


def demo_portfolio_utils():
    """Demonstrate portfolio management utilities."""
    print("=" * 60)
    print("PORTFOLIO MANAGEMENT UTILITIES DEMONSTRATION")
    print("=" * 60)

    # Sample portfolio
    holdings = {
        "AAPL": {"shares": 100, "avg_cost": 140.0},
        "GOOGL": {"shares": 20, "avg_cost": 2600.0},
        "MSFT": {"shares": 50, "avg_cost": 230.0},
        "TSLA": {"shares": 30, "avg_cost": 750.0},
    }

    # Current prices
    prices = {"AAPL": 150.0, "GOOGL": 2800.0, "MSFT": 250.0, "TSLA": 800.0}

    # Sector mapping
    sectors = {
        "AAPL": "Technology",
        "GOOGL": "Technology",
        "MSFT": "Technology",
        "TSLA": "Consumer Discretionary",
    }

    # Stock betas
    betas = {"AAPL": 1.2, "GOOGL": 1.1, "MSFT": 0.9, "TSLA": 2.0}

    print("\n1. Portfolio Valuation:")
    portfolio_value = calculate_portfolio_value(holdings, prices)
    print(f"  Total Portfolio Value: ${portfolio_value:,.2f}")

    print("\n2. Portfolio Allocation:")
    allocation = calculate_portfolio_allocation(holdings, prices)
    for symbol, pct in sorted(allocation.items(), key=lambda x: x[1], reverse=True):
        print(f"  {symbol}: {pct:.2f}%")

    print("\n3. Portfolio Return:")
    portfolio_return = calculate_portfolio_return(holdings, prices)
    print(f"  Total Return: {portfolio_return:.2f}%")

    print("\n4. Portfolio Beta:")
    portfolio_beta = calculate_portfolio_beta(holdings, prices, betas)
    print(f"  Portfolio Beta: {portfolio_beta:.2f}")

    print("\n5. Diversification Metrics:")
    diversity = calculate_diversification_metrics(holdings, prices, sectors)
    print(f"  Number of Positions: {diversity['number_of_positions']}")
    print(f"  Largest Position: {diversity['largest_position']:.2f}%")
    print(f"  HHI Index: {diversity['hhi']:.4f}")
    print(f"  Effective Positions: {diversity['effective_positions']:.2f}")
    print("  Sector Concentration:")
    for sector, pct in diversity["sector_concentration"].items():
        print(f"    {sector}: {pct:.2f}%")

    print("\n6. Rebalancing Example:")
    target_allocation = {"AAPL": 0.4, "GOOGL": 0.3, "MSFT": 0.2, "TSLA": 0.1}
    trades = rebalance_portfolio(target_allocation, holdings, prices, portfolio_value)

    print("  Required Trades:")
    for symbol, trade in trades.items():
        print(
            f"    {symbol}: {trade['action']} {trade['shares']:.1f} shares (${trade['value']:,.2f})"
        )

    print("\n7. Position Sizing:")
    risk_per_trade = 0.02  # 2% risk per trade
    stop_loss = 0.05  # 5% stop loss

    for symbol, price in prices.items():
        shares = calculate_position_size(
            portfolio_value, risk_per_trade, stop_loss, price
        )
        position_value = shares * price
        print(f"  {symbol} @ ${price}: {shares} shares (${position_value:,.2f})")

    print("\n8. Portfolio Turnover:")
    # Simulate some trades
    holdings_after = holdings.copy()
    holdings_after["AAPL"]["shares"] = 120  # Buy 20 more shares
    holdings_after["TSLA"]["shares"] = 25  # Sell 5 shares

    turnover = calculate_portfolio_turnover(holdings, holdings_after, prices)
    print(f"  Portfolio Turnover: {turnover:.2f}%")


def main():
    """Main function to run demonstrations."""
    demo_portfolio_utils()
    print("\n🎉 Portfolio management utilities demonstration complete!")


if __name__ == "__main__":
    main()
