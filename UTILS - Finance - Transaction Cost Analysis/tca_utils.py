"""
Transaction Cost Analysis (TCA)
---------------------------------
Tools for measuring and analyzing execution quality and market impact.

- VWAP/TWAP: Benchmark execution prices
- Implementation Shortfall: Total cost of a trading decision vs. arrival price
- Almgren-Chriss: Linear impact model (temporary + permanent components)
- Square-root impact: Empirical market impact rule
"""

import numpy as np
from typing import Union


def vwap(
    prices: Union[list, np.ndarray],
    volumes: Union[list, np.ndarray],
) -> float:
    """
    Volume Weighted Average Price.

    Args:
        prices: Trade prices.
        volumes: Trade volumes at each price.

    Returns:
        float: VWAP.

    Raises:
        ValueError: If total volume is zero.
    """
    prices = np.array(prices, dtype=float)
    volumes = np.array(volumes, dtype=float)
    total_vol = np.sum(volumes)
    if total_vol == 0:
        raise ValueError("Total volume cannot be zero")
    return float(np.sum(prices * volumes) / total_vol)


def twap(prices: Union[list, np.ndarray]) -> float:
    """
    Time Weighted Average Price: simple mean of equally-spaced prices.

    Args:
        prices: Prices at equally spaced time intervals.

    Returns:
        float: TWAP.
    """
    return float(np.mean(prices))


def vwap_slippage(
    execution_price: float,
    vwap_price: float,
    side: str = "buy",
) -> float:
    """
    VWAP slippage in basis points.
    Buy: positive = paid more than VWAP (bad).
    Sell: positive = received less than VWAP (bad).

    Returns:
        float: Slippage in basis points.
    """
    if side == "buy":
        return float((execution_price - vwap_price) / vwap_price * 10_000)
    return float((vwap_price - execution_price) / vwap_price * 10_000)


def implementation_shortfall(
    decision_price: float,
    execution_prices: Union[list, np.ndarray],
    execution_quantities: Union[list, np.ndarray],
    final_price: float = None,
) -> dict:
    """
    Implementation Shortfall (IS): total cost vs. decision price.
    IS_bps = (avg_exec - decision_price) / decision_price * 10,000

    Args:
        decision_price: Price when trade decision was made.
        execution_prices: List of fill prices.
        execution_quantities: List of fill quantities.
        final_price: End-of-day price for missed opportunity cost (optional).

    Returns:
        dict: average_execution_price, implementation_shortfall_bps,
              missed_opportunity_bps, total_cost_bps.
    """
    exec_prices = np.array(execution_prices, dtype=float)
    exec_qtys = np.array(execution_quantities, dtype=float)
    total_qty = np.sum(exec_qtys)

    if total_qty == 0:
        raise ValueError("No executions provided")

    avg_exec = float(np.sum(exec_prices * exec_qtys) / total_qty)
    is_bps = float((avg_exec - decision_price) / decision_price * 10_000)

    missed_opp = 0.0
    if final_price is not None:
        missed_opp = float((final_price - decision_price) / decision_price * 10_000)

    return {
        "average_execution_price": avg_exec,
        "decision_price": decision_price,
        "implementation_shortfall_bps": is_bps,
        "missed_opportunity_bps": missed_opp,
        "total_cost_bps": is_bps + missed_opp,
    }


def almgren_chriss_impact(
    order_size: float,
    adv: float,
    sigma: float,
    T: float,
    eta: float = 0.1,
    gamma: float = 0.1,
) -> dict:
    """
    Almgren-Chriss linear impact model (uniform trading schedule).
    Temporary impact: h(v) = eta * v/adv (per-unit cost at rate v)
    Permanent impact: g(v) = gamma * v/adv (lasting price shift)

    Args:
        order_size: Total shares to trade.
        adv: Average daily volume.
        sigma: Daily volatility (decimal).
        T: Days to complete order.
        eta: Temporary impact coefficient.
        gamma: Permanent impact coefficient.

    Returns:
        dict: participation_rate, temporary/permanent/expected impact in bps.
    """
    participation_rate = float(order_size / (adv * T))
    daily_rate = order_size / T

    temp_impact = float(eta * daily_rate / adv)
    perm_impact = float(gamma * order_size / adv)
    expected_shortfall = float(0.5 * perm_impact + temp_impact)
    timing_risk = float(sigma**2 * T / 6)

    return {
        "participation_rate": participation_rate,
        "temporary_impact_bps": temp_impact * 10_000,
        "permanent_impact_bps": perm_impact * 10_000,
        "expected_shortfall_bps": expected_shortfall * 10_000,
        "timing_risk_variance": timing_risk,
    }


def sqrt_market_impact(
    order_size: float,
    adv: float,
    sigma: float,
    alpha: float = 0.5,
) -> float:
    """
    Square-root market impact (empirical rule).
    Impact = sigma * alpha * sqrt(order_size / adv) * 10,000

    Args:
        order_size: Trade size.
        adv: Average daily volume.
        sigma: Daily volatility.
        alpha: Impact coefficient (typically 0.3–1.0).

    Returns:
        float: Market impact in basis points.
    """
    return float(sigma * alpha * np.sqrt(order_size / adv) * 10_000)


if __name__ == "__main__":
    print("Transaction Cost Analysis")
    print("=" * 40)

    prices = [100.10, 100.15, 100.12, 100.20, 100.18]
    volumes = [1000, 2000, 1500, 3000, 2500]
    vwap_price = vwap(prices, volumes)
    print(f"\nVWAP:  {vwap_price:.4f}")
    print(f"TWAP:  {twap(prices):.4f}")
    print(f"Buy slippage vs VWAP: {vwap_slippage(100.16, vwap_price, 'buy'):.2f} bps")

    is_result = implementation_shortfall(
        decision_price=100.00,
        execution_prices=[100.05, 100.10, 100.15],
        execution_quantities=[1000, 1000, 1000],
        final_price=100.20,
    )
    print(f"\nImplementation Shortfall:")
    for k, v in is_result.items():
        print(f"  {k:35s}: {v:.4f}")

    impact = almgren_chriss_impact(100_000, 1_000_000, 0.015, T=5)
    print(f"\nAlmgren-Chriss Impact (100k shares, 1M ADV, 5 days):")
    print(f"  Participation rate:  {impact['participation_rate']:.1%}")
    print(f"  Temporary impact:    {impact['temporary_impact_bps']:.2f} bps")
    print(f"  Expected shortfall:  {impact['expected_shortfall_bps']:.2f} bps")

    sqr = sqrt_market_impact(100_000, 1_000_000, 0.015)
    print(f"\nSqrt Impact: {sqr:.2f} bps")
