"""
Brinson Performance Attribution
--------------------------------
Decomposes portfolio active return vs benchmark into:
  - Allocation effect: over/underweight sectors
  - Selection effect: stock picking within sectors
  - Interaction effect: combination of allocation and selection

Reference: Brinson, Hood, Beebower (1986).
"""

from typing import Union

import numpy as np


def brinson_attribution(
    portfolio_weights: Union[list, np.ndarray],
    portfolio_returns: Union[list, np.ndarray],
    benchmark_weights: Union[list, np.ndarray],
    benchmark_returns: Union[list, np.ndarray],
) -> dict:
    """
    Brinson-Hood-Beebower attribution model.

    Active return = Allocation + Selection + Interaction.

    Args:
        portfolio_weights: Portfolio weights per sector/group.
        portfolio_returns: Portfolio sector returns.
        benchmark_weights: Benchmark weights per sector/group.
        benchmark_returns: Benchmark sector returns.

    Returns:
        dict: per-sector and total allocation, selection, interaction, active return.
    """
    wp = np.array(portfolio_weights, dtype=float)
    rp = np.array(portfolio_returns, dtype=float)
    wb = np.array(benchmark_weights, dtype=float)
    rb = np.array(benchmark_returns, dtype=float)

    if not (len(wp) == len(rp) == len(wb) == len(rb)):
        raise ValueError("All input arrays must have equal length")

    benchmark_total = float(np.sum(wb * rb))

    allocation = (wp - wb) * (rb - benchmark_total)
    selection = wb * (rp - rb)
    interaction = (wp - wb) * (rp - rb)

    total_alloc = float(np.sum(allocation))
    total_sel = float(np.sum(selection))
    total_inter = float(np.sum(interaction))
    portfolio_total = float(np.sum(wp * rp))
    active_return = portfolio_total - benchmark_total

    return {
        "allocation_per_sector": allocation,
        "selection_per_sector": selection,
        "interaction_per_sector": interaction,
        "total_allocation": total_alloc,
        "total_selection": total_sel,
        "total_interaction": total_inter,
        "portfolio_return": portfolio_total,
        "benchmark_return": benchmark_total,
        "active_return": float(active_return),
        "explained": float(total_alloc + total_sel + total_inter),
    }


def two_factor_brinson(
    portfolio_weights: Union[list, np.ndarray],
    portfolio_returns: Union[list, np.ndarray],
    benchmark_weights: Union[list, np.ndarray],
    benchmark_returns: Union[list, np.ndarray],
) -> dict:
    """
    Two-factor variant: combines selection + interaction into a single
    'security selection' effect (commonly used in practice).

    Returns:
        dict: allocation effect, selection effect (incl. interaction), total.
    """
    wp = np.array(portfolio_weights, dtype=float)
    rp = np.array(portfolio_returns, dtype=float)
    wb = np.array(benchmark_weights, dtype=float)
    rb = np.array(benchmark_returns, dtype=float)

    benchmark_total = float(np.sum(wb * rb))
    allocation = (wp - wb) * (rb - benchmark_total)
    selection = wp * (rp - rb)

    return {
        "allocation_per_sector": allocation,
        "selection_per_sector": selection,
        "total_allocation": float(np.sum(allocation)),
        "total_selection": float(np.sum(selection)),
        "active_return": float(np.sum(wp * rp) - benchmark_total),
    }


def information_ratio(
    portfolio_returns: Union[list, np.ndarray],
    benchmark_returns: Union[list, np.ndarray],
    periods_per_year: int = 252,
) -> float:
    """
    Information Ratio = mean(active return) / tracking error, annualized.

    Args:
        portfolio_returns: Portfolio return series.
        benchmark_returns: Benchmark return series.
        periods_per_year: Annualization factor.

    Returns:
        float: Annualized IR.
    """
    rp = np.array(portfolio_returns, dtype=float)
    rb = np.array(benchmark_returns, dtype=float)
    active = rp - rb
    te = float(np.std(active, ddof=1))
    if te == 0.0:
        return 0.0
    return float(np.mean(active) / te * np.sqrt(periods_per_year))


def tracking_error(
    portfolio_returns: Union[list, np.ndarray],
    benchmark_returns: Union[list, np.ndarray],
    periods_per_year: int = 252,
) -> float:
    """
    Annualized tracking error (std of active returns).
    """
    rp = np.array(portfolio_returns, dtype=float)
    rb = np.array(benchmark_returns, dtype=float)
    return float(np.std(rp - rb, ddof=1) * np.sqrt(periods_per_year))


if __name__ == "__main__":
    sectors = ["Tech", "Finance", "Energy", "Healthcare"]
    wp = [0.40, 0.20, 0.10, 0.30]
    rp = [0.12, 0.05, -0.03, 0.08]
    wb = [0.30, 0.25, 0.20, 0.25]
    rb = [0.10, 0.06, -0.02, 0.07]

    print("Brinson Attribution")
    print("=" * 50)
    res = brinson_attribution(wp, rp, wb, rb)
    print(f"Portfolio  : {res['portfolio_return']:.4f}")
    print(f"Benchmark  : {res['benchmark_return']:.4f}")
    print(f"Active     : {res['active_return']:.4f}")
    print(f"Allocation : {res['total_allocation']:.4f}")
    print(f"Selection  : {res['total_selection']:.4f}")
    print(f"Interaction: {res['total_interaction']:.4f}")
