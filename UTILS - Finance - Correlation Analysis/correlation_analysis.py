"""
Correlation Analysis
Tools for analyzing correlations between financial instruments.
"""

from typing import Dict, List, Tuple

import numpy as np


def pearson_correlation(returns1: List[float], returns2: List[float]) -> float:
    """
    Calculate Pearson correlation coefficient between two return series.

    Args:
        returns1: First return series
        returns2: Second return series

    Returns:
        Correlation coefficient between -1 and 1
    """
    if len(returns1) != len(returns2):
        raise ValueError("Return series must have same length")

    r1 = np.array(returns1)
    r2 = np.array(returns2)

    return np.corrcoef(r1, r2)[0, 1]


def rolling_correlation(returns1: List[float], returns2: List[float], window: int = 30) -> List[float]:
    """
    Calculate rolling correlation between two return series.

    Args:
        returns1: First return series
        returns2: Second return series
        window: Rolling window size

    Returns:
        List of rolling correlations
    """
    if len(returns1) != len(returns2):
        raise ValueError("Return series must have same length")

    r1 = np.array(returns1)
    r2 = np.array(returns2)
    n = len(r1)

    correlations = []
    for i in range(window, n + 1):
        corr = np.corrcoef(r1[i - window : i], r2[i - window : i])[0, 1]
        correlations.append(corr)

    return correlations


def correlation_matrix(returns_dict: Dict[str, List[float]]) -> Tuple[np.ndarray, List[str]]:
    """
    Calculate correlation matrix for multiple assets.

    Args:
        returns_dict: Dictionary mapping asset names to return series

    Returns:
        Tuple of (correlation matrix, list of asset names)
    """
    assets = list(returns_dict.keys())

    returns_matrix = np.array([returns_dict[asset] for asset in assets])
    corr_matrix = np.corrcoef(returns_matrix)

    return corr_matrix, assets


def ewma_correlation(returns1: List[float], returns2: List[float], lambda_param: float = 0.94) -> List[float]:
    """
    Calculate exponentially weighted moving average correlation.

    Args:
        returns1: First return series
        returns2: Second return series
        lambda_param: Decay factor

    Returns:
        List of EWMA correlations
    """
    r1 = np.array(returns1)
    r2 = np.array(returns2)
    n = len(r1)

    ewma_cov = np.zeros(n)
    ewma_var1 = np.zeros(n)
    ewma_var2 = np.zeros(n)

    ewma_cov[0] = r1[0] * r2[0]
    ewma_var1[0] = r1[0] ** 2
    ewma_var2[0] = r2[0] ** 2

    for i in range(1, n):
        ewma_cov[i] = lambda_param * ewma_cov[i - 1] + (1 - lambda_param) * r1[i] * r2[i]
        ewma_var1[i] = lambda_param * ewma_var1[i - 1] + (1 - lambda_param) * r1[i] ** 2
        ewma_var2[i] = lambda_param * ewma_var2[i - 1] + (1 - lambda_param) * r2[i] ** 2

    ewma_corr = ewma_cov / (np.sqrt(ewma_var1) * np.sqrt(ewma_var2))

    return ewma_corr.tolist()


def rank_correlation(returns1: List[float], returns2: List[float]) -> float:
    """
    Calculate Spearman rank correlation coefficient.
    More robust to outliers than Pearson correlation.

    Args:
        returns1: First return series
        returns2: Second return series

    Returns:
        Rank correlation coefficient
    """
    from scipy.stats import spearmanr

    return spearmanr(returns1, returns2)[0]


def tail_correlation(returns1: List[float], returns2: List[float], quantile: float = 0.05) -> Tuple[float, float]:
    """
    Calculate correlation in the tails of the distribution.
    Useful for understanding correlation during extreme events.

    Args:
        returns1: First return series
        returns2: Second return series
        quantile: Quantile threshold for tail definition

    Returns:
        Tuple of (lower tail correlation, upper tail correlation)
    """
    r1 = np.array(returns1)
    r2 = np.array(returns2)

    lower_threshold = np.quantile(r1, quantile)
    upper_threshold = np.quantile(r1, 1 - quantile)

    lower_mask = r1 <= lower_threshold
    upper_mask = r1 >= upper_threshold

    lower_corr = np.corrcoef(r1[lower_mask], r2[lower_mask])[0, 1] if np.sum(lower_mask) > 1 else np.nan
    upper_corr = np.corrcoef(r1[upper_mask], r2[upper_mask])[0, 1] if np.sum(upper_mask) > 1 else np.nan

    return lower_corr, upper_corr


def correlation_stability(returns1: List[float], returns2: List[float], n_splits: int = 5) -> Dict[str, float]:
    """
    Test correlation stability across different time periods.

    Args:
        returns1: First return series
        returns2: Second return series
        n_splits: Number of periods to split data into

    Returns:
        Dictionary with stability metrics
    """
    n = len(returns1)
    split_size = n // n_splits

    correlations = []
    for i in range(n_splits):
        start = i * split_size
        end = start + split_size if i < n_splits - 1 else n
        corr = pearson_correlation(returns1[start:end], returns2[start:end])
        correlations.append(corr)

    return {
        "mean": np.mean(correlations),
        "std": np.std(correlations),
        "min": np.min(correlations),
        "max": np.max(correlations),
        "range": np.max(correlations) - np.min(correlations),
    }


if __name__ == "__main__":
    np.random.seed(42)

    returns1 = np.random.normal(0.001, 0.02, 100)
    returns2 = 0.7 * returns1 + 0.3 * np.random.normal(0.001, 0.02, 100)

    print("Correlation Analysis Demo")
    print("=" * 50)

    pearson = pearson_correlation(returns1.tolist(), returns2.tolist())
    print(f"Pearson Correlation: {pearson:.4f}")

    rolling = rolling_correlation(returns1.tolist(), returns2.tolist(), window=20)
    print(f"Rolling Correlation (latest): {rolling[-1]:.4f}")

    lower_tail, upper_tail = tail_correlation(returns1.tolist(), returns2.tolist())
    print(f"Lower Tail Correlation: {lower_tail:.4f}")
    print(f"Upper Tail Correlation: {upper_tail:.4f}")

    stability = correlation_stability(returns1.tolist(), returns2.tolist())
    print("\nCorrelation Stability:")
    print(f"  Mean: {stability['mean']:.4f}")
    print(f"  Std Dev: {stability['std']:.4f}")
    print(f"  Range: {stability['range']:.4f}")
