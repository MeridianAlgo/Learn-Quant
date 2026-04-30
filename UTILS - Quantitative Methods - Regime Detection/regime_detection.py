"""
Market Regime Detection
------------------------
Identifies distinct market regimes (bull/bear, low/high volatility)
using Gaussian Mixture Models and moving average crossover heuristics.

Regimes help adapt trading strategies to current market conditions.
"""

import numpy as np
from typing import Union


def moving_average_regime(
    prices: Union[list, np.ndarray],
    short_window: int = 50,
    long_window: int = 200,
) -> np.ndarray:
    """
    Bull/Bear regime via moving average crossover.
    Regime 1 (Bull): short MA > long MA
    Regime 0 (Bear): short MA <= long MA

    Args:
        prices: Price series.
        short_window: Short MA period.
        long_window: Long MA period.

    Returns:
        np.ndarray: Regime labels (0/1), NaN for warmup period.
    """
    prices = np.array(prices, dtype=float)
    n = len(prices)
    regimes = np.full(n, np.nan)

    for i in range(long_window - 1, n):
        short_ma = np.mean(prices[i - short_window + 1: i + 1])
        long_ma = np.mean(prices[i - long_window + 1: i + 1])
        regimes[i] = 1.0 if short_ma > long_ma else 0.0

    return regimes


def volatility_regime(
    returns: Union[list, np.ndarray],
    window: int = 21,
    n_regimes: int = 3,
) -> np.ndarray:
    """
    Classify volatility regimes (low/medium/high) using rolling volatility
    and quantile thresholds.

    Args:
        returns: Return series.
        window: Rolling window for realized volatility.
        n_regimes: Number of regime buckets (2 or 3).

    Returns:
        np.ndarray: Regime labels (0=low, 1=medium, 2=high).
    """
    returns = np.array(returns)
    n = len(returns)
    rolling_vol = np.full(n, np.nan)

    for i in range(window - 1, n):
        rolling_vol[i] = np.std(returns[i - window + 1: i + 1]) * np.sqrt(252)

    valid = rolling_vol[~np.isnan(rolling_vol)]
    thresholds = np.percentile(valid, np.linspace(0, 100, n_regimes + 1)[1:-1])

    labels = np.full(n, np.nan)
    for i in range(window - 1, n):
        v = rolling_vol[i]
        label = 0
        for threshold in thresholds:
            if v > threshold:
                label += 1
        labels[i] = float(label)

    return labels


def gaussian_mixture_regime(
    returns: Union[list, np.ndarray],
    n_regimes: int = 2,
) -> dict:
    """
    Fit a Gaussian Mixture Model to detect regimes from return distribution.

    Args:
        returns: Return series.
        n_regimes: Number of regimes.

    Returns:
        dict: labels, means, stds, weights per regime.

    Raises:
        ImportError: If scikit-learn is not installed.
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        raise ImportError("scikit-learn required: pip install scikit-learn")

    returns_arr = np.array(returns).reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_regimes, covariance_type="full", random_state=42)
    gmm.fit(returns_arr)
    labels = gmm.predict(returns_arr)

    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_

    order = np.argsort(means)
    means = means[order]
    stds = stds[order]
    weights = weights[order]
    remap = {int(old): int(new) for new, old in enumerate(order)}
    labels = np.array([remap[int(l)] for l in labels])

    return {
        "labels": labels,
        "means": means,
        "stds": stds,
        "weights": weights,
        "n_regimes": n_regimes,
    }


def regime_stats(returns: Union[list, np.ndarray], labels: np.ndarray) -> dict:
    """
    Compute per-regime return statistics.

    Args:
        returns: Return series.
        labels: Regime label array (same length).

    Returns:
        dict: Per-regime mean, std, count, annualized metrics.
    """
    returns = np.array(returns)
    unique = sorted(set(int(l) for l in labels if not np.isnan(l)))
    result = {}
    for r in unique:
        mask = labels == r
        regime_rets = returns[mask]
        result[r] = {
            "mean": float(np.mean(regime_rets)),
            "std": float(np.std(regime_rets)),
            "count": int(np.sum(mask)),
            "annualized_return": float(np.mean(regime_rets) * 252),
            "annualized_vol": float(np.std(regime_rets) * np.sqrt(252)),
        }
    return result


if __name__ == "__main__":
    np.random.seed(42)
    n = 500
    regime_true = np.random.choice([0, 1], size=n, p=[0.6, 0.4])
    returns = np.where(
        regime_true == 0,
        np.random.normal(0.0005, 0.01, n),
        np.random.normal(-0.001, 0.02, n),
    )

    print("Regime Detection")
    print("=" * 40)

    try:
        gmm_result = gaussian_mixture_regime(returns, n_regimes=2)
        stats = regime_stats(returns, gmm_result["labels"])
        print("\nGMM Regimes:")
        for regime, s in stats.items():
            label = "Bullish" if s["mean"] > 0 else "Bearish"
            print(f"  Regime {regime} ({label}): "
                  f"ann_return={s['annualized_return']:.2%}, "
                  f"ann_vol={s['annualized_vol']:.2%}, "
                  f"n={s['count']}")
    except ImportError:
        print("scikit-learn not available; using volatility regime fallback")
        vol_labels = volatility_regime(returns, n_regimes=2)
        stats = regime_stats(returns, vol_labels[~np.isnan(vol_labels)].astype(int))
        print("Volatility Regimes:", stats)
