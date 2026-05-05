"""
Cointegration Tests and Utilities
----------------------------------
Cointegration: a linear combination of non-stationary series that is stationary.
Foundation of pairs trading and statistical arbitrage.

Implements:
  - Augmented Dickey-Fuller (ADF) stationarity test (simplified)
  - Engle-Granger two-step cointegration test
  - OLS hedge ratio
  - Half-life of mean reversion (Ornstein-Uhlenbeck fit)
"""

from typing import Union

import numpy as np


def ols_hedge_ratio(y: Union[list, np.ndarray], x: Union[list, np.ndarray]) -> dict:
    """
    OLS regression: y = alpha + beta * x + e.

    Args:
        y: Dependent series.
        x: Independent series.

    Returns:
        dict: alpha, beta, residuals, r_squared.
    """
    y = np.array(y, dtype=float)
    x = np.array(x, dtype=float)
    if len(y) != len(x):
        raise ValueError("y and x must be the same length")

    x_mat = np.column_stack([np.ones_like(x), x])
    coeffs, *_ = np.linalg.lstsq(x_mat, y, rcond=None)
    alpha, beta = float(coeffs[0]), float(coeffs[1])
    fitted = alpha + beta * x
    residuals = y - fitted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "alpha": alpha,
        "beta": beta,
        "residuals": residuals,
        "r_squared": float(r2),
    }


def adf_test(series: Union[list, np.ndarray], lags: int = 1) -> dict:
    """
    Augmented Dickey-Fuller test for unit root (simplified).

    Regresses delta_y_t = rho * y_{t-1} + sum(gamma_i * delta_y_{t-i}) + e_t.
    Null: rho = 0 (unit root, non-stationary).
    A more negative t-stat rejects the null (stationary).

    Args:
        series: Time series.
        lags: Number of lagged differences.

    Returns:
        dict: t_stat, rho, n_obs. Compare t_stat to MacKinnon critical values.
    """
    y = np.array(series, dtype=float)
    n = len(y)
    if n < lags + 5:
        raise ValueError("series too short for ADF with these lags")

    dy = np.diff(y)
    rows = n - 1 - lags
    Y = dy[lags:]
    X_cols = [y[lags : n - 1], np.ones(rows)]
    for i in range(1, lags + 1):
        X_cols.append(dy[lags - i : n - 1 - i])
    X = np.column_stack(X_cols)

    coeffs, *_ = np.linalg.lstsq(X, Y, rcond=None)
    fitted = X @ coeffs
    resid = Y - fitted
    df = rows - X.shape[1]
    sigma2 = float(np.sum(resid**2) / df) if df > 0 else float("nan")
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se_rho = float(np.sqrt(cov[0, 0]))
    rho = float(coeffs[0])
    t_stat = rho / se_rho if se_rho > 0 else float("nan")

    return {
        "t_stat": t_stat,
        "rho": rho,
        "n_obs": rows,
        "stationary_5pct": t_stat < -2.86,
        "stationary_1pct": t_stat < -3.43,
    }


def engle_granger(
    y: Union[list, np.ndarray],
    x: Union[list, np.ndarray],
    lags: int = 1,
) -> dict:
    """
    Engle-Granger two-step cointegration test.
      1. Regress y on x via OLS (find hedge ratio).
      2. Run ADF on residuals.

    Stationary residuals → cointegrated.

    Args:
        y, x: Two series.
        lags: ADF lag order on residuals.

    Returns:
        dict: hedge ratio (beta), spread, ADF stats.
    """
    fit = ols_hedge_ratio(y, x)
    adf = adf_test(fit["residuals"], lags=lags)
    return {
        "alpha": fit["alpha"],
        "hedge_ratio": fit["beta"],
        "spread": fit["residuals"],
        "r_squared": fit["r_squared"],
        "adf_t_stat": adf["t_stat"],
        "cointegrated_5pct": adf["stationary_5pct"],
        "cointegrated_1pct": adf["stationary_1pct"],
    }


def half_life(spread: Union[list, np.ndarray]) -> float:
    """
    Half-life of mean reversion via Ornstein-Uhlenbeck AR(1) fit.

    delta_s_t = -theta * s_{t-1} + e_t  ->  half_life = ln(2) / theta.

    Args:
        spread: Stationary spread series.

    Returns:
        float: Half-life in periods. Inf if no mean reversion detected.
    """
    s = np.array(spread, dtype=float)
    s_lag = s[:-1]
    ds = np.diff(s)
    X = np.column_stack([np.ones_like(s_lag), s_lag])
    coeffs, *_ = np.linalg.lstsq(X, ds, rcond=None)
    theta = -float(coeffs[1])
    if theta <= 0:
        return float("inf")
    return float(np.log(2.0) / theta)


def zscore_spread(spread: Union[list, np.ndarray], window: int = 60) -> np.ndarray:
    """
    Rolling z-score of a spread for entry/exit signals.

    Args:
        spread: Spread series.
        window: Lookback for mean and std.

    Returns:
        np.ndarray: z-scores (NaN for warmup).
    """
    s = np.array(spread, dtype=float)
    n = len(s)
    z = np.full(n, np.nan)
    for i in range(window - 1, n):
        w = s[i - window + 1 : i + 1]
        mu = float(np.mean(w))
        sd = float(np.std(w, ddof=1))
        z[i] = (s[i] - mu) / sd if sd > 0 else 0.0
    return z


if __name__ == "__main__":
    np.random.seed(42)
    n = 500
    x = np.cumsum(np.random.normal(0, 1, n))
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.7 * spread[t - 1] + np.random.normal(0, 0.5)
    y = 1.5 * x + spread + 10.0

    print("Engle-Granger Cointegration Test")
    print("=" * 40)
    eg = engle_granger(y, x, lags=1)
    print(f"hedge ratio    = {eg['hedge_ratio']:.4f}  (true ~1.5)")
    print(f"R^2            = {eg['r_squared']:.4f}")
    print(f"ADF t-stat     = {eg['adf_t_stat']:.4f}")
    print(f"cointegrated@5%= {eg['cointegrated_5pct']}")
    print(f"half-life      = {half_life(eg['spread']):.2f} periods")
