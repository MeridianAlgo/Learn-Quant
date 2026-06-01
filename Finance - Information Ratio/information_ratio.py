"""
Information Ratio & Active Management Metrics
----------------------------------------------
When a portfolio is run *against a benchmark*, the questions that matter are not
"how much did we return?" but "how much did we beat the benchmark by, and how
reliably?". This module covers the core metrics of active management:

- Active return     — portfolio return minus benchmark return.
- Tracking error    — volatility of the active return (the risk taken to deviate
                      from the benchmark).
- Information Ratio — active return per unit of tracking error. The active-
                      management analogue of the Sharpe ratio.
- Appraisal ratio   — Jensen's alpha divided by the volatility of regression
                      residuals (idiosyncratic risk), via a CAPM-style fit.

These are the numbers used to judge a long-only active manager or a market-
neutral overlay. A higher Information Ratio means more consistent outperformance
per unit of benchmark-relative risk.
"""

import numpy as np

# Trading days per year — used to annualise daily statistics.
TRADING_DAYS = 252


def active_returns(portfolio: np.ndarray, benchmark: np.ndarray) -> np.ndarray:
    """
    Active (excess-over-benchmark) returns, period by period.

    Args:
        portfolio: Array of portfolio returns.
        benchmark: Array of benchmark returns (same length).

    Returns:
        np.ndarray: portfolio - benchmark, element-wise.
    """
    portfolio = np.asarray(portfolio, dtype=float)
    benchmark = np.asarray(benchmark, dtype=float)
    if portfolio.shape != benchmark.shape:
        raise ValueError("portfolio and benchmark must be the same length")
    return portfolio - benchmark


def tracking_error(portfolio: np.ndarray, benchmark: np.ndarray, periods: int = TRADING_DAYS) -> float:
    """
    Annualised tracking error: the standard deviation of active returns.

    Args:
        portfolio: Portfolio returns.
        benchmark: Benchmark returns.
        periods: Periods per year for annualisation (252 daily, 12 monthly).

    Returns:
        float: Annualised tracking error.
    """
    active = active_returns(portfolio, benchmark)
    # Volatility scales with the square root of time.
    return float(active.std(ddof=1) * np.sqrt(periods))


def information_ratio(portfolio: np.ndarray, benchmark: np.ndarray, periods: int = TRADING_DAYS) -> float:
    """
    Information Ratio = annualised active return / annualised tracking error.

    Interpreted like a Sharpe ratio but relative to a benchmark. IR > 0.5 is
    good, > 1.0 is excellent and rare over long horizons.

    Args:
        portfolio: Portfolio returns.
        benchmark: Benchmark returns.
        periods: Periods per year for annualisation.

    Returns:
        float: Information Ratio (0.0 if tracking error is zero).
    """
    active = active_returns(portfolio, benchmark)
    te = active.std(ddof=1)
    if te == 0:
        return 0.0
    # Annualise numerator (mean × periods) and denominator (std × sqrt(periods));
    # the ratio simplifies to mean/std × sqrt(periods).
    return float(active.mean() / te * np.sqrt(periods))


def appraisal_ratio(portfolio: np.ndarray, benchmark: np.ndarray, periods: int = TRADING_DAYS) -> dict:
    """
    Appraisal ratio via a CAPM-style regression of portfolio on benchmark.

    Fits  r_p = alpha + beta * r_b + epsilon  by ordinary least squares. The
    appraisal ratio is alpha divided by the volatility of the residuals
    (idiosyncratic risk) — it measures stock-picking skill net of market beta.

    Args:
        portfolio: Portfolio returns.
        benchmark: Benchmark returns.
        periods: Periods per year for annualisation.

    Returns:
        dict with annualised "alpha", "beta", "residual_vol", and
        "appraisal_ratio".
    """
    r_p = np.asarray(portfolio, dtype=float)
    r_b = np.asarray(benchmark, dtype=float)
    # Design matrix [1, r_b] for the intercept (alpha) and slope (beta).
    X = np.column_stack([np.ones_like(r_b), r_b])
    coef, *_ = np.linalg.lstsq(X, r_p, rcond=None)
    alpha, beta = coef
    residuals = r_p - X @ coef
    resid_vol = residuals.std(ddof=1)

    ann_alpha = alpha * periods
    ann_resid_vol = resid_vol * np.sqrt(periods)
    ratio = ann_alpha / ann_resid_vol if ann_resid_vol > 0 else 0.0
    return {
        "alpha": float(ann_alpha),
        "beta": float(beta),
        "residual_vol": float(ann_resid_vol),
        "appraisal_ratio": float(ratio),
    }


if __name__ == "__main__":
    print("Information Ratio & Active Management Metrics")
    print("=" * 46)

    rng = np.random.default_rng(11)
    # Benchmark: broad market. Portfolio: benchmark beta of ~1.05 plus a small
    # consistent alpha and some idiosyncratic noise.
    bench = rng.normal(0.0004, 0.011, TRADING_DAYS * 2)
    port = 0.0002 + 1.05 * bench + rng.normal(0, 0.004, len(bench))

    te = tracking_error(port, bench)
    ir = information_ratio(port, bench)
    print(f"\nAnnualised tracking error : {te:.2%}")
    print(f"Information Ratio         : {ir:.3f}")

    ap = appraisal_ratio(port, bench)
    print("\nCAPM-style regression:")
    print(f"  Annualised alpha   : {ap['alpha']:+.2%}")
    print(f"  Beta               : {ap['beta']:.3f}")
    print(f"  Residual vol       : {ap['residual_vol']:.2%}")
    print(f"  Appraisal ratio    : {ap['appraisal_ratio']:.3f}")
