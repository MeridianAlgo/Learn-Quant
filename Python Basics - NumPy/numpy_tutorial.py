"""NumPy Tutorial for Quantitative Finance.

Run with:
    python numpy_tutorial.py

Covers arrays, vectorised returns, descriptive statistics, broadcasting,
covariance/portfolio variance, and boolean indexing — the NumPy primitives
that appear in virtually every quant codebase.
"""

from pathlib import Path

import numpy as np

SOURCE_FILE = Path(__file__).resolve()


def intro() -> None:
    print("\n" + "#" * 60)
    print("NUMPY FOR QUANTITATIVE FINANCE")
    print("#" * 60)
    print("Executing file:", SOURCE_FILE.name)
    print("Folder location:", SOURCE_FILE.parent.relative_to(Path.cwd()))
    print("NumPy is the backbone of numerical computing in Python.\n")


def arrays_and_dtypes() -> None:
    print("=" * 60)
    print("ARRAYS AND DTYPES")
    print("=" * 60)

    prices = np.array([150.0, 152.3, 148.7, 155.1, 153.9])
    print(f"Price array: {prices}")
    print(f"dtype: {prices.dtype}, shape: {prices.shape}, ndim: {prices.ndim}")

    # 2-D: rows = days, cols = [open, high, low, close]
    ohlc = np.array([
        [150.0, 155.0, 149.0, 152.3],
        [152.3, 156.0, 151.0, 148.7],
        [148.7, 154.0, 147.5, 155.1],
    ])
    print(f"\nOHLC matrix shape: {ohlc.shape}  (days x fields)")
    print(f"Closing prices (col 3): {ohlc[:, 3]}")


def vectorised_returns() -> None:
    print("\n" + "=" * 60)
    print("VECTORISED RETURNS")
    print("=" * 60)

    prices = np.array([100.0, 102.0, 99.5, 105.0, 103.2, 108.0])
    daily_returns = np.diff(prices) / prices[:-1]
    log_returns = np.log(prices[1:] / prices[:-1])

    print(f"Prices:        {prices}")
    print(f"Daily returns: {np.round(daily_returns, 4)}")
    print(f"Log returns:   {np.round(log_returns, 4)}")
    print(f"Cumulative return: {(prices[-1] / prices[0] - 1):.2%}")


def descriptive_statistics() -> None:
    print("\n" + "=" * 60)
    print("DESCRIPTIVE STATISTICS (VECTORISED)")
    print("=" * 60)

    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 252)  # one year of daily returns

    mu = returns.mean()
    sigma = returns.std(ddof=1)
    sharpe = (mu / sigma) * np.sqrt(252)

    print(f"Mean daily return:   {mu:.4%}")
    print(f"Daily volatility:    {sigma:.4%}")
    print(f"Annualised Sharpe:   {sharpe:.3f}")
    print(f"Min / Max:           {returns.min():.4%} / {returns.max():.4%}")
    skewness = ((returns - mu) ** 3).mean() / sigma ** 3
    print(f"Skewness:            {skewness:.3f}")


def broadcasting_demo() -> None:
    print("\n" + "=" * 60)
    print("BROADCASTING")
    print("=" * 60)

    weights = np.array([0.4, 0.3, 0.3])
    asset_returns = np.array([
        [0.010,  0.005, -0.003],
        [-0.005, 0.012,  0.008],
        [0.008, -0.002,  0.015],
    ])
    # Multiply each row by weights without a loop, then sum
    portfolio_returns = (asset_returns * weights).sum(axis=1)
    print(f"Weights:            {weights}")
    print(f"Asset returns (3 days x 3 assets):\n{asset_returns}")
    print(f"Portfolio returns:  {np.round(portfolio_returns, 6)}")


def covariance_and_portfolio_variance() -> None:
    print("\n" + "=" * 60)
    print("COVARIANCE AND PORTFOLIO VARIANCE")
    print("=" * 60)

    np.random.seed(7)
    ret_matrix = np.random.normal(0, 0.015, (252, 3))   # 252 days x 3 assets
    cov = np.cov(ret_matrix.T)                           # 3x3 covariance matrix
    weights = np.array([0.5, 0.3, 0.2])

    portfolio_var = weights @ cov @ weights               # quadratic form
    portfolio_vol = np.sqrt(portfolio_var * 252)         # annualised

    print(f"Annualised covariance matrix:\n{np.round(cov * 252, 6)}")
    print(f"\nWeights:             {weights}")
    print(f"Portfolio variance:  {portfolio_var:.8f}")
    print(f"Annual portfolio vol:{portfolio_vol:.4%}")


def boolean_indexing_and_pnl() -> None:
    print("\n" + "=" * 60)
    print("BOOLEAN INDEXING — DAILY P&L ANALYSIS")
    print("=" * 60)

    np.random.seed(1)
    daily_pnl = np.random.normal(500, 2000, 252)

    winning = daily_pnl[daily_pnl >= 0]
    losing = daily_pnl[daily_pnl < 0]

    print(f"Total trading days:  {len(daily_pnl)}")
    print(f"Winning days:        {len(winning)}")
    print(f"Losing days:         {len(losing)}")
    print(f"Win rate:            {len(winning) / len(daily_pnl):.2%}")
    print(f"Avg win:             ${winning.mean():,.2f}")
    print(f"Avg loss:            ${losing.mean():,.2f}")
    print(f"Profit factor:       {winning.sum() / abs(losing.sum()):.3f}")


def main() -> None:
    intro()
    arrays_and_dtypes()
    vectorised_returns()
    descriptive_statistics()
    broadcasting_demo()
    covariance_and_portfolio_variance()
    boolean_indexing_and_pnl()
    print(
        "\n\U0001f389 NumPy tutorial complete! "
        "Next: explore the Pandas tutorial for time-series data manipulation."
    )


if __name__ == "__main__":
    main()
