"""Interactive Portfolio Optimization Tutorial – Markowitz & Beyond.

Run with:
    python portfolio_tutorial.py

Covers Modern Portfolio Theory (Markowitz), the efficient frontier,
Sharpe ratio maximization, diversification, and practical portfolio
construction. Each section ends with an interactive quiz question.
"""

from __future__ import annotations

import math
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BORDER = "=" * 70


def _ask(question: str, choices: List[str], correct: int, explanation: str) -> None:
    print(f"\n  Q: {question}")
    for i, c in enumerate(choices):
        print(f"     {chr(65 + i)}) {c}")
    while True:
        raw = input("  Your answer (A/B/C/D): ").strip().upper()
        if raw and raw[0] in "ABCD" and ord(raw[0]) - 65 < len(choices):
            break
        print("  Please enter A, B, C, or D.")
    chosen = ord(raw[0]) - 65
    if chosen == correct:
        print("  Correct!")
    else:
        print(f"  Not quite. The answer is {chr(65 + correct)}.")
    print(f"  Explanation: {explanation}\n")


def _header(title: str) -> None:
    print("\n" + BORDER)
    print(title.upper())
    print(BORDER)


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _portfolio_stats(
    weights: List[float],
    means: List[float],
    cov_matrix: List[List[float]],
) -> Tuple[float, float]:
    """Return (expected_return, portfolio_std) for given weights."""
    ret = _dot(weights, means)
    variance = 0.0
    for i, wi in enumerate(weights):
        for j, wj in enumerate(weights):
            variance += wi * wj * cov_matrix[i][j]
    return ret, math.sqrt(max(variance, 0.0))


def _sharpe(ret: float, std: float, rf: float = 0.02 / 252) -> float:
    if std == 0:
        return 0.0
    return (ret - rf) / std


# ---------------------------------------------------------------------------
# Section 1 – Portfolio Return & Risk
# ---------------------------------------------------------------------------


def section_portfolio_basics() -> None:
    _header("Section 1 – Portfolio Return & Risk")

    print(
        """
A PORTFOLIO is a collection of assets held in certain proportions (weights).

PORTFOLIO RETURN (expected)
  E[R_p] = sum(w_i * E[R_i])
  Weighted average of individual expected returns.

PORTFOLIO RISK (variance)
  Var(R_p) = sum_i sum_j  w_i * w_j * Cov(R_i, R_j)

  This is NOT a simple weighted average. The cross-terms (covariances)
  mean that combining assets can REDUCE total risk below the weighted sum.
  This is the mathematical basis for DIVERSIFICATION.

PORTFOLIO VOLATILITY
  sigma_p = sqrt(Var(R_p))
"""
    )

    # Two-asset example
    w_a, w_b = 0.6, 0.4
    mu_a, mu_b = 0.10, 0.15  # annual expected returns
    sigma_a, sigma_b = 0.12, 0.20  # annual volatilities
    corr_ab = 0.3  # correlation

    cov_ab = corr_ab * sigma_a * sigma_b
    port_return = w_a * mu_a + w_b * mu_b
    port_var = w_a**2 * sigma_a**2 + w_b**2 * sigma_b**2 + 2 * w_a * w_b * cov_ab
    port_vol = math.sqrt(port_var)

    # Weighted average vol (what you'd get with corr=1)
    weighted_vol = w_a * sigma_a + w_b * sigma_b

    print(f"  Asset A: E[R]={mu_a:.0%}, sigma={sigma_a:.0%}")
    print(f"  Asset B: E[R]={mu_b:.0%}, sigma={sigma_b:.0%}")
    print("  Weights: 60% A, 40% B")
    print(f"  Correlation(A,B): {corr_ab:.1f}")
    print(f"\n  Portfolio E[R]   = {w_a:.1f}*{mu_a:.0%} + {w_b:.1f}*{mu_b:.0%} = {port_return:.2%}")
    print(f"  Portfolio Var    = {port_var:.6f}")
    print(f"  Portfolio Vol    = {port_vol:.4f} ({port_vol:.2%})")
    print(f"\n  Weighted avg vol = {weighted_vol:.4f} ({weighted_vol:.2%})")
    print(f"  Diversification benefit: {(weighted_vol - port_vol):.4f} ({(weighted_vol - port_vol):.2%} reduction)")

    print(
        f"""
The portfolio volatility ({port_vol:.2%}) is LESS than the weighted average ({weighted_vol:.2%}).
This reduction is DIVERSIFICATION at work — combining assets with correlation
< 1 reduces total portfolio risk without necessarily reducing expected return.
"""
    )

    _ask(
        "You combine two assets with correlation = -1.0 in the right proportions. The portfolio volatility will be:",
        [
            "The average of the two individual volatilities",
            "Greater than either individual volatility",
            "Zero — perfect negative correlation enables complete risk elimination",
            "Equal to the higher of the two volatilities",
        ],
        2,
        "With perfect negative correlation (rho = -1), there exist weights that make "
        "the portfolio variance exactly zero. The cross-term 2*w_a*w_b*Cov becomes "
        "maximally negative and cancels the variance terms completely. "
        "In practice, rho = -1 is theoretical — but low/negative correlation still "
        "provides substantial diversification benefits.",
    )


# ---------------------------------------------------------------------------
# Section 2 – The Efficient Frontier
# ---------------------------------------------------------------------------


def section_efficient_frontier() -> None:
    _header("Section 2 – The Efficient Frontier")

    print(
        """
MARKOWITZ MEAN-VARIANCE FRAMEWORK (1952)

Harry Markowitz showed that investors should care only about:
  1. Expected return  (maximize it)
  2. Portfolio variance (minimize it)

For a given level of risk, there is a MAXIMUM achievable return.
The set of such optimal portfolios is the EFFICIENT FRONTIER.

EFFICIENT FRONTIER PROPERTIES
  - No rational investor holds a portfolio BELOW the frontier (dominated)
  - Portfolios on the frontier cannot increase return without increasing risk
  - The frontier is a curve in (sigma, return) space
  - The GLOBAL MINIMUM VARIANCE (GMV) portfolio is the leftmost point

CAPITAL MARKET LINE (CML)
  When a risk-free asset is added, the optimal portfolio is on the
  straight line tangent to the efficient frontier — the TANGENCY PORTFOLIO.
  This tangency portfolio has the highest Sharpe ratio of all portfolios.
"""
    )

    # Sweep weights and show return/risk for two assets
    mu_a, mu_b = 0.08, 0.14
    sigma_a, sigma_b = 0.10, 0.20
    corr = 0.2
    cov_ab = corr * sigma_a * sigma_b
    rf = 0.03  # annual risk-free

    print(f"\n  Asset A: E[R]={mu_a:.0%}, sigma={sigma_a:.0%}")
    print(f"  Asset B: E[R]={mu_b:.0%}, sigma={sigma_b:.0%}")
    print(f"  Correlation(A,B): {corr:.1f}")
    print(f"  Risk-free rate: {rf:.0%}")
    print()
    print(f"  {'Weight_A':>8}  {'Weight_B':>8}  {'Return':>8}  {'Vol':>8}  {'Sharpe':>8}")
    print("  " + "-" * 52)

    best_sharpe = -999.0
    best_weights = (0.0, 0.0)
    best_ret = 0.0
    best_vol = 0.0

    for i in range(11):
        wa = i / 10.0
        wb = 1.0 - wa
        ret = wa * mu_a + wb * mu_b
        var = wa**2 * sigma_a**2 + wb**2 * sigma_b**2 + 2 * wa * wb * cov_ab
        vol = math.sqrt(var)
        sr = (ret - rf) / vol
        marker = " <-- Max Sharpe" if sr > best_sharpe else ""
        if sr > best_sharpe:
            best_sharpe = sr
            best_weights = (wa, wb)
            best_ret = ret
            best_vol = vol
        print(f"  {wa:>8.0%}  {wb:>8.0%}  {ret:>8.2%}  {vol:>8.2%}  {sr:>8.4f}{marker}")

    print(f"\n  Optimal (max Sharpe) portfolio: {best_weights[0]:.0%} A, {best_weights[1]:.0%} B")
    print(f"  Return={best_ret:.2%}, Vol={best_vol:.2%}, Sharpe={best_sharpe:.4f}")

    _ask(
        "The Global Minimum Variance (GMV) portfolio is the portfolio with:",
        [
            "The highest expected return on the efficient frontier",
            "The highest Sharpe ratio",
            "The lowest possible portfolio volatility",
            "Equal weights across all assets",
        ],
        2,
        "The GMV portfolio minimises variance (risk) without regard to return. "
        "It is the leftmost point on the efficient frontier in risk-return space. "
        "It does NOT have the highest Sharpe ratio — the tangency portfolio does.",
    )


# ---------------------------------------------------------------------------
# Section 3 – Sharpe Ratio Maximization
# ---------------------------------------------------------------------------


def section_sharpe_max() -> None:
    _header("Section 3 – Sharpe Ratio Maximization")

    print(
        """
THE TANGENCY PORTFOLIO
  The portfolio with the highest Sharpe ratio lies at the tangency point
  between the Capital Market Line (CML) and the efficient frontier.

  All rational investors (under CAPM assumptions) hold:
  - A mix of the tangency portfolio (risky assets)
  - The risk-free asset (cash or T-bills)

  How much to hold of each depends on risk tolerance.

ANALYTICAL SOLUTION (two assets)
  The tangency portfolio maximises: (E[R_p] - rf) / sigma_p

  For two assets, optimal weights can be found analytically or by
  numerical optimisation (e.g., gradient descent, scipy.minimize).

PRACTICAL CONSTRAINTS
  In practice, portfolios have additional constraints:
  - Long-only: w_i >= 0 (no short selling)
  - Weight bounds: w_i in [0.05, 0.40] (diversification limits)
  - Sector limits: sum of tech stocks <= 30%
  - Transaction costs: rebalancing has costs
"""
    )

    # Three-asset example
    assets = ["SPY", "TLT", "GLD"]
    means = [0.10, 0.04, 0.06]  # annual expected returns
    vols = [0.15, 0.08, 0.12]

    # Correlation matrix (approximate)
    corr_matrix = [
        [1.00, -0.30, 0.05],
        [-0.30, 1.00, 0.10],
        [0.05, 0.10, 1.00],
    ]

    # Build covariance matrix
    cov = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            cov[i][j] = corr_matrix[i][j] * vols[i] * vols[j]

    rf = 0.04  # annual risk-free

    print(f"\n  Assets: {assets}")
    print(f"  Expected returns: {[f'{m:.0%}' for m in means]}")
    print(f"  Volatilities:     {[f'{v:.0%}' for v in vols]}")
    print(f"  Risk-free rate:   {rf:.0%}")
    print()

    # Grid search over weights (step 10%)
    best_sharpe = -999.0
    best_w = (0.0, 0.0, 0.0)
    best_ret = 0.0
    best_vol = 0.0

    print(f"  {'SPY':>6}  {'TLT':>6}  {'GLD':>6}  {'Return':>8}  {'Vol':>8}  {'Sharpe':>8}")
    print("  " + "-" * 56)
    top_rows = []
    for ia in range(0, 11, 2):
        for ib in range(0, 11 - ia, 2):
            ic = 10 - ia - ib
            wa, wb, wc = ia / 10, ib / 10, ic / 10
            w = [wa, wb, wc]
            ret, vol = _portfolio_stats(w, means, cov)
            sr = (ret - rf) / vol
            top_rows.append((sr, wa, wb, wc, ret, vol))
            if sr > best_sharpe:
                best_sharpe = sr
                best_w = (wa, wb, wc)
                best_ret = ret
                best_vol = vol

    # Show best 5
    top_rows.sort(reverse=True)
    for sr, wa, wb, wc, ret, vol in top_rows[:6]:
        marker = " <-- Best" if (wa, wb, wc) == best_w else ""
        print(f"  {wa:>6.0%}  {wb:>6.0%}  {wc:>6.0%}  {ret:>8.2%}  {vol:>8.2%}  {sr:>8.4f}{marker}")

    print(f"\n  Best portfolio: SPY={best_w[0]:.0%}, TLT={best_w[1]:.0%}, GLD={best_w[2]:.0%}")
    print(f"  Return={best_ret:.2%}, Vol={best_vol:.2%}, Sharpe={best_sharpe:.4f}")

    _ask(
        "An investor maximises Sharpe ratio to find their tangency portfolio. "
        "They then invest 60% in that portfolio and 40% in T-bills. Compared to "
        "the tangency portfolio, their combined portfolio has:",
        [
            "Higher return and higher Sharpe ratio",
            "Lower return and lower Sharpe ratio — but the same Sharpe ratio as the tangency",
            "The same return but lower volatility",
            "Lower Sharpe ratio because holding cash hurts risk-adjusted returns",
        ],
        1,
        "Mixing the tangency portfolio with the risk-free asset moves along the CML. "
        "Return and volatility both decrease proportionally, so Sharpe ratio stays the same "
        "(the CML is a straight line through the risk-free rate and tangency portfolio). "
        "The investor gets a lower-risk version with identical risk-adjusted return.",
    )


# ---------------------------------------------------------------------------
# Section 4 – Diversification & Limitations
# ---------------------------------------------------------------------------


def section_diversification() -> None:
    _header("Section 4 – Diversification & Limitations of MPT")

    print(
        """
DIVERSIFICATION: HOW MANY STOCKS DO YOU NEED?

Adding assets to a portfolio reduces idiosyncratic (stock-specific) risk.
However, SYSTEMATIC (market) risk cannot be diversified away.

  Total risk = Systematic risk + Idiosyncratic risk

Research shows:
  - ~10-20 stocks eliminate most idiosyncratic risk
  - Beyond ~30 stocks, additional diversification benefit is minimal
  - Remaining risk is market beta (correlation with the overall market)

LIMITATIONS OF MARKOWITZ MPT
  1. Input sensitivity  – Small changes in expected returns dramatically
                          change the optimal portfolio (estimation error)
  2. Estimation error   – Historical means/covariances are noisy predictors
  3. Normal assumption  – Returns are NOT normally distributed (fat tails)
  4. Static model       – MPT is single-period; real portfolios rebalance
  5. Correlation breaks – In crises, correlations spike toward 1.0, exactly
                          when diversification is most needed
  6. Transaction costs  – Frequent rebalancing is expensive

PRACTICAL IMPROVEMENTS
  - Black-Litterman model  – Combines MPT with investor views
  - Robust optimisation    – Accounts for estimation uncertainty
  - Risk parity            – Equal risk contribution from each asset
  - Factor investing       – Diversify across risk factors, not just assets
"""
    )

    # Show idiosyncratic risk reduction
    annual_market_var = 0.04  # market variance
    annual_stock_idio_var = 0.10  # average idiosyncratic variance per stock

    print(f"\n  Market variance (systematic, cannot diversify): {annual_market_var:.2f}")
    print(f"  Single-stock idiosyncratic variance: {annual_stock_idio_var:.2f}")
    print()
    print(f"  {'n_stocks':>8}  {'Idio Var':>10}  {'Total Var':>10}  {'Total Vol':>10}")
    print("  " + "-" * 44)
    for n in [1, 5, 10, 20, 30, 50, 100]:
        idio_var = annual_stock_idio_var / n
        total_var = annual_market_var + idio_var
        total_vol = math.sqrt(total_var)
        print(f"  {n:>8}  {idio_var:>10.4f}  {total_var:>10.4f}  {total_vol:>10.4f}")

    print(
        f"""
Notice: Portfolio volatility drops sharply from n=1 to n=20, then levels off.
The floor is sqrt(market_variance) = {math.sqrt(annual_market_var):.4f} — systematic risk you cannot escape.
"""
    )

    _ask(
        "Markowitz MPT relies on historical mean returns to determine optimal weights. "
        "What is the main practical problem with this approach?",
        [
            "It is computationally too slow for large portfolios",
            "Historical means are poor predictors of future returns, causing the "
            "portfolio to tilt heavily toward recently outperforming assets",
            "MPT requires all assets to be uncorrelated, which is never true",
            "The efficient frontier cannot be computed with more than 10 assets",
        ],
        1,
        "Mean estimation error is the dominant problem in MPT. Small errors in "
        "expected returns produce wildly different 'optimal' portfolios because the "
        "optimiser over-fits to noisy signals. The Black-Litterman model and shrinkage "
        "estimators help address this by blending historical data with prior beliefs.",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print("\n" + BORDER)
    print("PORTFOLIO OPTIMIZATION – INTERACTIVE TUTORIAL")
    print(BORDER)
    print(
        """
Welcome! This tutorial walks through Markowitz Modern Portfolio Theory
and the practical tools used to construct and evaluate portfolios:

  1. Portfolio Return & Risk  – weighted return, variance, diversification
  2. The Efficient Frontier   – Markowitz framework, optimal portfolios
  3. Sharpe Ratio Maximisation – tangency portfolio, Capital Market Line
  4. Diversification & Limits  – how many stocks, MPT pitfalls

Press ENTER to begin each section.
"""
    )

    sections = [
        ("Portfolio Return & Risk", section_portfolio_basics),
        ("The Efficient Frontier", section_efficient_frontier),
        ("Sharpe Ratio Maximization", section_sharpe_max),
        ("Diversification & Limitations", section_diversification),
    ]

    for title, fn in sections:
        input(f"Press ENTER to start: {title} ...")
        fn()

    print("\n" + BORDER)
    print("TUTORIAL COMPLETE")
    print(BORDER)
    print(
        """
Topics covered:
  1. Portfolio Return & Risk   – weighted mean, covariance-based variance
  2. Efficient Frontier        – Markowitz, GMV portfolio, frontier shape
  3. Sharpe Maximisation       – tangency portfolio, CML, grid search
  4. Diversification           – systematic vs idiosyncratic risk, MPT limits

Recommended next steps:
  -> UTILS - Risk Metrics                              (VaR, CVaR, drawdown)
  -> UTILS - Quantitative Methods - Factor Models     (Fama-French, factor risk)
  -> UTILS - Quantitative Methods - Statistics        (statistical underpinnings)
  -> UTILS - Finance - Beta Calculator                (systematic risk measurement)
"""
    )


if __name__ == "__main__":
    main()
