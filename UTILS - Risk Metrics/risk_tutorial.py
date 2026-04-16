"""Interactive Risk Metrics Tutorial – VaR, CVaR, Drawdown & More.

Run with:
    python risk_tutorial.py

Covers Value at Risk, Conditional VaR (Expected Shortfall), maximum
drawdown, Sharpe/Sortino ratios, and portfolio risk decomposition.
Each section ends with an interactive quiz question.
"""

from __future__ import annotations

import math
import statistics
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BORDER = "=" * 70


def _norm_cdf(x: float) -> float:
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def _norm_ppf(p: float) -> float:
    """Approximate inverse normal CDF using rational approximation."""
    # Abramowitz and Stegun approximation
    if p <= 0 or p >= 1:
        raise ValueError("p must be in (0, 1)")
    sign = 1.0 if p >= 0.5 else -1.0
    q = min(p, 1 - p)
    t = math.sqrt(-2.0 * math.log(q))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    num = c0 + c1 * t + c2 * t * t
    den = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t
    return sign * (t - num / den)


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


# ---------------------------------------------------------------------------
# Section 1 – Value at Risk (VaR)
# ---------------------------------------------------------------------------


def _historical_var(returns: List[float], confidence: float) -> float:
    sorted_r = sorted(returns)
    idx = int((1 - confidence) * len(sorted_r))
    return -sorted_r[max(idx - 1, 0)]


def _parametric_var(mu: float, sigma: float, confidence: float, horizon_days: int = 1) -> float:
    z = _norm_ppf(1 - confidence)
    return -(mu * horizon_days + z * sigma * math.sqrt(horizon_days))


def section_var() -> None:
    _header("Section 1 – Value at Risk (VaR)")

    print(
        """
VALUE AT RISK (VaR) answers:
  "What is the maximum loss I can expect with X% confidence over Y days?"

EXAMPLE: 1-day 95% VaR = $50,000 means:
  There is only a 5% chance of losing MORE than $50,000 in a single day.
  (Equivalently, 95% of days the loss will be LESS than $50,000.)

THREE COMMON METHODS

1. HISTORICAL SIMULATION
   Sort past returns, take the (1-confidence) percentile.
   Pros: captures fat tails, no distributional assumption.
   Cons: relies entirely on historical data.

2. PARAMETRIC (Variance-Covariance)
   Assumes returns are normally distributed.
   VaR = -(mu * T + z * sigma * sqrt(T))
   where z is the standard normal quantile at (1-confidence).
   Pros: fast, analytical. Cons: underestimates fat tails.

3. MONTE CARLO
   Simulate thousands of return paths, take the percentile.
   Pros: handles complex portfolios. Cons: computationally expensive.
"""
    )

    # Historical VaR
    returns = [
        0.012, -0.005, 0.023, -0.018, 0.007, 0.031, -0.009, 0.015, -0.003,
        0.019, -0.024, 0.008, -0.031, 0.016, -0.011, 0.022, -0.007, 0.013,
        -0.028, 0.009, 0.017, -0.006, 0.011, -0.034, 0.020,
    ]
    portfolio_value = 1_000_000.0
    conf = 0.95

    hist_var = _historical_var(returns, conf)
    mu = statistics.mean(returns)
    sigma = statistics.stdev(returns)
    param_var = _parametric_var(mu, sigma, conf)

    print(f"  Portfolio: ${portfolio_value:,.0f}")
    print(f"  {len(returns)} days of returns (mean={mu:.4f}, sigma={sigma:.4f})")
    print(f"\n  Historical VaR (95%)  = {hist_var:.4f}  (${hist_var * portfolio_value:,.2f})")
    print(f"  Parametric VaR (95%)  = {param_var:.4f}  (${param_var * portfolio_value:,.2f})")

    print(
        """
SCALING VAR ACROSS HORIZONS
  VaR scales with sqrt(T) under normality (square-root-of-time rule):
  10-day VaR ≈ 1-day VaR * sqrt(10)

  This approximation breaks down for:
  - Fat-tailed distributions
  - Autocorrelated (trending) returns
"""
    )

    scale = math.sqrt(10)
    print(f"  1-day Parametric VaR  = {param_var:.4f}  (${param_var * portfolio_value:,.2f})")
    print(f"  10-day VaR (scaled)   = {param_var * scale:.4f}  (${param_var * scale * portfolio_value:,.2f})")

    _ask(
        "A portfolio has 1-day 99% VaR of $200,000. What does this mean?",
        [
            "The portfolio will lose exactly $200,000 on 1% of days",
            "There is a 99% chance of losing at most $200,000 in a day",
            "The portfolio loses an average of $200,000 per day",
            "The maximum possible daily loss is $200,000",
        ],
        1,
        "99% VaR means that 99% of days the loss will be $200,000 or less. "
        "On the remaining 1% of days, losses EXCEED $200,000. "
        "VaR does NOT cap the loss — it is a confidence-level threshold, not a maximum.",
    )


# ---------------------------------------------------------------------------
# Section 2 – Conditional VaR (Expected Shortfall)
# ---------------------------------------------------------------------------


def section_cvar() -> None:
    _header("Section 2 – Conditional VaR / Expected Shortfall (CVaR)")

    print(
        """
PROBLEM WITH VaR
  VaR tells you the threshold but says NOTHING about how bad losses are
  beyond that threshold. Two portfolios can have the same 95% VaR but
  very different tail behaviour.

CONDITIONAL VaR (CVaR) – also called EXPECTED SHORTFALL (ES)
  CVaR = average loss GIVEN that the loss exceeds the VaR threshold.

  "On the worst 5% of days, how much do we lose on average?"

FORMULA (historical)
  CVaR = average of all returns worse than the VaR percentile

CVaR is generally preferred by regulators (Basel III/IV, FRTB) because
it is a COHERENT risk measure — it satisfies sub-additivity, meaning
diversified portfolios always show lower risk than the sum of their parts.
VaR is NOT always coherent.
"""
    )

    returns = [
        0.012, -0.005, 0.023, -0.018, 0.007, 0.031, -0.009, 0.015, -0.003,
        0.019, -0.024, 0.008, -0.031, 0.016, -0.011, 0.022, -0.007, 0.013,
        -0.028, 0.009, 0.017, -0.006, 0.011, -0.034, 0.020,
    ]
    portfolio_value = 1_000_000.0
    conf = 0.95

    sorted_r = sorted(returns)
    cutoff_idx = int((1 - conf) * len(sorted_r))
    tail_returns = sorted_r[:max(cutoff_idx, 1)]
    cvar = -statistics.mean(tail_returns)
    var = _historical_var(returns, conf)

    print(f"  Portfolio: ${portfolio_value:,.0f}, {conf:.0%} confidence")
    print(f"\n  Historical VaR  (95%) = {var:.4f}  (${var * portfolio_value:,.2f})")
    print(f"  Historical CVaR (95%) = {cvar:.4f}  (${cvar * portfolio_value:,.2f})")
    print(f"\n  Tail returns driving CVaR: {[f'{r:.4f}' for r in tail_returns]}")
    print(
        f"\n  CVaR > VaR by ${(cvar - var) * portfolio_value:,.2f} — this extra amount"
        " represents the average excess loss beyond the VaR threshold."
    )

    print(
        """
REGULATORY USE
  - Basel III moved from VaR to CVaR (Expected Shortfall) for market risk
  - CVaR penalises fat tails more severely
  - A strategy that looks OK by VaR may look much worse under CVaR
"""
    )

    _ask(
        "Portfolio A: 95% VaR = $100k, CVaR = $110k. "
        "Portfolio B: 95% VaR = $100k, CVaR = $300k. "
        "Both have the same VaR. Which portfolio has worse tail risk?",
        [
            "Portfolio A — it has a lower CVaR",
            "Portfolio B — on the worst 5% of days, average loss is $300k vs $110k",
            "They are identical — same VaR means same risk",
            "Cannot compare without knowing the full distribution",
        ],
        1,
        "Same VaR but Portfolio B's CVaR is $300k vs $110k for A. "
        "CVaR captures how bad the tail losses are. Portfolio B has much more severe "
        "losses on its worst days — greater tail risk despite identical VaR.",
    )


# ---------------------------------------------------------------------------
# Section 3 – Drawdown Analysis
# ---------------------------------------------------------------------------


def _drawdown_series(equity_curve: List[float]) -> Tuple[float, int, int]:
    """Return (max_drawdown, peak_idx, trough_idx) from an equity curve."""
    peak = equity_curve[0]
    peak_idx = 0
    max_dd = 0.0
    trough_idx = 0
    best_peak_idx = 0

    for i, val in enumerate(equity_curve):
        if val > peak:
            peak = val
            peak_idx = i
        dd = (peak - val) / peak
        if dd > max_dd:
            max_dd = dd
            trough_idx = i
            best_peak_idx = peak_idx

    return max_dd, best_peak_idx, trough_idx


def section_drawdown() -> None:
    _header("Section 3 – Drawdown Analysis")

    print(
        """
DRAWDOWN measures the peak-to-trough decline of a portfolio's equity curve.

  Drawdown(t) = (Peak_value_up_to_t - Current_value) / Peak_value_up_to_t

MAXIMUM DRAWDOWN (MDD)
  The largest single peak-to-trough decline across the entire history.
  A key measure of how painful the strategy was to hold during bad periods.

DRAWDOWN DURATION
  How long (in days/months) the portfolio stayed below its previous peak.
  Investors often abandon strategies during long drawdown periods.

CALMAR RATIO
  Annual return / Maximum Drawdown
  Higher = better. Measures return per unit of drawdown risk.
  A Calmar > 1 is generally considered good.
"""
    )

    equity = [
        100.0, 103.0, 107.0, 105.0, 110.0, 115.0, 112.0, 108.0,
        104.0, 100.0, 97.0, 102.0, 106.0, 111.0, 116.0, 120.0,
    ]
    dates = [f"Day {i+1:>2}" for i in range(len(equity))]

    max_dd, peak_i, trough_i = _drawdown_series(equity)
    total_return = (equity[-1] - equity[0]) / equity[0]
    # Annualised return assuming 252 trading days
    years = len(equity) / 252.0
    ann_return = (1 + total_return) ** (1 / years) - 1
    calmar = ann_return / max_dd if max_dd > 0 else float("inf")

    print(f"  Equity curve: {equity}")
    print(f"\n  Start value  : ${equity[0]:.2f}")
    print(f"  End value    : ${equity[-1]:.2f}")
    print(f"  Total return : {total_return:.2%}")
    print(f"\n  Peak at {dates[peak_i]}: ${equity[peak_i]:.2f}")
    print(f"  Trough at {dates[trough_i]}: ${equity[trough_i]:.2f}")
    print(f"  Maximum Drawdown: {max_dd:.2%}")
    print(f"  Drawdown duration: {trough_i - peak_i} days")
    print(f"\n  Annualised return (scaled to 252 days): {ann_return:.2%}")
    print(f"  Calmar Ratio: {calmar:.2f}")

    _ask(
        "A strategy peaks at $200k, falls to $160k before recovering. "
        "What is the maximum drawdown?",
        [
            "20% — (200k - 160k) / 200k",
            "40k — the dollar loss from peak to trough",
            "25% — (200k - 160k) / 160k",
            "80% — (160k / 200k)",
        ],
        0,
        "Max drawdown = (peak - trough) / peak = (200k - 160k) / 200k = 40k / 200k = 20%. "
        "Always divide by the PEAK value, not the trough.",
    )


# ---------------------------------------------------------------------------
# Section 4 – Sharpe & Sortino Ratios
# ---------------------------------------------------------------------------


def _sharpe(ret: float, std: float, rf: float = 0.0002) -> float:
    """Annualised Sharpe ratio given periodic return, std dev, and risk-free rate."""
    if std == 0:
        return 0.0
    return (ret - rf) / std


def section_sharpe_sortino() -> None:
    _header("Section 4 – Sharpe & Sortino Ratios")

    print(
        """
SHARPE RATIO measures risk-adjusted return:
  Sharpe = (Mean return - Risk-free rate) / Std deviation of returns

  Higher is better. A Sharpe > 1.0 is good; > 2.0 is excellent.
  Drawback: std dev penalises BOTH upside and downside volatility equally.

SORTINO RATIO fixes this by penalising ONLY downside volatility:
  Sortino = (Mean return - Target return) / Downside deviation

  Downside deviation = std dev of returns BELOW the target (e.g., 0)
  Higher is better. More relevant for strategies with positive skew.

WHEN TO USE EACH
  - Symmetric return distributions → Sharpe and Sortino tell similar stories
  - Positively skewed strategies   → Sortino is more favourable (and more honest)
  - Negatively skewed strategies   → Sortino may look worse than Sharpe
"""
    )

    rf = 0.0002  # daily risk-free (~5% annual / 252)
    target = 0.0  # minimum acceptable return

    returns = [
        0.012, -0.005, 0.023, -0.018, 0.007, 0.031, -0.009, 0.015, -0.003,
        0.019, -0.024, 0.008, -0.031, 0.016, -0.011, 0.022, -0.007, 0.013,
    ]

    mean_r = statistics.mean(returns)
    std_r = statistics.stdev(returns)

    downside = [r for r in returns if r < target]
    downside_dev = statistics.stdev(downside) if len(downside) > 1 else std_r

    sharpe = (mean_r - rf) / std_r
    sortino = (mean_r - target) / downside_dev

    # Annualise (multiply by sqrt(252) for daily data)
    sharpe_ann = sharpe * math.sqrt(252)
    sortino_ann = sortino * math.sqrt(252)

    print(f"  {len(returns)} daily returns | Mean={mean_r:.4f} | Std={std_r:.4f}")
    print(f"  Daily risk-free rate  = {rf:.4f}")
    print(f"  Downside deviation    = {downside_dev:.4f} ({len(downside)} negative days)")
    print(f"\n  Daily Sharpe  = ({mean_r:.4f} - {rf:.4f}) / {std_r:.4f} = {sharpe:.4f}")
    print(f"  Daily Sortino = ({mean_r:.4f} - {target:.4f}) / {downside_dev:.4f} = {sortino:.4f}")
    print(f"\n  Annualised Sharpe  = {sharpe_ann:.4f}  (daily * sqrt(252))")
    print(f"  Annualised Sortino = {sortino_ann:.4f}")

    _ask(
        "Two strategies both have Sharpe = 1.2. Strategy A has Sortino = 1.5, "
        "Strategy B has Sortino = 0.9. Which should you prefer?",
        [
            "Strategy B — lower Sortino means more downside cushion",
            "Strategy A — higher Sortino means its gains outweigh its downside losses",
            "They are equal — same Sharpe means same risk-adjusted performance",
            "Cannot determine without knowing the raw returns",
        ],
        1,
        "Strategy A's higher Sortino (1.5 vs 0.9) means its downside volatility "
        "is lower relative to its mean return. Strategy A generates the same "
        "Sharpe-adjusted return but with less downside risk — prefer A.",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print("\n" + BORDER)
    print("RISK METRICS – INTERACTIVE TUTORIAL")
    print(BORDER)
    print(
        """
Welcome! This tutorial walks through the core risk metrics used by
professional portfolio managers, quants, and risk officers:

  1. Value at Risk (VaR)         – loss threshold at a confidence level
  2. Conditional VaR (CVaR)      – average loss beyond the VaR threshold
  3. Drawdown Analysis           – peak-to-trough pain metrics
  4. Sharpe & Sortino Ratios     – risk-adjusted return measurement

Press ENTER to begin each section.
"""
    )

    sections = [
        ("Value at Risk", section_var),
        ("Conditional VaR / Expected Shortfall", section_cvar),
        ("Drawdown Analysis", section_drawdown),
        ("Sharpe & Sortino Ratios", section_sharpe_sortino),
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
  1. Value at Risk       – historical, parametric, scaling
  2. CVaR/Expected Shortfall – coherent risk, tail averages
  3. Drawdown            – MDD, duration, Calmar ratio
  4. Sharpe & Sortino    – risk-adjusted return ratios

Recommended next steps:
  -> UTILS - Quantitative Methods - Statistics      (distribution theory behind VaR)
  -> UTILS - Quantitative Methods - Stochastic Processes  (Monte Carlo VaR)
  -> UTILS - Portfolio Optimizer                    (apply risk metrics to portfolios)
"""
    )


if __name__ == "__main__":
    main()
