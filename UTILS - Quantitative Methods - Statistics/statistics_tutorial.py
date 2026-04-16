"""Interactive Statistics Tutorial for Quantitative Finance.

Run with:
    python statistics_tutorial.py

Covers normal distribution, Z-scores, hypothesis testing, correlation,
skewness, kurtosis, and how each concept applies to financial data.
Each section ends with a short quiz.
"""

from __future__ import annotations

import math
import statistics
from typing import List

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BORDER = "=" * 70
THIN = "-" * 70


def _ask(question: str, choices: List[str], correct: int, explanation: str) -> None:
    """Present a multiple-choice question and give immediate feedback."""
    print(f"\n  Q: {question}")
    for i, choice in enumerate(choices):
        print(f"     {chr(65 + i)}) {choice}")
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
# Section 1 – Descriptive Statistics
# ---------------------------------------------------------------------------


def section_descriptive() -> None:
    _header("Section 1 – Descriptive Statistics")

    daily_returns = [0.012, -0.005, 0.023, -0.018, 0.007, 0.031, -0.009, 0.015, -0.003, 0.019]
    pct = [r * 100 for r in daily_returns]

    mean_r = statistics.mean(daily_returns)
    median_r = statistics.median(daily_returns)
    stdev_r = statistics.stdev(daily_returns)
    variance_r = statistics.variance(daily_returns)

    print(
        """
A trading strategy recorded the following 10 daily returns (as decimals):

  {}

Descriptive statistics summarise the *central tendency* and *spread*.
""".format(
            ", ".join(f"{r:+.3f}" for r in daily_returns)
        )
    )

    print(f"  Mean return    : {mean_r:+.4f}  ({mean_r * 100:+.2f}%)")
    print(f"  Median return  : {median_r:+.4f}  ({median_r * 100:+.2f}%)")
    print(f"  Std deviation  : {stdev_r:.4f}  ({stdev_r * 100:.2f}%)")
    print(f"  Variance       : {variance_r:.6f}")

    print(
        """
KEY INSIGHT
-----------
- The MEAN is your expected daily gain/loss.
- The STANDARD DEVIATION measures uncertainty (risk).
- High standard deviation = high volatility = higher risk.
- A Sharpe Ratio uses mean / std to express risk-adjusted return.
"""
    )

    _ = pct  # used implicitly in the explanation below

    _ask(
        "A stock has a mean daily return of 0.05% and std dev of 1.2%. "
        "Another has mean 0.05% but std dev of 0.4%. Which is riskier?",
        [
            "Stock with std dev 1.2% – wider spread around the mean",
            "Stock with std dev 0.4% – tighter spread around the mean",
            "Both are equally risky since means are equal",
            "Cannot determine without more data",
        ],
        0,
        "Standard deviation measures dispersion of returns. A larger std dev means returns "
        "vary more wildly, so the stock is harder to predict — that is higher risk.",
    )


# ---------------------------------------------------------------------------
# Section 2 – Normal Distribution & Z-Scores
# ---------------------------------------------------------------------------


def _norm_cdf(z: float) -> float:
    """Approximate CDF of the standard normal using the math.erf function."""
    return (1.0 + math.erf(z / math.sqrt(2.0))) / 2.0


def _norm_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


def section_normal_distribution() -> None:
    _header("Section 2 – Normal Distribution & Z-Scores")

    print(
        """
The NORMAL DISTRIBUTION (bell curve) appears everywhere in finance:
  - Daily stock returns are approximately normal (with fat tails)
  - Portfolio returns under Markowitz theory are assumed normal
  - Black-Scholes options pricing assumes lognormal prices

FORMULA  PDF(x) = (1 / (sigma * sqrt(2*pi))) * exp(-0.5 * ((x - mu) / sigma)^2)

A Z-SCORE converts a raw value into standard deviations from the mean:
  z = (x - mean) / std_dev

Interpretation:
  z =  0   → exactly at the mean
  z = +1   → 1 std dev above mean  (top ~84th percentile)
  z = -1   → 1 std dev below mean  (bottom ~16th percentile)
  z = +2   → 2 std devs above mean (top ~97.7th percentile)
"""
    )

    mu, sigma = 0.0008, 0.012  # daily mean return, daily std dev
    bad_day = -0.03  # a -3% day

    z = (bad_day - mu) / sigma
    prob_below = _norm_cdf(z)

    print(f"  Example portfolio: mu = {mu:.4f}, sigma = {sigma:.4f}")
    print(f"  A -3.0% day has Z-score = ({bad_day:.4f} - {mu:.4f}) / {sigma:.4f} = {z:.2f}")
    print(f"  Probability of a return this bad or worse = {prob_below:.4f} ({prob_below * 100:.2f}%)")
    print(f"  So roughly 1 in {int(round(1 / prob_below))} trading days is this bad or worse.")

    print(
        """
RULE OF THUMB – Empirical Rule
  68% of returns fall within ±1 sigma
  95% of returns fall within ±2 sigma
  99.7% of returns fall within ±3 sigma

A return beyond ±3 sigma is rare but NOT impossible in real markets —
fat tails mean extreme events happen more often than the normal model predicts.
"""
    )

    _ask(
        "A stock's daily returns have mean = 0% and std dev = 1%. "
        "What is the Z-score of a +2.5% return?",
        [
            "0.025",
            "2.5",
            "1.25",
            "0.25",
        ],
        1,
        "Z = (x - mean) / std_dev = (0.025 - 0.000) / 0.01 = 2.5. "
        "This means the return is 2.5 standard deviations above the mean.",
    )

    _ask(
        "Approximately what percentage of normally-distributed returns fall "
        "within ±2 standard deviations of the mean?",
        [
            "68%",
            "95%",
            "99.7%",
            "50%",
        ],
        1,
        "The empirical rule: ±1 sigma = ~68%, ±2 sigma = ~95%, ±3 sigma = ~99.7%.",
    )


# ---------------------------------------------------------------------------
# Section 3 – Correlation & Covariance
# ---------------------------------------------------------------------------


def _cov(a: List[float], b: List[float]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (n - 1)


def _pearson(a: List[float], b: List[float]) -> float:
    sa = statistics.stdev(a)
    sb = statistics.stdev(b)
    if sa == 0 or sb == 0:
        return 0.0
    return _cov(a, b) / (sa * sb)


def section_correlation() -> None:
    _header("Section 3 – Correlation & Covariance")

    spy = [0.012, -0.005, 0.023, -0.018, 0.007, 0.031, -0.009, 0.015, -0.003, 0.019]
    qqq = [0.015, -0.007, 0.021, -0.016, 0.009, 0.028, -0.011, 0.018, -0.002, 0.022]
    tlt = [-0.003, 0.008, -0.006, 0.011, -0.002, -0.009, 0.006, -0.004, 0.007, -0.005]

    corr_spy_qqq = _pearson(spy, qqq)
    corr_spy_tlt = _pearson(spy, tlt)
    cov_spy_qqq = _cov(spy, qqq)

    print(
        """
COVARIANCE measures how two assets move together (sign matters):
  Positive covariance → both tend to move in the same direction
  Negative covariance → they tend to move in opposite directions
  Near zero           → movements are largely independent

CORRELATION standardises covariance to [-1, +1]:
  rho = Cov(A, B) / (sigma_A * sigma_B)

  +1.0 → perfect positive correlation (move identically)
   0.0 → no linear relationship
  -1.0 → perfect negative correlation (mirror image)

Portfolio diversification works because adding LOW or NEGATIVE
correlation assets reduces overall portfolio volatility.
"""
    )

    print(f"  10-day returns  SPY: {[f'{r:+.3f}' for r in spy]}")
    print(f"  10-day returns  QQQ: {[f'{r:+.3f}' for r in qqq]}")
    print(f"  10-day returns  TLT: {[f'{r:+.3f}' for r in tlt]}")
    print(f"\n  Cov(SPY, QQQ)  = {cov_spy_qqq:.6f}")
    print(f"  Corr(SPY, QQQ) = {corr_spy_qqq:.4f}  (equity indices move together)")
    print(f"  Corr(SPY, TLT) = {corr_spy_tlt:.4f}  (stocks vs bonds often move opposite)")

    print(
        """
PORTFOLIO VOLATILITY FORMULA (two assets):
  sigma_p^2 = w_A^2 * sigma_A^2 + w_B^2 * sigma_B^2 + 2 * w_A * w_B * Cov(A,B)

When Corr(A,B) = -1, the cross-term is maximally negative → lowest portfolio risk.
"""
    )

    _ask(
        "You hold 50% SPY and 50% TLT. TLT has negative correlation with SPY. "
        "Compared to holding 100% SPY, your portfolio volatility will be:",
        [
            "Higher, because you now hold two risky assets",
            "The same, because returns are unchanged on average",
            "Lower, because the negative correlation reduces combined variance",
            "Undefined — you need the exact correlation to say anything",
        ],
        2,
        "Negative correlation means when SPY falls, TLT tends to rise, partially "
        "offsetting losses. The cross-term in the variance formula is negative, "
        "pulling down total portfolio variance.",
    )


# ---------------------------------------------------------------------------
# Section 4 – Hypothesis Testing
# ---------------------------------------------------------------------------


def section_hypothesis_testing() -> None:
    _header("Section 4 – Hypothesis Testing")

    print(
        """
Quants use hypothesis testing to decide whether observed results are due
to skill (alpha) or just random chance (luck).

FRAMEWORK
---------
H0 (null hypothesis)      : The strategy has zero edge (mean return = 0)
H1 (alternative hypothesis): The strategy has positive mean return

TEST STATISTIC  t = (sample_mean - 0) / (sample_std / sqrt(n))

If |t| > critical value, we REJECT H0 at the chosen significance level.

SIGNIFICANCE LEVELS
  alpha = 0.05  → 95% confidence (t_crit ≈ 1.96 for large n)
  alpha = 0.01  → 99% confidence (t_crit ≈ 2.576 for large n)

P-VALUE: the probability of observing a t-statistic this extreme under H0.
  p < 0.05 → statistically significant at 5% level.
"""
    )

    returns = [0.003, -0.001, 0.005, 0.002, -0.002, 0.004, 0.001, 0.003, 0.002, 0.004, 0.001, 0.003]
    n = len(returns)
    mean_r = statistics.mean(returns)
    std_r = statistics.stdev(returns)
    t_stat = mean_r / (std_r / math.sqrt(n))

    print(f"  Strategy backtest: {n} monthly returns")
    print(f"  Sample mean  = {mean_r:.4f} ({mean_r * 100:.2f}%)")
    print(f"  Sample std   = {std_r:.4f} ({std_r * 100:.2f}%)")
    print(f"  t-statistic  = {mean_r:.6f} / ({std_r:.6f} / sqrt({n})) = {t_stat:.3f}")
    print(f"\n  At alpha=0.05, critical value ≈ 2.201 (t-distribution, df={n - 1})")
    if abs(t_stat) > 2.201:
        print(f"  |t| = {abs(t_stat):.3f} > 2.201 → REJECT H0. Strategy shows significant positive returns.")
    else:
        print(f"  |t| = {abs(t_stat):.3f} ≤ 2.201 → FAIL TO REJECT H0. Results not statistically significant.")

    print(
        """
COMMON PITFALL – Data Snooping / p-Hacking
If you test 20 strategies and 1 passes at p<0.05, that is likely luck.
Use a Bonferroni correction: alpha_adjusted = alpha / num_tests
"""
    )

    _ask(
        "A backtest shows t-statistic = 1.5 with 60 months of data. "
        "The critical value at 95% confidence is ~2.0. What do you conclude?",
        [
            "Reject H0 — the strategy has proven alpha",
            "Fail to reject H0 — results are not statistically significant",
            "The strategy is definitely worthless",
            "Need more data before any test can be run",
        ],
        1,
        "t = 1.5 < 2.0 critical value. We fail to reject H0 — we cannot "
        "conclude the mean return is significantly different from zero at 95% confidence. "
        "More data might change this conclusion.",
    )


# ---------------------------------------------------------------------------
# Section 5 – Skewness & Kurtosis
# ---------------------------------------------------------------------------


def _skewness(data: List[float]) -> float:
    n = len(data)
    mu = sum(data) / n
    sigma = statistics.stdev(data)
    return (sum((x - mu) ** 3 for x in data) / n) / sigma**3


def _kurtosis(data: List[float]) -> float:
    """Excess kurtosis (subtract 3 to compare with normal distribution)."""
    n = len(data)
    mu = sum(data) / n
    sigma = statistics.stdev(data)
    return (sum((x - mu) ** 4 for x in data) / n) / sigma**4 - 3.0


def section_skewness_kurtosis() -> None:
    _header("Section 5 – Skewness & Kurtosis")

    print(
        """
Real financial returns are NOT perfectly normal. Two statistics quantify
the deviation from normality:

SKEWNESS  (third standardised moment)
  Positive skew → right tail is longer → occasional large gains
  Negative skew → left tail is longer → occasional large losses
  Normal distribution: skewness = 0

  Most equity return series have NEGATIVE skew (crash risk).

EXCESS KURTOSIS  (fourth standardised moment minus 3)
  Positive excess kurtosis (leptokurtic) → fatter tails than normal
  Negative excess kurtosis (platykurtic) → thinner tails
  Normal distribution: excess kurtosis = 0

  Most financial assets have POSITIVE excess kurtosis (fat tails).
  Fat tails mean extreme events happen MORE often than the normal model predicts.
  This is why VaR calculated under normality underestimates true tail risk.
"""
    )

    # Two return series: one normal-ish, one with a fat tail event
    normal_returns = [0.005, -0.003, 0.008, -0.002, 0.004, 0.006, -0.001, 0.003, 0.002, 0.005]
    fat_tail_returns = [0.005, -0.003, 0.008, -0.002, 0.004, 0.006, -0.001, 0.003, -0.065, 0.005]

    skew_n = _skewness(normal_returns)
    kurt_n = _kurtosis(normal_returns)
    skew_f = _skewness(fat_tail_returns)
    kurt_f = _kurtosis(fat_tail_returns)

    print("  Series A (no extreme events):")
    print(f"    Skewness        = {skew_n:+.4f}")
    print(f"    Excess kurtosis = {kurt_n:+.4f}")

    print("\n  Series B (one -6.5% crash day added):")
    print(f"    Skewness        = {skew_f:+.4f}  (negative = left tail heavier)")
    print(f"    Excess kurtosis = {kurt_f:+.4f}  (positive = fatter tails)")

    print(
        """
JARQUE-BERA TEST checks normality using skewness and kurtosis together.
A high JB statistic → the series is NOT normally distributed.
"""
    )

    _ask(
        "A hedge fund strategy has excess kurtosis = +4.2. Compared to a "
        "normal distribution, extreme loss days will be:",
        [
            "Less frequent — thin tails reduce the chance of outliers",
            "Equally frequent — kurtosis does not affect tail probability",
            "More frequent — positive excess kurtosis means fatter tails",
            "Only more frequent for gains, not losses",
        ],
        2,
        "Positive excess kurtosis (leptokurtic) means the distribution has "
        "heavier tails than normal. Both very large gains AND very large losses "
        "occur more often than a Gaussian model would predict.",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print("\n" + BORDER)
    print("QUANTITATIVE METHODS – STATISTICS INTERACTIVE TUTORIAL")
    print(BORDER)
    print(
        """
Welcome! This tutorial walks through five core statistics concepts that
every quant must master. Each section contains:

  - Concept explanation with finance context
  - A live worked example
  - A short quiz question to test your understanding

Press ENTER to begin each section, or type Q at any quiz to quit.
"""
    )

    sections = [
        ("Descriptive Statistics", section_descriptive),
        ("Normal Distribution & Z-Scores", section_normal_distribution),
        ("Correlation & Covariance", section_correlation),
        ("Hypothesis Testing", section_hypothesis_testing),
        ("Skewness & Kurtosis", section_skewness_kurtosis),
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
  1. Descriptive Statistics  – mean, median, std dev, variance
  2. Normal Distribution     – PDF, Z-scores, empirical rule
  3. Correlation             – covariance, Pearson correlation, diversification
  4. Hypothesis Testing      – t-test, p-values, data snooping
  5. Skewness & Kurtosis     – fat tails, negative skew, Jarque-Bera

Recommended next steps:
  -> UTILS - Quantitative Methods - Regression Analysis
  -> UTILS - Quantitative Methods - Stochastic Processes
  -> UTILS - Risk Metrics  (apply these stats to VaR and CVaR)
"""
    )


if __name__ == "__main__":
    main()
