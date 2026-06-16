"""
Bayesian Inference
------------------
A strategy wins 7 of its first 10 trades. Is its true win rate 70%? Almost
certainly not — ten trades is barely any evidence. Bayesian inference gives you
the disciplined way to answer questions like this: start with a **prior** belief,
observe **data**, and combine them into a **posterior** that quantifies what you
now believe *and* how uncertain you still are.

    posterior  proportional to  likelihood x prior

This module focuses on two **conjugate** models, where the posterior has the
same form as the prior and the update is a one-line arithmetic step:

* **Beta-Binomial** — estimating a probability (a strategy's win rate, a
  default rate) from successes and failures.
* **Normal-Normal** — estimating an unknown mean (an asset's expected return)
  with known variance, which is exactly the "shrink a noisy estimate toward a
  prior" idea behind James-Stein and Black-Litterman.

The payoff is honest uncertainty: instead of a single point estimate you get a
distribution and a **credible interval** you can actually act on.
"""

from __future__ import annotations

from typing import Tuple

from scipy import stats


def beta_binomial_update(prior_alpha: float, prior_beta: float, successes: int, failures: int) -> Tuple[float, float]:
    """Update a Beta(alpha, beta) prior with observed successes and failures.

    For a probability ``p`` with a Beta prior, observing Binomial data gives a
    Beta posterior — you simply add successes to ``alpha`` and failures to
    ``beta``. ``Beta(1, 1)`` is a flat "I know nothing" prior.

    Args:
        prior_alpha: Prior pseudo-count of successes (> 0).
        prior_beta: Prior pseudo-count of failures (> 0).
        successes: Number of observed successes.
        failures: Number of observed failures.

    Returns:
        ``(posterior_alpha, posterior_beta)``.
    """
    if prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError("Prior parameters must be positive")
    if successes < 0 or failures < 0:
        raise ValueError("Counts must be non-negative")
    return prior_alpha + successes, prior_beta + failures


def beta_mean(alpha: float, beta: float) -> float:
    """Posterior mean of a Beta(alpha, beta) — the point estimate of ``p``."""
    return alpha / (alpha + beta)


def beta_credible_interval(alpha: float, beta: float, level: float = 0.95) -> Tuple[float, float]:
    """Equal-tailed credible interval for a Beta(alpha, beta) posterior.

    Unlike a frequentist confidence interval, a 95% credible interval means
    exactly what people *think* a confidence interval means: there is a 95%
    probability the parameter lies inside it, given the prior and data.

    Args:
        alpha: Posterior alpha.
        beta: Posterior beta.
        level: Credible mass to capture (0.95 = 95%).
    """
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
    tail = (1.0 - level) / 2.0
    dist = stats.beta(alpha, beta)
    return float(dist.ppf(tail)), float(dist.ppf(1.0 - tail))


def normal_known_variance_update(
    prior_mean: float, prior_var: float, data: list, data_var: float
) -> Tuple[float, float]:
    """Bayesian update of an unknown mean with known observation variance.

    With a Normal prior on the mean and Normal data of known variance, the
    posterior mean is a **precision-weighted average** of the prior mean and the
    sample mean. The more data (or the noisier the prior), the more the
    posterior leans on the data — this is shrinkage made precise.

    Args:
        prior_mean: Prior belief about the mean.
        prior_var: Variance of that prior belief (larger = less confident).
        data: Observed samples.
        data_var: Known variance of a single observation.

    Returns:
        ``(posterior_mean, posterior_var)``.
    """
    n = len(data)
    if n == 0:
        return prior_mean, prior_var
    if prior_var <= 0 or data_var <= 0:
        raise ValueError("Variances must be positive")
    sample_mean = sum(data) / n
    prior_precision = 1.0 / prior_var
    data_precision = n / data_var
    post_precision = prior_precision + data_precision
    post_mean = (prior_precision * prior_mean + data_precision * sample_mean) / post_precision
    return post_mean, 1.0 / post_precision


def probability_greater_than(alpha: float, beta: float, threshold: float) -> float:
    """Posterior probability that ``p`` exceeds *threshold* under Beta(alpha, beta).

    Handy for decision rules like "what is the chance this strategy's true win
    rate beats 50%?"
    """
    return float(1.0 - stats.beta(alpha, beta).cdf(threshold))


if __name__ == "__main__":
    print("Bayesian Inference")
    print("=" * 40)

    # A strategy wins 7 of 10 trades. Start from a flat Beta(1, 1) prior.
    a0, b0 = 1.0, 1.0
    a, b = beta_binomial_update(a0, b0, successes=7, failures=3)
    lo, hi = beta_credible_interval(a, b, 0.95)
    print("Win rate after 7 wins / 3 losses (flat prior):")
    print(f"  posterior mean       : {beta_mean(a, b):.3f}")
    print(f"  95% credible interval: [{lo:.3f}, {hi:.3f}]")
    print(f"  P(win rate > 50%)    : {probability_greater_than(a, b, 0.5):.3f}")

    # Same record, but now skeptical prior worth ~40 trades centred on 50%.
    a, b = beta_binomial_update(20, 20, successes=7, failures=3)
    lo, hi = beta_credible_interval(a, b, 0.95)
    print("\nSame record, skeptical Beta(20, 20) prior:")
    print(f"  posterior mean       : {beta_mean(a, b):.3f}  (pulled back toward 0.50)")
    print(f"  95% credible interval: [{lo:.3f}, {hi:.3f}]")

    # Shrinking a noisy mean-return estimate toward a prior of 0.
    samples = [0.012, -0.004, 0.020, 0.008, -0.010]
    post_mean, post_var = normal_known_variance_update(0.0, 0.0001, samples, 0.0004)
    raw = sum(samples) / len(samples)
    print(f"\nMean daily return: raw {raw:+.5f} -> Bayesian {post_mean:+.5f} (shrunk toward 0)")
