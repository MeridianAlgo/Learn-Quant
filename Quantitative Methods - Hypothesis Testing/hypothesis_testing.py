"""
Hypothesis Testing
------------------
A hypothesis test asks a simple question. You see a number in your data, a mean
return, a difference between two strategies, an edge that looks real, and you
want to know whether it could just be noise. The test gives you a p value, the
probability of seeing a result at least this extreme if nothing real were going
on. A small p value is evidence that the effect is not luck.

This lesson builds the everyday tests from scratch. It computes each statistic by
hand so you can see exactly where it comes from, and only borrows the t and
normal distributions from scipy to turn a statistic into a p value. The same
machinery answers most practical questions a quant has about a track record.
"""

from __future__ import annotations

import math

from scipy import stats


def _mean(x: list[float]) -> float:
    return sum(x) / len(x)


def _sample_std(x: list[float]) -> float:
    """Sample standard deviation using the n minus 1 correction.

    The minus one, called the degrees of freedom correction, keeps the estimate
    unbiased when you only have a sample rather than the whole population.
    """
    n = len(x)
    if n < 2:
        raise ValueError("need at least two observations for a standard deviation")
    m = _mean(x)
    var = sum((v - m) ** 2 for v in x) / (n - 1)
    return math.sqrt(var)


def one_sample_ttest(sample: list[float], mu0: float = 0.0) -> dict:
    """Test whether a sample mean differs from a known value mu0.

    Use this to ask whether an average daily return is really above zero. The
    statistic is how many standard errors the sample mean sits away from mu0.
    Returns the t statistic, the degrees of freedom, and the two sided p value.
    """
    n = len(sample)
    if n < 2:
        raise ValueError("one sample t test needs at least two observations")
    m = _mean(sample)
    se = _sample_std(sample) / math.sqrt(n)
    if se == 0:
        raise ValueError("sample has zero variance, the t statistic is undefined")
    t = (m - mu0) / se
    df = n - 1
    p = 2 * stats.t.sf(abs(t), df)
    return {"t": t, "df": df, "p_value": p, "mean": m}


def two_sample_ttest(a: list[float], b: list[float], equal_var: bool = False) -> dict:
    """Test whether two independent samples have different means.

    The default is Welch's t test, which does not assume the two groups share a
    variance and is the safer choice for real return series. Set equal_var to
    True for the classic pooled Student test. Returns the statistic, the degrees
    of freedom, and the two sided p value.
    """
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        raise ValueError("each sample needs at least two observations")
    ma, mb = _mean(a), _mean(b)
    va = _sample_std(a) ** 2
    vb = _sample_std(b) ** 2

    if equal_var:
        # Pooled variance assumes both groups share one spread.
        pooled = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
        se = math.sqrt(pooled * (1 / na + 1 / nb))
        df = na + nb - 2
    else:
        # Welch keeps the variances separate and adjusts the degrees of freedom.
        se = math.sqrt(va / na + vb / nb)
        num = (va / na + vb / nb) ** 2
        den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
        df = num / den

    if se == 0:
        raise ValueError("combined variance is zero, the t statistic is undefined")
    t = (ma - mb) / se
    p = 2 * stats.t.sf(abs(t), df)
    return {"t": t, "df": df, "p_value": p, "mean_a": ma, "mean_b": mb}


def z_test(sample: list[float], mu0: float, sigma: float) -> dict:
    """Test a sample mean against mu0 when the population sigma is known.

    The z test is the t test's simpler cousin. It is the right tool when you
    genuinely know the population standard deviation, which in practice usually
    means a large sample where the estimate is trustworthy. Returns the z
    statistic and the two sided p value from the normal distribution.
    """
    n = len(sample)
    if n < 1:
        raise ValueError("z test needs at least one observation")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    se = sigma / math.sqrt(n)
    z = (_mean(sample) - mu0) / se
    p = 2 * stats.norm.sf(abs(z))
    return {"z": z, "p_value": p}


def confidence_interval(sample: list[float], confidence: float = 0.95) -> tuple[float, float]:
    """Return the confidence interval for a sample mean using the t distribution.

    A ninety five percent interval is the range that would contain the true mean
    in ninety five out of a hundred repeated samples. It uses the t distribution
    so it stays honest for small samples where the normal approximation is loose.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    n = len(sample)
    if n < 2:
        raise ValueError("confidence interval needs at least two observations")
    m = _mean(sample)
    se = _sample_std(sample) / math.sqrt(n)
    crit = stats.t.ppf(0.5 + confidence / 2, n - 1)
    margin = crit * se
    return (m - margin, m + margin)


def reject_null(p_value: float, alpha: float = 0.05) -> bool:
    """Report whether a p value clears the significance threshold alpha.

    Rejecting the null means the result is unlikely enough under chance that you
    treat the effect as real. Alpha is the false positive rate you accept, and
    five percent is the common default. This is a convenience, not new evidence.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    return bool(p_value < alpha)


if __name__ == "__main__":
    print("Hypothesis Testing")
    print("=" * 40)

    # A small run of daily returns in percent. Is the average really above zero?
    daily = [0.12, -0.05, 0.20, 0.08, -0.10, 0.15, 0.03, 0.18, -0.02, 0.11]

    one = one_sample_ttest(daily, mu0=0.0)
    print("\nOne sample t test, mean daily return versus zero")
    print(f"  mean    {one['mean']:.4f}")
    print(f"  t stat  {one['t']:.3f}")
    print(f"  p value {one['p_value']:.4f}")
    print(f"  reject the no edge null at 5 percent  {reject_null(one['p_value'])}")

    lo, hi = confidence_interval(daily, 0.95)
    print(f"\n95 percent confidence interval for the mean  [{lo:.4f}, {hi:.4f}]")

    # Two strategies, are their average returns different?
    strat_a = [0.10, 0.12, 0.09, 0.14, 0.11, 0.13, 0.08]
    strat_b = [0.06, 0.05, 0.08, 0.04, 0.07, 0.03, 0.06]
    two = two_sample_ttest(strat_a, strat_b)
    print("\nWelch two sample t test, strategy A versus strategy B")
    print(f"  mean A  {two['mean_a']:.4f}   mean B  {two['mean_b']:.4f}")
    print(f"  t stat  {two['t']:.3f}")
    print(f"  p value {two['p_value']:.6f}")
    print(f"  the difference is significant at 5 percent  {reject_null(two['p_value'])}")

    z = z_test(daily, mu0=0.0, sigma=0.10)
    print("\nZ test with a known sigma of 0.10")
    print(f"  z stat  {z['z']:.3f}   p value {z['p_value']:.4f}")
