<p class="lq-badges"><span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-cat">Quantitative Methods</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Quantitative Methods - Entropy"
    python "entropy.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Quantitative%20Methods%20-%20Entropy)

---
# Quantitative Methods, Entropy and Information

Entropy measures how surprised you should expect to be. A coin that always
lands heads carries no information, because you already knew the answer, and
its entropy is zero. A fair coin is the most uncertain a two sided bet can
be, and measured in base two its entropy is exactly one bit. Every other
distribution sits somewhere between those two extremes, and the number tells
you how many yes or no questions it would take on average to pin down the
outcome.

Three related quantities fall out of that one idea, and each answers a
question that comes up constantly in quantitative work.

## The pieces

* `shannon_entropy(probabilities, base)` is the uncertainty of a known
  distribution, in bits by default.
* `entropy_from_samples(samples, base)` counts observed outcomes into an
  empirical distribution first, which is what you need with real data.
* `normalized_entropy(probabilities)` divides by the maximum possible for
  that many outcomes, giving a zero to one score that compares across sizes.
* `kl_divergence(p, q, base)` is the cost of believing `q` when the truth is
  `p`.
* `mutual_information(x, y, base)` is how much knowing one label sequence
  tells you about the other.
* `discretize(values, bins)` buckets a continuous series so the discrete
  measures above can be applied to it.

## What divergence is good for

Kullback Leibler divergence is the extra cost, in bits, of describing
outcomes that really come from `p` using a code built for `q`. That makes it
the natural score for how badly a model's assumed distribution misses the
realised one. Fit a normal distribution to a return series, compare it to the
empirical histogram, and the divergence puts a number on exactly how much the
fat tails are costing you.

Two properties matter in practice. It is never negative and it is zero only
when the two distributions agree exactly. And it is not symmetric, so the
divergence from `q` to `p` is a different number than from `p` to `q`. That
asymmetry is a feature, not a defect. Being wrong in a place the truth visits
often should cost more than being wrong somewhere it rarely goes.

## Mutual information versus correlation

Correlation only sees straight lines. A variable that reliably drives the
magnitude of returns while leaving the sign alone will show a correlation
near zero, and a screen built on correlation will throw it away. Mutual
information sees it, because it asks only whether knowing one variable
reduces uncertainty about the other, without ever proposing a shape for the
relationship.

The price is that mutual information needs discrete outcomes and therefore
needs binning, and the bin count is a real choice. Too few bins and the
relationship is smoothed away. Too many and every observation lands in its
own bucket, at which point the measure reports a great deal of shared
information about what is really just noise. Sample size sets the ceiling on
how many bins you can honestly afford.

## Edge cases

Zero probability outcomes contribute nothing to entropy, which is the
standard convention since something that never happens carries no surprise.
Divergence returns infinity when `q` rules out something `p` allows, which is
the honest answer and a good reason to smooth an empirical distribution
before using it as `q`. Probabilities are normalized on the way in, so raw
counts can be passed directly.

## Example

```python
from entropy import discretize, kl_divergence, mutual_information, shannon_entropy

print(shannon_entropy([0.5, 0.5]))
print(shannon_entropy([0.9, 0.1]))
print(kl_divergence([0.5, 0.5], [0.9, 0.1]))

returns = [0.01, -0.02, 0.03, -0.01, 0.005, -0.004, 0.02, -0.03]
volume = [100, 190, 310, 95, 60, 44, 205, 290]
print(mutual_information(discretize(returns, 3), discretize(volume, 3)))
```

## Where to go next

* For the linear measure this one generalises see
  [`Finance - Correlation Analysis`](Finance - Correlation Analysis.md).
* For using these scores to pick model inputs see
  [`Machine Learning - Feature Engineering`](Machine Learning - Feature Engineering.md).
* For the distributions being compared see
  [`Quantitative Methods - Statistics`](Quantitative Methods - Statistics.md).


---

## Continue in Quantitative Methods

<div class="grid cards" markdown>

-   :material-function-variant: __[Quantitative Methods - Bayesian Inference](Quantitative Methods - Bayesian Inference.md)__

    A strategy wins 7 of its first 10 trades. Is its true win rate 70%? Almost

-   :material-function-variant: __[Quantitative Methods - Bootstrap](Quantitative Methods - Bootstrap.md)__

    The bootstrap estimates the sampling distribution of **any** statistic by resampling the observed data with replacement — no normality assumption required. It is the honest way to put confidence intervals around backtest metrics like Sharpe ratio, mean return, or maximum drawdown.

-   :material-function-variant: __[Quantitative Methods - Cointegration](Quantitative Methods - Cointegration.md)__

    Cointegration: two non-stationary series whose **linear combination is stationary**. Backbone of statistical arbitrage and pairs trading.

-   :material-function-variant: __[Quantitative Methods - Copulas](Quantitative Methods - Copulas.md)__

    This module demonstrates the concept of Copulas, specifically the Gaussian Copula, used in quantitative finance to model the dependency structure between multivariate random variables.

-   :material-function-variant: __[Quantitative Methods - Extreme Value Theory](Quantitative Methods - Extreme Value Theory.md)__

    Most risk models assume returns are normally distributed. They are not —

-   :material-function-variant: __[Quantitative Methods - Factor Models](Quantitative Methods - Factor Models.md)__

    Factor models explain asset returns as a linear combination of systematic **factors** plus a stock-specific residual. The **Fama-French 3-Factor Model (1992)** extended CAPM by adding two well-documented risk premia: the **Size premium** (SMB) and the **Value premium** (HML), dramatically improving the explanation of cross-sectional stock returns.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
