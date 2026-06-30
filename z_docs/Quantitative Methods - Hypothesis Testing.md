<p class="lq-badges"><span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-cat">Quantitative Methods</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Quantitative Methods - Hypothesis Testing"
    python "hypothesis_testing.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Quantitative%20Methods%20-%20Hypothesis%20Testing)

---
# Quantitative Methods, Hypothesis Testing

You found an edge. The average daily return of your strategy is positive, one
backtest beats another, a signal seems to predict the next move. The hard
question is whether any of it is real or whether you are looking at noise. A
hypothesis test answers that question with a number called the p value, the
probability of seeing a result at least this extreme if nothing real were going
on. A small p value is evidence that the pattern is not luck.

This lesson builds the everyday tests by hand so you can see where each
statistic comes from, and only borrows the t and normal distributions to turn a
statistic into a p value.

## The idea in one paragraph

Every test starts with a null hypothesis, the boring assumption that there is no
effect, the mean is zero, the two strategies are the same. You compute a
statistic that measures how far your data sits from that assumption, counted in
standard errors. Then you ask how often pure chance would produce a statistic
that large. That probability is the p value. If it is below a threshold you
chose in advance, usually five percent, you reject the null and treat the effect
as real.

## The tests in this module

* `one_sample_ttest(sample, mu0)` checks whether a sample mean differs from a
  fixed value such as zero. This is the test for whether a strategy has a real
  positive average return.
* `two_sample_ttest(a, b, equal_var)` checks whether two independent series have
  different means. The default is the Welch version, which does not assume the
  two have the same spread and is the safer choice for real returns.
* `z_test(sample, mu0, sigma)` is the simpler cousin of the t test for when you
  genuinely know the population standard deviation, which usually means a large
  sample.
* `confidence_interval(sample, confidence)` returns the range that would contain
  the true mean in a given fraction of repeated samples.
* `reject_null(p_value, alpha)` is a small convenience that reports whether a p
  value clears your significance threshold.

## Why the t test and not the normal

For a small sample you do not know the true standard deviation, you estimate it,
and that estimate is itself uncertain. The t distribution has fatter tails than
the normal to account for that extra uncertainty, so it gives honest answers
when you have a handful of observations rather than thousands. As the sample
grows the t distribution converges to the normal and the two tests agree.

## Example

```python
from hypothesis_testing import one_sample_ttest, confidence_interval, reject_null

daily = [0.12, -0.05, 0.20, 0.08, -0.10, 0.15, 0.03, 0.18, -0.02, 0.11]

result = one_sample_ttest(daily, mu0=0.0)
print(result["t"])          # how many standard errors above zero
print(result["p_value"])    # the chance of this under no real edge
print(reject_null(result["p_value"]))   # True if significant at five percent

low, high = confidence_interval(daily, 0.95)
print(low, high)            # where the true mean plausibly sits
```

## A warning worth keeping

A p value below five percent does not mean the effect is large or that you will
make money, only that it is unlikely to be pure chance. Test enough useless
signals and roughly one in twenty will pass by accident, which is why mining a
backtest for the best p value is a trap. Decide what you are testing before you
look, and treat a single significant result as a hint rather than a verdict.

## Where to go next

* For the descriptive statistics these tests are built on see
  [`Quantitative Methods - Statistics`](Quantitative Methods - Statistics.md).
* For resampling based significance that makes fewer assumptions see
  [`Quantitative Methods - Bootstrap`](Quantitative Methods - Bootstrap.md).
* For fitting relationships rather than testing means see
  [`Quantitative Methods - Regression Analysis`](Quantitative Methods - Regression Analysis.md).


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
