<p class="lq-badges"><span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-cat">Quantitative Methods</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Quantitative Methods - Bayesian Inference"
    python "bayesian_inference.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Quantitative%20Methods%20-%20Bayesian%20Inference)

---
# Quantitative Methods — Bayesian Inference

A strategy wins 7 of its first 10 trades. Is its true win rate 70%? Almost
certainly not — ten trades is barely any evidence. **Bayesian inference** is the
disciplined way to answer questions like this: start with a **prior** belief,
observe **data**, and combine them into a **posterior** that captures what you
now believe *and* how uncertain you still are.

```
posterior  ∝  likelihood × prior
```

The payoff is honest uncertainty. Instead of a single point estimate you get a
whole distribution — and a **credible interval** you can actually act on.

## Functions

| Function | Description |
|---|---|
| `beta_binomial_update(prior_alpha, prior_beta, successes, failures)` | Conjugate update for a probability |
| `beta_mean(alpha, beta)` | Posterior mean of a Beta distribution |
| `beta_credible_interval(alpha, beta, level)` | Equal-tailed credible interval for `p` |
| `probability_greater_than(alpha, beta, threshold)` | Posterior `P(p > threshold)` |
| `normal_known_variance_update(prior_mean, prior_var, data, data_var)` | Bayesian update of an unknown mean |

## Conjugate models

This module focuses on **conjugate** priors, where the posterior has the same
form as the prior and the update collapses to simple arithmetic:

- **Beta-Binomial** — estimating a probability (a win rate, a default rate).
  A `Beta(α, β)` prior plus `s` successes and `f` failures gives a
  `Beta(α + s, β + f)` posterior. `Beta(1, 1)` is the flat "I know nothing"
  prior.
- **Normal-Normal (known variance)** — estimating an unknown mean. The
  posterior mean is a **precision-weighted average** of the prior mean and the
  sample mean — shrinkage made precise.

## Example

```python
from bayesian_inference import (
    beta_binomial_update, beta_mean, beta_credible_interval, probability_greater_than,
)

# 7 wins, 3 losses from a flat prior.
a, b = beta_binomial_update(1, 1, successes=7, failures=3)
print(beta_mean(a, b))                       # 0.667 — not 0.70
print(beta_credible_interval(a, b, 0.95))    # (0.39, 0.89) — wide! only 10 trades
print(probability_greater_than(a, b, 0.5))   # 0.887 — probably beats a coin flip
```

## Frequentist vs. Bayesian intervals

A 95% **credible** interval means exactly what people *wish* a confidence
interval meant: given the prior and the data, there is a 95% probability the
parameter lies inside it. That is a statement about the *parameter*, not about
hypothetical repeated experiments — which is why it is so natural for decision
making.

## Shrinkage — taming noisy estimates

Sample means of returns are notoriously noisy. A Normal-Normal update pulls a
raw estimate toward your prior in proportion to how little data you have:

```
posterior mean = (prior_precision · prior_mean + data_precision · sample_mean)
                 / (prior_precision + data_precision)
```

This is the same instinct behind James-Stein estimators and the
[`Black-Litterman`](Portfolio Management - Black Litterman.md) model,
which blends market-implied returns (the prior) with an investor's views (the
data).

## Practical notes

- The strength of a Beta prior is `α + β` — read it as a number of "pseudo
  trades". `Beta(20, 20)` says "I'd need ~40 real trades to be talked out of
  50/50".
- Credible intervals here are **equal-tailed** (the same probability in each
  tail); highest-posterior-density intervals differ for skewed posteriors.
- For non-conjugate models you would sample the posterior (MCMC) instead — but
  the intuition you build here carries over directly.
- Estimating uncertainty in a *backtest* metric? See
  [`Quantitative Methods - Bootstrap`](Quantitative Methods - Bootstrap.md)
  for the resampling counterpart.


---

## Continue in Quantitative Methods

<div class="grid cards" markdown>

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

-   :material-function-variant: __[Quantitative Methods - GARCH](Quantitative Methods - GARCH.md)__

    GARCH (Generalized Autoregressive Conditional Heteroskedasticity) captures **volatility clustering** — high-volatility days tend to follow high-volatility days. Used for risk forecasting, option pricing, and VaR.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
