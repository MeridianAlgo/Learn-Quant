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
[`Black-Litterman`](../Portfolio%20Management%20-%20Black%20Litterman/) model,
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
  [`Quantitative Methods - Bootstrap`](../Quantitative%20Methods%20-%20Bootstrap/)
  for the resampling counterpart.
