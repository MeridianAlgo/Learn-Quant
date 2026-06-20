<p class="lq-badges"><span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-cat">Quantitative Methods</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Quantitative Methods - Markov Chains"
    python "markov_chains.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Quantitative%20Methods%20-%20Markov%20Chains)

---
# Quantitative Methods — Markov Chains

A **Markov chain** models a system that hops between a finite set of *states*
where the next state depends only on the current one — never on the full
history. That "memoryless" assumption is crude, but it is the backbone of a
surprising amount of quant work: market **regime models** (bull / bear /
sideways), **credit-rating migration** matrices, and toy models of volatility
clustering.

Everything you need lives in one object — the **transition matrix** `P`, where
`P[i, j]` is the probability of moving from state `i` to state `j` in one step.
Each row is a probability distribution, so it sums to 1. This module builds `P`,
takes its powers, finds its long-run behaviour, and simulates paths — all from
NumPy.

## Functions

| Function | Description |
|---|---|
| `normalize_rows(matrix)` | Turn observed transition counts into a row-stochastic `P` |
| `is_stochastic(P)` | Check every row is a valid probability distribution |
| `n_step(P, n)` | The `n`-step transition matrix `P^n` (Chapman-Kolmogorov) |
| `stationary_distribution(P)` | Long-run fraction of time in each state (`pi P = pi`) |
| `simulate(P, start, steps, random_state)` | Generate a sample path of states |
| `expected_return_time(P, state)` | Mean steps to revisit a state (`1 / pi[state]`) |

## The stationary distribution

Multiply a starting distribution by `P` over and over and — for a well-behaved
(irreducible, aperiodic) chain — it settles to a fixed vector `pi` that no longer
changes: `pi P = pi`. That `pi` is the **long-run fraction of time** the chain
spends in each state, independent of where it started. We compute it as the left
eigenvector of `P` for eigenvalue 1.

It answers a genuinely useful question: *if today's regime can be any of bull,
bear or flat, what fraction of all days are bull days in the long run?*

## Estimating P from data

You rarely know `P`; you estimate it. Count how many times the market went from
each regime to each other regime, then normalise each row:

```python
import numpy as np
from markov_chains import normalize_rows, stationary_distribution

counts = np.array([[820,  30,  60],   # Bull -> Bull/Bear/Flat
                   [ 25, 300,  40],   # Bear -> ...
                   [ 70,  45, 410]])  # Flat -> ...

P = normalize_rows(counts)
print(P.sum(axis=1))               # [1. 1. 1.]
print(stationary_distribution(P))  # long-run regime mix
```

## Example — market regimes

```python
import numpy as np
from markov_chains import stationary_distribution, n_step, simulate

P = np.array([[0.90, 0.03, 0.07],
              [0.05, 0.85, 0.10],
              [0.15, 0.10, 0.75]])

print(stationary_distribution(P))  # e.g. [0.58 0.18 0.24]
print(n_step(P, 5)[0])             # where you are 5 days after a Bull day
print(simulate(P, start=0, steps=10, random_state=1))  # a sample path
```

## Practical notes

- **The diagonal is "stickiness."** Regimes persist, so real transition matrices
  have a dominant diagonal — markets stay bull far more often than they flip.
- **Aperiodic + irreducible** guarantees a unique stationary distribution. A
  chain with an unreachable state, or one that cycles deterministically, breaks
  that — `stationary_distribution` will still return *a* fixed vector, but
  interpret it with care.
- For continuous, mean-reverting state instead of discrete regimes, see
  [`Quantitative Methods - Stochastic Processes`](Quantitative Methods - Stochastic Processes.md).
- For *detecting* which regime you are in from returns, see
  [`Quantitative Methods - Regime Detection`](Quantitative Methods - Regime Detection.md).
- The same machinery powers the prediction step in the
  [`Quantitative Methods - Kalman Filter`](Quantitative Methods - Kalman Filter.md).


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
