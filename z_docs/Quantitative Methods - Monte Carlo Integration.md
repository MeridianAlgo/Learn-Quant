<p class="lq-badges"><span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-cat">Quantitative Methods</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Quantitative Methods - Monte Carlo Integration"
    python "monte_carlo_integration.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Quantitative%20Methods%20-%20Monte%20Carlo%20Integration)

---
# Quantitative Methods, Monte Carlo Integration

Some integrals have no tidy formula, and in high dimensions the grid methods
of classical numerical analysis collapse under their own cost. Monte Carlo
sidesteps both problems with an idea that sounds too simple to work. An
integral is just an average in disguise, so estimate the average by sampling.
Throw random points at the function, take their mean, scale by the width of
the region, and the law of large numbers walks you toward the answer at a
rate that does not care how many dimensions the problem has.

The price you pay is noise. The estimate carries a standard error that
shrinks with the square root of the sample count, so each extra digit of
accuracy costs a hundred times more samples. Variance reduction techniques
claw some of that back for free, and this lesson builds the simplest one,
antithetic sampling. It closes with the flagship financial application,
pricing a European option by simulation and checking the answer against the
Black Scholes closed form.

## The pieces

* `mc_integrate(f, a, b, n, seed)` estimates the integral of f over an
  interval by averaging the function at uniform random points. It returns the
  estimate and its standard error, because a Monte Carlo answer without an
  error bar is only half an answer.
* `antithetic_integrate(f, a, b, n, seed)` pairs every draw u with its mirror
  image a plus b minus u. The two are negatively correlated, so averaging each
  pair cancels part of the noise at no extra sampling cost.
* `mc_european_call(spot, strike, rate, sigma, maturity, n, seed)` simulates
  where the stock ends up under the risk neutral measure, averages the
  discounted payoffs, and reports the price with its standard error.
* `black_scholes_call(...)` is the exact closed form used as the yardstick.

## Why the error bar matters

A Monte Carlo estimate is a random variable. Run it twice with different seeds
and you get two different numbers, and without the standard error you cannot
tell whether the difference is meaningful. The standard error also tells you
what accuracy will cost. Because it falls like one over the square root of n,
halving the error means quadrupling the samples, and one more decimal place
means a hundred times the work. That brutal arithmetic is why variance
reduction is not a nicety but the core craft of the field.

## Why finance leans on this

The option pricing example is one dimensional and Black Scholes answers it
exactly, which is precisely why it makes a good classroom, you can see the
simulation converge to the truth. The real payoff comes where no formula
exists. A payoff depending on the average price over a year, on the worst of
five stocks, or on an interest rate path is an integral over hundreds of
dimensions, and there Monte Carlo is not merely competitive, it is the only
practical tool. The pattern is always the same one this lesson builds,
simulate paths, average discounted payoffs, report the error.

## Example

```python
import math
import numpy as np
from monte_carlo_integration import mc_integrate, mc_european_call, black_scholes_call

estimate, stderr = mc_integrate(np.sin, 0, math.pi, n=50_000, seed=0)
print(estimate, "+/-", stderr)   # near 2, the exact area under sin

price, err = mc_european_call(100, 105, 0.05, 0.2, 1.0, n=200_000, seed=0)
print(price, "+/-", err)
print(black_scholes_call(100, 105, 0.05, 0.2, 1.0))  # the exact answer
```

## Where to go next

* For root finding and quadrature on a grid see
  [`Quantitative Methods - Numerical Methods`](Quantitative Methods - Numerical Methods.md).
* For the closed form being reproduced here see
  [`Black-Scholes Option Pricing`](Black-Scholes Option Pricing.md).
* For simulation of whole portfolios rather than one payoff see
  [`Monte Carlo Portfolio Simulator`](Monte Carlo Portfolio Simulator.md).
* For the random processes the samples are drawn from see
  [`Quantitative Methods - Stochastic Processes`](Quantitative Methods - Stochastic Processes.md).


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
