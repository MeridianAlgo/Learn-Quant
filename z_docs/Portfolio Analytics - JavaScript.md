# Portfolio Analytics, JavaScript

<p class="lq-meta">Advanced <span>/</span> Portfolio Management <span>/</span> JavaScript <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Portfolio%20Analytics%20-%20JavaScript">source</a></p>


A portfolio is not the sum of its holdings, and the whole of portfolio theory
follows from that one fact. Two assets that each move ten percent a year can
combine into something that moves seven, provided they do not move together,
and that reduction is free in the sense that it costs no expected return.
Everything in this file is a way of measuring or exploiting that.

Pure Node.js, no dependencies. Run it with `node portfolioAnalytics.js`.

## The pieces

* `covarianceMatrix(returns)` builds the matrix from a table of returns where
  rows are periods and columns are assets.
* `correlationMatrix(cov)` rescales it to the more readable form.
* `portfolioReturn(weights, expectedReturns)` and `portfolioVariance(weights,
  cov)` and `portfolioVolatility(weights, cov)` are the basics.
* `solve(A, b)` is Gauss Jordan elimination with partial pivoting, which is
  the only linear algebra the rest of the file needs.
* `minVarianceWeights(cov)` returns the lowest risk combination available.
* `tangencyWeights(cov, expectedReturns, riskFreeRate)` returns the best
  Sharpe ratio combination.
* `riskContributions(weights, cov)` splits the portfolio risk across holdings.
* `diversificationRatio(weights, cov)` scores how much the combination
  actually bought you.

## Why the linear algebra is written out here

Both optimizers need to solve a system of equations against the covariance
matrix. At realistic portfolio sizes that system is a handful of rows, and
Gauss Jordan with partial pivoting is about thirty lines. Pulling in a matrix
library for that would be a poor trade, and writing it out means you can read
what the optimizer is doing rather than trusting a call.

The pivoting matters. Without it the elimination divides by whatever happens
to sit on the diagonal, and covariance matrices routinely have small values
there. The solver throws when the pivot is effectively zero, which in practice
means two of your assets are the same asset, or you have more assets than
periods of history and the matrix cannot possibly be invertible.

## Weights are not risk

The most useful function in this file is `riskContributions`, and the demo
exists mostly to make its point. An equal weighted portfolio of three assets
does not carry a third of its risk in each. The volatile asset that correlates
with everything else can easily account for more than half, while the quiet
one contributes almost nothing but occupies the same share of capital.

Risk parity is the idea of solving that directly, by choosing weights so the
contributions come out equal instead of the weights. You cannot get there in
closed form, it needs iteration, so it lives in its own module. What you can
do here is measure the imbalance in whatever portfolio you already hold, which
is usually a more uncomfortable number than people expect.

## What the optimizer will do if you let it

`minVarianceWeights` and `tangencyWeights` are the unconstrained solutions. A
negative weight means the optimum wants a short position, and if that is not
allowed by the mandate then the answer needs constraining, which is a quadratic
program rather than a solve.

The larger problem is that both are extremely sensitive to their inputs, and
the tangency portfolio especially so, because expected returns are the least
reliable estimates in finance and the optimizer treats them as certainty. Nudge
one expected return by a percent and the recommended weights can swing wildly.
This is not a flaw in the code, it is the honest consequence of asking for the
exact optimum of a problem whose inputs are guesses. The usual defences are
shrinking the covariance estimate toward something simple, imposing weight
bounds, or starting from equilibrium returns rather than forecasts, which is
what Black Litterman does.

There is a reason equal weighting is a hard benchmark to beat out of sample.

## Diversification collapses when you need it

The diversification ratio compares the weighted average of the individual
volatilities against the volatility of the combination. One means the
diversification achieved nothing, and higher is better. The uncomfortable
property is that it depends entirely on correlations, and correlations rise
in a crisis. The portfolio that showed a ratio of one and a half in calm
markets can be sitting near one at precisely the moment the protection was
supposed to pay off. Measure it in a stressed sample as well as a full one.

## Example

```javascript
const { covarianceMatrix, minVarianceWeights, riskContributions } = require('./portfolioAnalytics');

const returns = [
  [0.01, 0.002], [-0.02, 0.004], [0.015, -0.001],
  [0.005, 0.003], [-0.01, 0.006], [0.02, 0.001],
];

const cov = covarianceMatrix(returns);
const weights = minVarianceWeights(cov);

console.log(weights);
console.log(riskContributions(weights, cov).percent);
```

## Where to go next

* For the constrained frontier in Python see
  [`Portfolio Optimizer`](Portfolio Optimizer.md).
* For equalising the contributions rather than measuring them see
  [`Portfolio Management - Risk Parity`](Portfolio Management - Risk Parity.md).
* For starting from equilibrium instead of forecasts see
  [`Portfolio Management - Black Litterman`](Portfolio Management - Black Litterman.md).
* For better covariance estimates than the sample one see
  [`Finance - Covariance Estimation`](Finance - Covariance Estimation.md).

---

## Running it

```bash
cd "Portfolio Analytics - JavaScript"
node "portfolioAnalytics.js"
```

## More in Portfolio Management

- [Monte Carlo Portfolio Simulator](Monte Carlo Portfolio Simulator.md): This utility helps you forecast possible futures for a portfolio using random simulations—a key idea in finance, risk management, and statistics!
- [Portfolio Management](Portfolio Management.md): This folder contains utilities for portfolio management, risk analysis, and investment optimization.
- [Portfolio Management - Black Litterman](Portfolio Management - Black Litterman.md): The Black-Litterman (1990) model addresses the instability of mean-variance optimization by blending market equilibrium returns with investor views using Bayesian updating.
- [Portfolio Management - Hierarchical Risk Parity](Portfolio Management - Hierarchical Risk Parity.md): Markowitz optimisation needs the inverse of the covariance matrix, and the inverse of a noisy matrix is mostly noise.
- [Portfolio Management - Risk Parity](Portfolio Management - Risk Parity.md): Risk parity builds a portfolio where every asset contributes the same amount of risk to the total — not the same amount of capital.
- [Portfolio Optimizer](Portfolio Optimizer.md): This utility helps you find the best mix of assets for a portfolio, balancing risk and return using the foundation of Modern Portfolio Theory (MPT).

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
