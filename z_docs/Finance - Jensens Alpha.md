# Finance, Jensen's Alpha

<p class="lq-meta">Intermediate <span>/</span> Risk & Performance <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Finance%20-%20Jensens%20Alpha">source</a></p>


Beta tells you how much of a portfolio's return came from simply riding the
market. Jensen's alpha tells you what is left once that free ride is taken
away. You ask the CAPM what return a portfolio should have earned given the
beta it carried, you subtract that from what it actually earned, and the
remainder belongs to the manager. Positive alpha means the fund beat its own
risk profile. Negative alpha means the same exposure could have been bought
more cheaply through an index.

## The pieces

* `beta(asset_returns, market_returns)` is the slope of the asset against the
  market, its covariance divided by the market's variance.
* `expected_return(asset_returns, market_returns, risk_free_rate)` is the
  CAPM prediction, the risk free rate plus beta times the market's excess
  return.
* `jensens_alpha(...)` subtracts that prediction from the realised mean
  return. The risk free rate is quoted per period, matching the series.
* `annualized_alpha(...)` multiplies the per period alpha by the number of
  periods in a year.
* `alpha_t_statistic(...)` regresses excess asset returns on excess market
  returns and divides the intercept by its standard error.

## Why the t statistic matters more than the alpha

Alpha is a small number sitting on top of a very noisy one. A fund can post a
positive alpha for years purely by chance, and the raw figure gives you no
way to tell that apart from skill. The t statistic does. It compares the size
of the alpha to how much the fund's returns wander around the market line, so
a big alpha inside a wildly scattered return series scores lower than a
modest alpha that shows up consistently. A t statistic near two is the usual
rough threshold for taking an edge seriously, and even that is generous once
you remember how many funds are being measured at the same time.

## Leverage is not alpha

A fund that borrows to hold twice the market will beat the market in a rising
year, and its Sharpe ratio may look fine. Its alpha will not move, because
the extra return is exactly what the higher beta already predicted. This is
the whole point of the measure. It prices the free ride first and only then
looks at what remains.

## Edge cases

The market series needs some variance or beta is undefined, and the t
statistic needs at least three observations plus some residual scatter. Both
cases raise rather than return a misleading number. Alpha also inherits every
weakness of the CAPM itself, so a fund loaded on factors the single market
index does not capture will show an alpha that is really just an unmeasured
exposure.

## Example

```python
from jensens_alpha import annualized_alpha, alpha_t_statistic, beta

fund = [0.004, -0.002, 0.006, -0.001, 0.003, 0.002, -0.004, 0.005]
market = [0.005, -0.003, 0.004, -0.002, 0.004, 0.001, -0.005, 0.006]

print(beta(fund, market))
print(annualized_alpha(fund, market, risk_free_rate=0.0001))
print(alpha_t_statistic(fund, market))
```

## Where to go next

* For the model this whole measure is built on see [`CAPM`](CAPM.md).
* For reward per unit of beta rather than reward beyond beta see
  [`Finance - Treynor Ratio`](Finance - Treynor Ratio.md).
* For the regression machinery underneath the t statistic see
  [`Quantitative Methods - Regression Analysis`](Quantitative Methods - Regression Analysis.md).

---

## Running it

```bash
cd "Finance - Jensens Alpha"
python "jensens_alpha.py"
```

## More in Risk & Performance

- [Finance - Calmar Ratio](Finance - Calmar Ratio.md): The Sharpe ratio judges a strategy by how much its returns wobble around their average.
- [Finance - Information Ratio](Finance - Information Ratio.md): When a portfolio is judged against a benchmark, what matters is how much it beat the benchmark by — and how *reliably*.
- [Finance - Omega Ratio](Finance - Omega Ratio.md): The Sharpe ratio squeezes a whole return distribution into a mean and a standard deviation, which throws away everything interesting about its shape.
- [Finance - Performance Attribution](Finance - Performance Attribution.md): Brinson decomposition splits portfolio active return into allocation and selection effects — answering *"did we beat the benchmark by picking the right sectors or the right stocks?"*
- [Finance - Treynor Ratio](Finance - Treynor Ratio.md): The Sharpe ratio divides excess return by total volatility, which mixes two very different kinds of risk.
- [Risk Metrics](Risk Metrics.md): This module gives you quick, professional stats about risk in any list or array of investment returns.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
