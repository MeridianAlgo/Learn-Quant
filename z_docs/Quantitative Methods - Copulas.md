# Quantitative Methods - Copulas

<p class="lq-meta">Advanced <span>/</span> Quantitative Methods <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Quantitative%20Methods%20-%20Copulas">source</a></p>


This module demonstrates the concept of Copulas, specifically the Gaussian Copula, used in quantitative finance to model the dependency structure between multivariate random variables.

## Concepts
- **Sklar's Theorem:** States that any multivariate joint distribution can be written in terms of univariate marginal distribution functions and a copula which describes the dependence structure between the variables.
- **Gaussian Copula:** A copula constructed from a multivariate normal distribution. It allows you to model correlations between assets regardless of what their individual marginal distributions look like.
- **Applications:** Portfolio risk modeling, Value at Risk (VaR), Collateralized Debt Obligations (CDO) pricing.

## Example
Run `python copula_modeling.py` to see an example of fitting a Gaussian Copula to correlated returns and generating simulated samples that respect that correlation structure.

---

## Running it

```bash
cd "Quantitative Methods - Copulas"
python "copula_modeling.py"
```

## More in Quantitative Methods

- [Quantitative Methods - Bayesian Inference](Quantitative Methods - Bayesian Inference.md): A strategy wins 7 of its first 10 trades.
- [Quantitative Methods - Bootstrap](Quantitative Methods - Bootstrap.md): The bootstrap estimates the sampling distribution of any statistic by resampling the observed data with replacement — no normality assumption required.
- [Quantitative Methods - Cointegration](Quantitative Methods - Cointegration.md): Cointegration: two non-stationary series whose linear combination is stationary.
- [Quantitative Methods - Entropy](Quantitative Methods - Entropy.md): Entropy measures how surprised you should expect to be.
- [Quantitative Methods - Extreme Value Theory](Quantitative Methods - Extreme Value Theory.md): Most risk models assume returns are normally distributed.
- [Quantitative Methods - Factor Models](Quantitative Methods - Factor Models.md): Factor models explain asset returns as a linear combination of systematic factors plus a stock-specific residual.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
