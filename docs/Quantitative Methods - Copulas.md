# Quantitative Methods - Copulas

This module demonstrates the concept of Copulas, specifically the Gaussian Copula, used in quantitative finance to model the dependency structure between multivariate random variables.

## Concepts
- **Sklar's Theorem:** States that any multivariate joint distribution can be written in terms of univariate marginal distribution functions and a copula which describes the dependence structure between the variables.
- **Gaussian Copula:** A copula constructed from a multivariate normal distribution. It allows you to model correlations between assets regardless of what their individual marginal distributions look like.
- **Applications:** Portfolio risk modeling, Value at Risk (VaR), Collateralized Debt Obligations (CDO) pricing.

## Example
Run `python copula_modeling.py` to see an example of fitting a Gaussian Copula to correlated returns and generating simulated samples that respect that correlation structure.
