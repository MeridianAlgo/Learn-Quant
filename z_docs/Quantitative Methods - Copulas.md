<p class="lq-badges"><span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-cat">Quantitative Methods</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Quantitative Methods - Copulas"
    python "copula_modeling.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Quantitative%20Methods%20-%20Copulas)

---
# Quantitative Methods - Copulas

This module demonstrates the concept of Copulas, specifically the Gaussian Copula, used in quantitative finance to model the dependency structure between multivariate random variables.

## Concepts
- **Sklar's Theorem:** States that any multivariate joint distribution can be written in terms of univariate marginal distribution functions and a copula which describes the dependence structure between the variables.
- **Gaussian Copula:** A copula constructed from a multivariate normal distribution. It allows you to model correlations between assets regardless of what their individual marginal distributions look like.
- **Applications:** Portfolio risk modeling, Value at Risk (VaR), Collateralized Debt Obligations (CDO) pricing.

## Example
Run `python copula_modeling.py` to see an example of fitting a Gaussian Copula to correlated returns and generating simulated samples that respect that correlation structure.


---

## Continue in Quantitative Methods

<div class="grid cards" markdown>

-   :material-function-variant: __[Quantitative Methods - Bootstrap](Quantitative Methods - Bootstrap.md)__

    The bootstrap estimates the sampling distribution of **any** statistic by resampling the observed data with replacement — no normality assumption required. It is the honest way to put confidence intervals around backtest metrics like Sharpe ratio, mean return, or maximum drawdown.

-   :material-function-variant: __[Quantitative Methods - Cointegration](Quantitative Methods - Cointegration.md)__

    Cointegration: two non-stationary series whose **linear combination is stationary**. Backbone of statistical arbitrage and pairs trading.

-   :material-function-variant: __[Quantitative Methods - Extreme Value Theory](Quantitative Methods - Extreme Value Theory.md)__

    Most risk models assume returns are normally distributed. They are not —

-   :material-function-variant: __[Quantitative Methods - Factor Models](Quantitative Methods - Factor Models.md)__

    Factor models explain asset returns as a linear combination of systematic **factors** plus a stock-specific residual. The **Fama-French 3-Factor Model (1992)** extended CAPM by adding two well-documented risk premia: the **Size premium** (SMB) and the **Value premium** (HML), dramatically improving the explanation of cross-sectional stock returns.

-   :material-function-variant: __[Quantitative Methods - GARCH](Quantitative Methods - GARCH.md)__

    GARCH (Generalized Autoregressive Conditional Heteroskedasticity) captures **volatility clustering** — high-volatility days tend to follow high-volatility days. Used for risk forecasting, option pricing, and VaR.

-   :material-function-variant: __[Quantitative Methods - Interest Rate Models](Quantitative Methods - Interest Rate Models.md)__

    Continuous-time models for the evolution of the short (instantaneous) interest rate. Used for bond pricing, interest rate derivatives, and yield curve modeling.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
