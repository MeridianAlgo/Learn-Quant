<p class="lq-badges"><span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-cat">Quantitative Methods</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Quantitative Methods - Time Series"
    python "time_series_tools.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Quantitative%20Methods%20-%20Time%20Series)

---
# Quantitative Methods – Time Series Utility

## Overview

This utility introduces core time-series techniques used in quantitative finance. It serves as a bridge between the intermediate and advanced curriculum (`Documentation/Programs/level3_financial.py` and `level4_advanced.py`) and gives you reusable helpers for analyzing historical price data.

## Key Skills
- Generating and cleaning time-series price data
- Calculating rolling statistics (moving averages, volatility)
- Computing autocorrelation and partial autocorrelation
- Performing stationarity checks (Augmented Dickey-Fuller)
- Building a simple AR(1) forecast as a baseline model

## Files
- `time_series_tools.py`: Guided walkthrough with annotated prints and helper functions

## How to Run
```bash
python time_series_tools.py
```
Open the script while it runs to follow the inline commentary.

## Dependencies
- pandas
- numpy
- statsmodels (for the ADF test)

Install them with:
```bash
pip install pandas numpy statsmodels
```

## Practice Ideas
- Swap the simulated data with real prices from `yfinance`
- Experiment with different rolling window lengths for volatility
- Extend the AR(1) section to ARIMA using `statsmodels.tsa.arima.model`

## Related Modules
- `Documentation/Programs/level3_financial.py`
- `Documentation/Programs/level4_advanced.py`
- `UTILS - Technical Indicators/`

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
