# Quantitative Methods – Time Series Utility

<p class="lq-meta">Advanced <span>/</span> Quantitative Methods <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Quantitative%20Methods%20-%20Time%20Series">source</a></p>


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

## Running it

```bash
cd "Quantitative Methods - Time Series"
python "time_series_tools.py"
```

## More in Quantitative Methods

- [Quantitative Methods - Bayesian Inference](Quantitative Methods - Bayesian Inference.md): A strategy wins 7 of its first 10 trades.
- [Quantitative Methods - Bootstrap](Quantitative Methods - Bootstrap.md): The bootstrap estimates the sampling distribution of any statistic by resampling the observed data with replacement — no normality assumption required.
- [Quantitative Methods - Cointegration](Quantitative Methods - Cointegration.md): Cointegration: two non-stationary series whose linear combination is stationary.
- [Quantitative Methods - Copulas](Quantitative Methods - Copulas.md): This module demonstrates the concept of Copulas, specifically the Gaussian Copula, used in quantitative finance to model the dependency structure between multivariate random variables.
- [Quantitative Methods - Entropy](Quantitative Methods - Entropy.md): Entropy measures how surprised you should expect to be.
- [Quantitative Methods - Extreme Value Theory](Quantitative Methods - Extreme Value Theory.md): Most risk models assume returns are normally distributed.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
