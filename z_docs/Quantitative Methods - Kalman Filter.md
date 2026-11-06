# Multi-Purpose Kalman Filter

<p class="lq-meta">Advanced <span>/</span> Quantitative Methods <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Quantitative%20Methods%20-%20Kalman%20Filter">source</a></p>


## Overview
This module provides a pure Python implementation of a 1-Dimensional Kalman Filter. Kalman filters are recursive algorithms used to estimate the state of a linear dynamic system from a series of noisy measurements.

## Applications in Quant
- **Price Smoothing**: Filtering out high-frequency noise from price data to see the underlying trend.
- **Pairs Trading**: Estimating the dynamic hedge ratio (beta) between two cointegrated assets.
- **Volatility Estimation**: Smoothing realized volatility estimates.

## Usage
Run the script directly to see a simple example of tracking a constant value with noise.

```bash
python kalman_filter.py
```

---

## Running it

```bash
cd "Quantitative Methods - Kalman Filter"
python "kalman_filter.py"
```

## More in Quantitative Methods

- [Quantitative Methods - Bayesian Inference](Quantitative Methods - Bayesian Inference.md): A strategy wins 7 of its first 10 trades.
- [Quantitative Methods - Bootstrap](Quantitative Methods - Bootstrap.md): The bootstrap estimates the sampling distribution of any statistic by resampling the observed data with replacement — no normality assumption required.
- [Quantitative Methods - Cointegration](Quantitative Methods - Cointegration.md): Cointegration: two non-stationary series whose linear combination is stationary.
- [Quantitative Methods - Copulas](Quantitative Methods - Copulas.md): This module demonstrates the concept of Copulas, specifically the Gaussian Copula, used in quantitative finance to model the dependency structure between multivariate random variables.
- [Quantitative Methods - Entropy](Quantitative Methods - Entropy.md): Entropy measures how surprised you should expect to be.
- [Quantitative Methods - Extreme Value Theory](Quantitative Methods - Extreme Value Theory.md): Most risk models assume returns are normally distributed.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
