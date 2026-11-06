# Performance Analysis Utilities

<p class="lq-meta">Advanced <span>/</span> Quantitative Methods <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Quantitative%20Methods%20-%20Performance%20Analysis">source</a></p>


## Overview
This module provides quantitative performance metrics to evaluate risk-adjusted returns and the quality of investment strategies. Beyond simple metrics like the Sharpe Ratio, these tools help quants analyze tail risk, active management skill, and the statistical properties of return series.

## Key Metrics Included

### 1. Hurst Exponent
Characterizes the long-term memory of a time series.
- **H < 0.5**: Mean-reverting series.
- **H = 0.5**: Random walk.
- **H > 0.5**: Trending series.
Files: `hurst_exponent.py`

### 2. Omega Ratio
Measures the risk-adjusted return relative to a target return level. It considers the entire return distribution, representing the ratio of probability-weighted gains to probability-weighted losses.
Files: `omega_ratio.py`

### 3. Tail Ratio
Highlights the relationship between the extreme positive and negative outliers of a return distribution. It is the absolute value of the 95th percentile return divided by the absolute value of the 5th percentile return.
Files: `tail_ratio.py`

### 4. Gain-to-Pain Ratio
A metric popularized by market wizards like Jack Schwager, representing the sum of all returns divided by the absolute sum of all negative returns. It provides a quick way to gauge the consistency of a strategy.
Files: `gain_to_pain_ratio.py`

### 5. Tracking Error and Information Ratio
Metrics for active portfolio management.
- **Tracking Error**: The standard deviation of the difference between the portfolio and its benchmark.
- **Information Ratio**: The active return per unit of tracking error, measuring a manager's skill in outperforming the index.
Files: `active_performance.py`

## Usage
Each script contains a baseline implementation and a sample execution in the `if __name__ == "__main__":` block. To run any utility, execute it from the command line:

```bash
python hurst_exponent.py
```

## Portfolio Performance
These utilities are designed to be used in conjunction with risk metrics like Value at Risk (VaR) and Drawdown to provide a holistic view of portfolio performance and risk of ruin.

---

## Running it

```bash
cd "Quantitative Methods - Performance Analysis"
python "active_performance.py"
```

## More in Quantitative Methods

- [Quantitative Methods - Bayesian Inference](Quantitative Methods - Bayesian Inference.md): A strategy wins 7 of its first 10 trades.
- [Quantitative Methods - Bootstrap](Quantitative Methods - Bootstrap.md): The bootstrap estimates the sampling distribution of any statistic by resampling the observed data with replacement — no normality assumption required.
- [Quantitative Methods - Cointegration](Quantitative Methods - Cointegration.md): Cointegration: two non-stationary series whose linear combination is stationary.
- [Quantitative Methods - Copulas](Quantitative Methods - Copulas.md): This module demonstrates the concept of Copulas, specifically the Gaussian Copula, used in quantitative finance to model the dependency structure between multivariate random variables.
- [Quantitative Methods - Entropy](Quantitative Methods - Entropy.md): Entropy measures how surprised you should expect to be.
- [Quantitative Methods - Extreme Value Theory](Quantitative Methods - Extreme Value Theory.md): Most risk models assume returns are normally distributed.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
