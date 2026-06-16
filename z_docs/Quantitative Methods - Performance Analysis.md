<p class="lq-badges"><span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-cat">Quantitative Methods</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Quantitative Methods - Performance Analysis"
    python "active_performance.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Quantitative%20Methods%20-%20Performance%20Analysis)

---
# Performance Analysis Utilities

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
