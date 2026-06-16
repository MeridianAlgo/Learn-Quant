<p class="lq-badges"><span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-cat">Quantitative Methods</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Quantitative Methods - Regression Analysis"
    python "regression_tutorial.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Quantitative%20Methods%20-%20Regression%20Analysis)

---
# Quantitative Methods – Regression Analysis

## Overview

Regression analysis is the statistical "Swiss Army Knife" of quantitative finance. It allows you to quantify relationships between variables, such as how a stock moves relative to the market (Beta) or how factors drive returns.

## Key Concepts

### **Linear Regression**
- **Equation**: $y = \alpha + \beta x + \epsilon$
- **Beta ($\beta$)**: Sensitivity of asset to the market
- **Alpha ($\alpha$)**: Excess return independent of the market
- **R-Squared ($R^2$)**: How well the model explains the data (0 to 1)

### **Multiple Regression**
- Using multiple independent variables to explain returns.
- **Example**: Fama-French 3-Factor Model (Market, Size, Value).

### **Diagnostics**
- **Residuals**: The difference between actual and predicted values. Should be random noise.
- **t-statistic**: Is the coefficient significantly different from zero?
- **Standard Error**: The uncertainty in the estimate.

## Key Examples

### Calculating Beta
```python
import numpy as np

# Fit line: Stock Returns = alpha + beta * Market Returns
coeffs = np.polyfit(market_returns, stock_returns, 1)
beta = coeffs[0]
alpha = coeffs[1]

print(f"Beta: {beta:.2f}")
```

### Multiple Regression (Matrix Form)
```python
# y = X * beta
# X includes [1, Market, SMB, HML]
X = np.column_stack([np.ones(N), market, smb, hml])
beta = np.linalg.inv(X.T @ X) @ X.T @ y
```

## Files
- `regression_tutorial.py`: Interactive tutorial with examples

## How to Run
```bash
pip install numpy scipy
python regression_tutorial.py
```

## Financial Applications

### 1. Beta Calculation (CAPM)
Determine how risky a stock is compared to the S&P 500. High beta (>1) means more volatile; low beta (<1) means more stable.

### 2. Factor Investing
Identify which factors (Size, Momentum, Value, Quality) are driving a portfolio's performance using multiple regression.

### 3. Pairs Trading (Hedge Ratio)
Find the optimal hedge ratio between two correlated assets (e.g., Coke vs. Pepsi) to create a market-neutral spread.

### 4. Predictive Modeling
Forecast future returns based on lagged indicators (e.g., dividend yield, interest rates), though this is notoriously difficult!

## Best Practices

- **Check Assumptions**: Linear regression assumes linear relationship, constant variance (homoscedasticity), and independent errors.
- **Look at Residuals**: If residuals show a pattern, your model is missing something.
- **Avoid Overfitting**: Adding too many variables increases $R^2$ but hurts predictive power. Use Adjusted $R^2$.

---

*Master regression to uncover the hidden drivers of financial markets!*

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
