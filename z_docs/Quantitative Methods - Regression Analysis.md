# Quantitative Methods – Regression Analysis

<p class="lq-meta">Advanced <span>/</span> Quantitative Methods <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Quantitative%20Methods%20-%20Regression%20Analysis">source</a></p>


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

## Running it

```bash
cd "Quantitative Methods - Regression Analysis"
python "regression_tutorial.py"
```

## More in Quantitative Methods

- [Quantitative Methods - Bayesian Inference](Quantitative Methods - Bayesian Inference.md): A strategy wins 7 of its first 10 trades.
- [Quantitative Methods - Bootstrap](Quantitative Methods - Bootstrap.md): The bootstrap estimates the sampling distribution of any statistic by resampling the observed data with replacement — no normality assumption required.
- [Quantitative Methods - Cointegration](Quantitative Methods - Cointegration.md): Cointegration: two non-stationary series whose linear combination is stationary.
- [Quantitative Methods - Copulas](Quantitative Methods - Copulas.md): This module demonstrates the concept of Copulas, specifically the Gaussian Copula, used in quantitative finance to model the dependency structure between multivariate random variables.
- [Quantitative Methods - Entropy](Quantitative Methods - Entropy.md): Entropy measures how surprised you should expect to be.
- [Quantitative Methods - Extreme Value Theory](Quantitative Methods - Extreme Value Theory.md): Most risk models assume returns are normally distributed.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
