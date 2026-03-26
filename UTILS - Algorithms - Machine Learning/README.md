# Algorithms – Machine Learning

## Overview

This module implements fundamental machine learning algorithms from scratch using only NumPy — no scikit-learn or frameworks. Building these algorithms by hand is the most effective way to understand what happens inside the black boxes used in production trading systems.

The focus is on supervised learning methods most relevant to quantitative finance: regression for factor modelling and price prediction, and classification for signal generation.

## Key Concepts

### Supervised Learning
The model learns a mapping f: X → y from labelled training data (X, y), then generalises to predict y for new, unseen X.

- **Regression**: predict a continuous value (e.g., next-period return, option price).
- **Classification**: predict a discrete label (e.g., up/down, buy/sell/hold).

### Gradient Descent
The core optimisation algorithm for most ML models:

```
theta = theta - alpha * gradient_of_loss(theta)
```

- alpha (learning rate): controls step size. Too large → diverges. Too small → slow.
- Gradient points in the direction of steepest ascent; we subtract it to descend.
- **Batch GD**: use all training samples per update (stable, slow on large datasets).
- **Stochastic GD**: use one sample per update (noisy, fast, escapes local minima).
- **Mini-batch GD**: use a small batch per update (best of both).

### Linear Regression from Scratch
Two equivalent approaches:
- **Analytical (Normal Equation)**: `theta = (X'X)^-1 X'y` — exact solution, O(n^3).
- **Gradient Descent**: iterative updates — scales to large datasets.

### Regularisation
Prevents overfitting by penalising large coefficients:
- **Ridge (L2)**: penalises sum of squared coefficients. Shrinks all coefficients.
- **Lasso (L1)**: penalises sum of absolute coefficients. Can zero out coefficients entirely (feature selection).

## Files
- `ml_algorithms.py`: Linear regression (gradient descent + normal equation), logistic regression, k-nearest neighbours, and decision tree implementations with financial examples.

## How to Run
```bash
python ml_algorithms.py
```

## Financial Applications

### 1. Return Prediction (Regression)
- Predict next-period asset return from lagged returns, volume, and factor exposures.
- Linear regression with Lasso regularisation performs implicit feature selection across many candidate predictors.

### 2. Signal Classification
- Classify market conditions as "long", "short", or "flat" based on technical indicator features.
- Logistic regression outputs a probability score usable as a position-sizing input.

### 3. Credit Scoring
- Predict probability of default from financial ratios (debt/equity, interest coverage, etc.).
- Logistic regression and decision trees are standard baselines for credit models.

### 4. Anomaly Detection
- Identify unusual order patterns or price moves using unsupervised methods (k-means, isolation forest).
- Used in market surveillance and risk management.

### 5. Factor Model Construction
- Use ML regression to find the combination of macro variables (inflation, yield, PMI) that best explains cross-sectional equity returns.

## Best Practices

- **Always split data into train/validation/test**: In finance, use a time-based split (never random shuffle) to avoid look-ahead bias.
- **Normalise features**: Gradient descent converges much faster when features are on the same scale.
- **Regularise**: Financial data is noisy and high-dimensional — always use Ridge or Lasso to prevent overfitting.
- **Out-of-sample is everything**: A model that fits the training period perfectly but fails out-of-sample is worse than useless in production.
- **Beware of non-stationarity**: Financial time series change regime over time — models trained on 2010–2015 data may fail completely in 2020.
