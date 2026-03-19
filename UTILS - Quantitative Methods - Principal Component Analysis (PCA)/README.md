# Principal Component Analysis (PCA) in Trading

## Overview
This module demonstrates the application of Principal Component Analysis (PCA) to financial timeseries data.

In quantitative finance, PCA is a robust technique used to reduce dimensionality and extract independent latent factors (eigenportfolios) from a large universe of asset returns. The first principal component typically represents the general market factor, while subsequent components represent sector and idiosyncratic factors.

## What You'll Learn
- How to apply PCA to stock returns to identify latent risk factors.
- Interpreting eigenvalues and the variance explained by each principal component.
- Creating continuous factor returns.
- Using the results for pairs-trading, statistical arbitrage, and robust portfolio optimization.

## Running the Code
Run the demonstration script:
```bash
python pca_trading.py
```
This script will produce the proportion of variance explained by the major factors and generate a scree plot visually showing factor convergence (`pca_variance.png`).
