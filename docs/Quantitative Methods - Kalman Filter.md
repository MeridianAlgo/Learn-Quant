# Multi-Purpose Kalman Filter

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
