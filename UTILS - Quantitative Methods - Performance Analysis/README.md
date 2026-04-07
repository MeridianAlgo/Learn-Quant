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
