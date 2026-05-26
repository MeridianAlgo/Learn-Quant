# Correlation Analysis

Analyze correlations between financial instruments for portfolio construction and risk management.

## Features

- Pearson Correlation
- Rolling Correlation
- Correlation Matrix
- EWMA Correlation
- Rank Correlation (Spearman)
- Tail Correlation
- Correlation Stability Analysis

## Usage

```python
from correlation_analysis import pearson_correlation, rolling_correlation

returns1 = [0.01, -0.02, 0.015, -0.01, 0.02]
returns2 = [0.008, -0.015, 0.012, -0.008, 0.018]

corr = pearson_correlation(returns1, returns2)
print(f"Correlation: {corr:.4f}")

rolling = rolling_correlation(returns1, returns2, window=3)
```

## Applications

- Portfolio diversification
- Pairs trading identification
- Risk management
- Hedge effectiveness
- Market regime detection
