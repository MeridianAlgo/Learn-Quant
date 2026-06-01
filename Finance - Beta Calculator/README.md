# Beta Calculator

## What is Beta?

**Beta** measures how much a stock or portfolio moves compared to the overall market.

- **Beta = 1.0**: Moves exactly with the market (if market goes up 10%, stock goes up 10%)
- **Beta > 1.0**: More volatile than the market (amplified gains and losses)
- **Beta < 1.0**: Less volatile than the market (smoother, steadier)
- **Beta < 0**: Moves opposite the market (rare, valuable for hedging)

### Real-World Examples

| Company | Beta | What It Means |
|---------|------|---------------|
| Apple | ~1.2 | 20% more volatile than S&P 500 |
| Procter & Gamble | ~0.7 | 30% less volatile (defensive stock) |
| Tesla | ~2.0+ | Twice as volatile as the market |
| Gold (uncorrelated) | ~0 | Moves independently |

## Core Concepts

### 1. **Systematic vs. Idiosyncratic Risk**

- **Systematic Risk** (Beta) = market risk you can't diversify away
  - Example: Interest rate hike hurts ALL stocks
  - Beta captures this
  
- **Idiosyncratic Risk** = company-specific risk you CAN diversify away
  - Example: CEO scandal, product recall
  - Diversification eliminates this, so it shouldn't affect your cost of capital

### 2. **The Beta Formula**

```
Beta = Covariance(Asset Returns, Market Returns)
       ─────────────────────────────────────
              Variance(Market Returns)
```

**In English**: 
- Numerator: How much does the asset move *with* the market?
- Denominator: How much does the market move on its own?
- Result: Ratio of co-movement to market volatility

### 3. **CAPM: Where Beta is Used**

```
Expected Return = Risk-Free Rate + Beta × (Market Return - Risk-Free Rate)
                = Rf + Beta × (Rm - Rf)
```

**Example calculation**:
```
Risk-free rate (US Treasury): 4.5%
Market risk premium: 6%
Stock beta: 1.2

Expected return = 4.5% + 1.2 × 6% = 4.5% + 7.2% = 11.7%
```

Investors demand an 11.7% return as compensation for accepting 20% more volatility than the market.

## Calculation Methods

### **1. Standard Beta** (Most Common)
Simple least-squares regression of asset returns vs. market returns.

```python
from beta_calculator import calculate_beta

stock_returns = [0.01, -0.02, 0.015, -0.01, 0.02]  # Daily returns
market_returns = [0.008, -0.015, 0.012, -0.008, 0.018]

beta = calculate_beta(stock_returns, market_returns)
# Result: beta ≈ 1.1
```

**When to use**: 
- Standard risk analysis
- CAPM valuation
- Portfolio allocation

### **2. Rolling Beta** (Track Changes Over Time)
Beta isn't static—it changes as market conditions change. Rolling beta lets you see when a stock's riskiness changes.

```python
from beta_calculator import rolling_beta

betas_over_time = rolling_beta(stock_returns, market_returns, window=60)
# 60-day rolling windows show beta evolution
```

**When to use**:
- Detecting market regime changes
- Risk management (identifying when risk is increasing)
- Strategy backtesting

**Example**: In bull markets, growth stocks (high beta) outperform. In bear markets, defensive stocks (low beta) outperform. Rolling beta helps you see when this shift occurs.

### **3. Levered vs. Unlevered Beta** (Accounting for Debt)
A company with debt (borrowed money) appears riskier because:
- Equity holders are residual claimants (take losses first)
- Debt holders have priority in bankruptcy
- Financial leverage amplifies volatility

**Unlevered Beta** (Asset Beta): Risk of just the operations, no debt

**Levered Beta** (Equity Beta): Risk including the impact of debt

```python
from beta_calculator import levered_beta, unlevered_beta

# Company has 50% debt-to-equity, 21% tax rate
unlevered = 1.0
levered = levered_beta(unlevered, debt_to_equity=0.5, tax_rate=0.21)
# levered ≈ 1.395 (higher risk because of debt)
```

**When to use**:
- Comparing companies with different capital structures
- Valuation (must unlever before applying different leverage)
- M&A analysis (understand true business risk vs. financing risk)

### **4. Downside Beta & Upside Beta** (Asymmetric Risk)
Particularly relevant for risk management.

- **Downside Beta**: How much does it fall when market falls? (matters most for risk)
- **Upside Beta**: How much does it rise when market rises?

If downside beta is materially greater than upside beta, the asset amplifies losses more than gains — an unfavourable profile.
If downside beta ≈ upside beta: Consistent risk profile

```python
from beta_calculator import downside_beta, upside_beta

down_beta = downside_beta(stock_returns, market_returns, threshold=0.0)
up_beta = upside_beta(stock_returns, market_returns, threshold=0.0)

print(f"Downside beta: {down_beta:.2f}")  # e.g., 1.3 (worse in downturns!)
print(f"Upside beta: {up_beta:.2f}")      # e.g., 1.1 (similar upside)
```

**Why this matters**: 
- A hedge fund with 1.0 beta is safe IF it has equal up/down beta
- But if downside beta = 2.0, it crashes twice as hard when market crashes
- This is why many hedge funds underperformed in 2022

### **5. Beta Decomposition** (Attribution)
Separate the components of beta to understand what's driving risk:
- Correlation with market
- Relative volatility to market

```python
from beta_calculator import beta_decomposition

decomp = beta_decomposition(stock_returns, market_returns)
# {'correlation': 0.85, 'relative_volatility': 1.30, 'beta': 1.105}

# Interpretation: 
# - Stock correlates 85% with market (fairly close)
# - Stock volatility is 30% higher than market
# - Combined beta: 0.85 × 1.30 = 1.105
```

**When to use**:
- Understanding if high beta comes from correlation or volatility
- Deciding whether to reduce risk via diversification or stock selection

### **6. Adjusted Beta (Blume Method)** (Handle Regression to Mean)
Raw beta can be noisy. Academic research (Blume, 1971) shows beta regresses toward 1.0 over time.

```python
from beta_calculator import adjusted_beta

raw_beta = 1.5
adj_beta = adjusted_beta(raw_beta)  # ≈ 1.33
```

**Why**: Extreme betas tend to become less extreme. Adjustment accounts for this.

**When to use**:
- Better long-term estimates of beta
- Valuation models
- Risk forecasting

## Method Comparison

| Method | Use Case | Pros | Cons |
|--------|----------|------|------|
| **Standard Beta** | CAPM, valuation | Simple, intuitive | Static, ignores regime changes |
| **Rolling Beta** | Risk monitoring | Shows changes over time | More computation, noisier |
| **Levered/Unlevered** | Valuation, M&A | Accounts for capital structure | Need debt info |
| **Downside Beta** | Risk management | Captures asymmetric risk | Requires threshold choice |
| **Upside Beta** | Opportunity assessment | Shows gain potential | Less emphasis in practice |
| **Beta Decomposition** | Understanding drivers | Shows correlation vs volatility | Requires full return series |
| **Adjusted Beta** | Long-term estimates | Better for forecasting | Assumes mean reversion |

## How to Run

```bash
python beta_calculator.py
```

Or import individual functions:

```python
from beta_calculator import calculate_beta, rolling_beta, levered_beta

# Your data
stock_returns = [...]  # Daily/monthly returns
market_returns = [...]  # Matching period returns

# Calculate
beta = calculate_beta(stock_returns, market_returns)
print(f"Beta: {beta:.4f}")
```

## Practical Examples

### Example 1: Compare Two Stocks
```python
# Apple vs. S&P 500
apple_returns = [...]  # Get from Yahoo Finance
market_returns = [...]

apple_beta = calculate_beta(apple_returns, market_returns)
# Result: ~1.2 (riskier than market)

# What does this mean for CAPM?
rf = 0.045  # 4.5% treasury
market_premium = 0.06  # 6% expected excess return
expected_return = rf + apple_beta * market_premium
print(f"Apple expected return: {expected_return:.1%}")  # 11.7%
```

### Example 2: Portfolio Beta
```python
# Your portfolio: 60% AAPL, 40% JNJ
aapl_beta = 1.2
jnj_beta = 0.7
portfolio_beta = 0.6 * aapl_beta + 0.4 * jnj_beta
# = 0.6 × 1.2 + 0.4 × 0.7 = 1.0
# The portfolio carries market-level risk even though AAPL alone is higher-beta.
```

### Example 3: Valuation with Leverage
```python
# Target company in M&A deal
# Current: beta 1.0, 30% debt-to-equity
# After acquisition: will have 60% debt-to-equity

current_beta = 1.0
new_de = 0.60
new_beta = levered_beta(
    unlevered_beta(current_beta, 0.30),
    new_de
)
# New beta will be higher—equity holders take on more risk
```

## Learning Path

**Prerequisites**: 
- [Python Basics – Numbers](../Python%20Basics%20-%20Numbers/) (percentages, arithmetic)
- [Quantitative Methods – Statistics](../Quantitative%20Methods%20-%20Statistics/) (covariance, variance, regression)
- [Finance – Correlation Analysis](../Finance%20-%20Correlation%20Analysis/)

**Builds into**:
- [CAPM](../CAPM/) for cost of equity estimation
- [Portfolio Optimizer](../Portfolio%20Optimizer/) for risk-adjusted allocation
- [Black-Scholes Option Pricing](../Black-Scholes%20Option%20Pricing/) for option risk

## Common Questions

**Q: My calculated beta differs from what Yahoo Finance shows. Why?**
A: Different time periods (1y vs 3y vs 5y), different market indices (S&P 500 vs Russell 1000), or different return frequencies (daily vs monthly).

**Q: What if beta is negative?**
A: Rare, but great for portfolios! Negative beta = hedge. Gold often has near-zero beta.

**Q: Should I use daily, weekly, or monthly returns?**
A: Monthly (more stable). Daily is noisier. Weekly is rarely used. Avoid intra-day unless doing HFT.

**Q: How often should I recalculate beta?**
A: For active management: quarterly or monthly. For passive: annually. For trading: use rolling beta.

**Q: Is beta constant?**
A: No! It changes with market regimes, business changes, leverage changes. Use rolling beta to track.

## References

- Sharpe, W. (1964). "Capital Asset Prices: A Theory of Market Equilibrium" (Original CAPM paper)
- Blume, M. (1971). "On the Assessment of Risk"
- Levered beta formula: Hamada, R. (1972)
