# Strategies – Mean Reversion

## 📋 Overview

Mean reversion is the statistical tendency for an asset's price to return to its historical average after deviating from it. While Momentum strategies bet on *continuation*, Mean Reversion strategies bet on *reversal* — buying when something is "too cheap" and selling when it is "too expensive" relative to recent history.

This utility implements a **Bollinger Band + RSI Mean Reversion Strategy** using a synthetic Ornstein-Uhlenbeck price process.

## 🎯 Key Concepts

### **Mean Reversion**
- Rooted in the statistical concept of *regression to the mean*.
- Works best in ranging, sideways markets where there is no dominant trend.
- Commonly applied to: pairs of correlated assets, interest rates, commodity spreads, volatility indices (e.g., VIX).

### **Bollinger Bands**
- Middle Band: Simple Moving Average (SMA) over `window` periods — our estimate of fair value.
- Upper/Lower Band: SMA ± `num_std` × rolling standard deviation.
- When price touches the **lower band**, it is statistically cheap (a potential buy).
- When price touches the **upper band**, it is statistically expensive (a potential sell/short).

### **RSI Confirmation Filter**
- RSI < 35 confirms *oversold* conditions — strengthens the lower band buy signal.
- RSI > 65 confirms *overbought* conditions — strengthens the upper band sell signal.
- Using dual confirmation dramatically reduces false entries in trending markets.

### **Ornstein-Uhlenbeck (OU) Process**
- The continuous-time stochastic model of mean reversion.
- Used in interest rate models (Vasicek), commodity pricing, and pairs trading.
- Governed by: `dx = kappa * (theta - x) * dt + sigma * dW`

## 💻 Logic Implemented

**Entry signals (both conditions required):**
- **Long (+1)**: Price < Lower Bollinger Band **AND** RSI < 35
- **Short (–1)**: Price > Upper Bollinger Band **AND** RSI > 65

**Exit signals:**
- Exit Long when price reverts above the SMA (fair value reached).
- Exit Short when price reverts below the SMA.

## 📂 Files
- `mean_reversion_strategy.py`: OU data generation, Bollinger Bands, RSI, signal logic, backtest engine, and performance metrics.

## 🚀 How to Run
```bash
python mean_reversion_strategy.py
```

## 🧠 Financial Applications

### 1. Statistical Arbitrage
- Trade the spread between two cointegrated assets (e.g., pairs trading).
- The spread is the OU process; enter when spread deviates too far.

### 2. Fixed Income / Interest Rates
- Short-term interest rates exhibit strong mean reversion.
- The Vasicek model (based on OU process) is used for bond pricing and interest rate derivatives.

### 3. Volatility Trading
- Implied volatility (VIX) is mean-reverting by nature.
- Selling options when VIX is high, buying when low, exploits this reversion.

### 4. Market-Making
- Market makers implicitly bet on short-term mean reversion.
- They buy at the bid and sell at the ask, profiting when prices revert within the spread.

## 💡 Best Practices

- **Market Regime Detection**: Mean reversion strategies fail badly in strong trends. Use a regime filter (e.g., ADX indicator) to disable the strategy when the market is trending.
- **Transaction Costs**: Mean reversion often requires many trades; commissions and slippage must be accounted for.
- **Parameter Sensitivity**: Test multiple window lengths (10, 20, 30 days) and band widths (1.5, 2.0, 2.5 std) — results can vary significantly.
- **Combining with Momentum**: Many professional quant funds run both momentum and mean reversion strategies simultaneously to diversify across regimes.
