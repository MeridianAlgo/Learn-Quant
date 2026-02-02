# Strategies – Momentum Trading

## 📋 Overview

Momentum trading is a strategy that capitalizes on the continuance of existing trends in the market. The core philosophy is "buy high, sell higher." If an asset's price is rising strongly, momentum traders assume it will continue to rise.

This utility implements a basic **Trend-Following Momentum Strategy** using Rate of Change (ROC) and Moving Averages.

## 🎯 Key Concepts

### **Momentum**
- Measures the speed or velocity of price changes.
- **Rate of Change (ROC)**: The percentage change between current price and price $n$ periods ago.
- Positive Momentum suggests an Uptrend; Negative Momentum suggests a Downtrend.

### **Trend Filtering**
- Momentum signals can be false in choppy markets.
- **Moving Averages (SMA/EMA)** are often used as a filter.
- Rule: Only take Long positions if `Price > SMA` (price is above the long-term trend).

## 💻 Logic Implemented

We combine two signals:
1.  **Momentum**: ROC(20) > 0
2.  **Trend**: Price > SMA(50)

**Signal Logic:**
- **Enter Long (1)**: When Momentum is positive AND Price is above the 50-period SMA.
- **Exit/Neutral (0)**: When Momentum turns negative OR Price drops below the SMA.

## 📂 Files
- `momentum_strategy.py`: Logic for generating synthetic data, calculating indicators, generating signals, and running a simple backtest.

## 🚀 How to Run
```bash
python momentum_strategy.py
```

## 🧠 Financial Applications

### 1. Cross-Sectional Momentum
- Buying the top N performing stocks in an index and shorting the bottom N.
- Using Relative Strength (not RSI) to compare assets.

### 2. Time-Series Momentum
- Focusing on a single asset's own history (Trend Following).
- Managed Futures / CTAs implementation.

### 3. Risk Management
- Momentum strategies often suffer from "Momentum Crashes" (sudden reversals).
- Volatility scaling is often used to manage position size.

## 💡 Best Practices

- **Lookback Period**: The choice of '$n$' (e.g., 12 months, 6 months, 20 days) drastically affects performance. Longer periods reduce noise but lag trend turning points.
- **Transaction Costs**: Frequent trading in momentum strategies can erode profits.
- **Diversification**: Apply momentum across multiple unconnected assets (Equities, Commodities, FX) to smooth returns.
