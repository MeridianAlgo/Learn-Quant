# Learn-Quant: Master Quantitative Finance and Python (v1.9.0)

[![Lint](https://github.com/MeridianAlgo/Learn-Quant/actions/workflows/lint.yml/badge.svg)](https://github.com/MeridianAlgo/Learn-Quant/actions/workflows/lint.yml)

**Welcome to Learn-Quant** Your comprehensive toolkit for mastering algorithmic trading, quantitative finance theory, and professional Python software engineering.

---

## Overview

Learn-Quant is a curated collection of over 60 self-contained modules designed to bridge the gap between academic theory and production-grade code. Whether you are a student, a software engineer transitioning into finance, or a trader learning to code, this repository provides the essential building blocks.

### Key Learning Outcomes
- **Master Quant Strategies**: Implement Kelly Criterion, Risk Parity, Pairs Trading, and Momentum.
- **Engineer Robust Systems**: Learn AsyncIO, Metaprogramming, Context Managers, and advanced OOP.
- **Deep Dive into Mathematics**: Kalman Filters, Yield Curve Modeling, Stochastic Processes, and Linear Algebra.
- **Build Core Tools**: Create Option Pricers, Risk Engines (VaR), and Backtesting Simulators.
- **Computer Science Algorithms**: Understand how Sorting, Graph Theory, and Dynamic Programming apply to market data.

---

## Repository Structure

Every folder is a fully functional lesson. Pick a topic and run the code.

### Level 1: Python Fundamentals
Essential coding skills for financial analysis.
- `UTILS - Python Basics - Numbers`: Floating point precision and financial math.
- `UTILS - Python Basics - Strings`: Ticker manipulation and news parsing.
- `UTILS - Python Basics - Control Flow`: Implementing trading logic and rules.
- `UTILS - Python Basics - Functions`: Building reusable quant libraries.

### Level 2: Data Structures and Algorithms
Optimizing performance for high-frequency environments.
- `UTILS - Data Structures`: Efficient use of Lists, Sets, Tuples, and Dictionaries.
- `UTILS - Algorithms - Sorting`: Algorithmic efficiency (Quicksort, Mergesort).
- `UTILS - Algorithms - Searching`: Binary search on time-series data.
- `UTILS - Algorithms - Graph`: Arbitrage detection using shortest paths.
- `UTILS - Algorithms - Dynamic Programming`: Optimizing execution paths.

### Level 3: Advanced Engineering
Writing professional, production-ready code.
- `UTILS - Advanced Python - AsyncIO`: Building high-throughput data pipelines.
- `UTILS - Advanced Python - OOP`: Designing scalable Trading Engines and Portfolio Managers.
- `UTILS - Advanced Python - Metaprogramming`: Singleton patterns and dynamic class creation.
- `UTILS - Advanced Python - Context Managers`: Handling database locks and atomic transactions.
- `UTILS - Advanced Python - Decorators`: Custom logging, timing, and error handling wrappers.
- `UTILS - Advanced Python - Error Handling`: Robust systems that never crash mid-trade.

### Level 4: Quantitative Methods
The mathematics of the markets.
- `UTILS - Quantitative Methods - Principal Component Analysis (PCA)`: Dimensionality reduction and eigenportfolios.
- `UTILS - Quantitative Methods - Kalman Filter`: Dynamic hedge ratios and noise filtering.
- `UTILS - Quantitative Methods - Stochastic Processes`: Geometric Brownian Motion and Monte Carlo.
- `UTILS - Quantitative Methods - Statistics`: Hypothesis testing, stationarity, and cointegration.
- `UTILS - Quantitative Methods - Regression`: Factor models and Alpha generation.
- `UTILS - Quantitative Methods - Linear Algebra`: Portfolio optimization and risk modelling.
- `UTILS - Yield Curve Modeling`: Interpolation and bootstrapping interest rate curves.

### Level 5: Strategies and Finance
Applied quantitative finance.
- `UTILS - Kelly Criterion`: Optimal position sizing and risk management.
- `UTILS - Risk Parity`: Multi-asset portfolio construction based on risk contribution.
- `UTILS - Strategies - Pairs Trading`: Statistical arbitrage and mean reversion.
- `UTILS - Strategies - Momentum Trading`: Trend following and signal generation.
- `UTILS - Black-Scholes Option Pricing`: Greeks, implied volatility, and derivatives pricing.
- `UTILS - Options - Binomial Tree Pricing`: Discrete-time options pricing models.
- `UTILS - Finance - Volatility Calculator`: Parkinson, Garman-Klass, and EWMA estimators.
- `UTILS - Portfolio Optimizer`: Efficient Frontier, Sharpe Ratio, and Markowitz optimization.
- `UTILS - Risk Metrics`: Value at Risk (VaR), CVaR, Drawdown, and Sortino Ratio.
- `UTILS - Technical Indicators`: Custom implementations of RSI, MACD, and Bollinger Bands.

### Level 6: AI and Alternative Data
Modern approaches to trading.
- `UTILS - AI Development`: Basic market prediction models.
- `UTILS - Sentiment Analysis on News`: NLP for fundamental analysis.
- `UTILS - Websocket Connection`: Real-time market data streaming.

---

## Usage

### 1. Installation
Clone the repository and install the required dependencies.
```bash
git clone https://github.com/MeridianAlgo/Learn-Quant
pip install -r requirements.txt
```

### 2. Running a Module

**Option A: Interactive Menu (New in v1.9.0)**
Run the new interactive platform command to browse and run modules dynamically:
```bash
python learn_quant.py
```

**Option B: Manual Execution**
Navigate to any directory and run the tutorial script.

**Example: Running the Kelly Criterion**
```bash
cd "UTILS - Kelly Criterion"
python kelly_criterion.py
```

**Example: Running the Momentum Strategy**
```bash
cd "UTILS - Strategies - Momentum Trading"
python momentum_strategy.py
```

---

## Contributing
We believe in open-source knowledge. Contributions are welcome!
- **Feedback**: If you find a bug, please open an issue.
- **Strategies**: Fork the repository and submit a pull request for new strategies.
- **Documentation**: Improvements to explanations are always appreciated.

---

### License
This project is open-sourced under the MIT License.

---

**Learn-Quant v1.9.0**
Quantitative Finance | Algorithmic Trading | Python Mastery
Maintained by MeridianAlgo

