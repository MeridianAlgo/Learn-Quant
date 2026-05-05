# Learn-Quant: Master Quantitative Finance & Python (v2.3.0)

[![Lint](https://github.com/MeridianAlgo/Learn-Quant/actions/workflows/lint.yml/badge.svg)](https://github.com/MeridianAlgo/Learn-Quant/actions/workflows/lint.yml)

**Welcome to Learn-Quant!** Your all-in-one, comprehensive toolkit for mastering algorithmic trading, quantitative finance theory, and professional Python software engineering.

---

---

## What is New in v2.3.0

Five new advanced quant modules added, plus the GitHub Pages deploy now uses the official Pages artifact pipeline (no more `gh-pages` branch force-push errors).

| Module | Directory | Highlights |
|---|---|---|
| garch.py | UTILS - Quantitative Methods - GARCH | EWMA + GARCH(1,1) MLE fit, multi-step variance forecast |
| cointegration.py | UTILS - Quantitative Methods - Cointegration | ADF test, Engle-Granger, half-life, rolling z-score for pairs |
| performance_attribution.py | UTILS - Finance - Performance Attribution | Brinson-Hood-Beebower 3-factor + 2-factor, IR, tracking error |
| stress_testing.py | UTILS - Risk Metrics - Stress Testing | Hypothetical + historical (2008, 2020, 1987, dotcom, 2022) shock engine, reverse stress |
| trend_following.py | UTILS - Strategies - Trend Following | Donchian breakout (Turtles), MA crossover, TSMOM, ATR sizing |

### Previous Releases
- **v2.2.0**: 13 quant finance modules (Kelly, FX, exotic options, Black-Litterman, regime detection, etc.)
- **v2.1.0**: Four interactive quiz-based tutorials (statistics, options, risk, portfolio)
- **v2.0.0**: Performance Analysis utils (Hurst, Omega, Tail, Gain-Pain)

## Overview

Learn-Quant is a massive, curated collection of over 60+ self-contained modules designed to bridge the gap between academic theory and production-grade code. Whether you are a student, a software engineer moving into finance, or a trader learning to code, this repository provides the building blocks you need.

### Key Learning Outcomes
- **Master Quant Strategies**: Implement Pairs Trading, Momentum, Mean Reversion, Position Sizing, and more.
- **Engineer Robust Systems**: Learn AsyncIO, Context Managers, Decorators, and advanced OOP.
- **Deep Dive into Math**: Kalman Filters, Stochastic Processes, Factor Models, Linear Algebra for Portfolio Theory.
- **Build Core Tools**: Create your own Option Pricers, Risk Engines (VaR), and Backtesting Simulators.
- **CS Algorithms**: Understand how Sorting, Graph Theory, and Dynamic Programming apply to market data.

---

## Repository Structure

Every folder is a fully functional lesson. Pick a topic and run the code.

### Level 1: Python Fundamentals
*Essential coding skills for financial analysis.*
- `UTILS - Python Basics - Numbers`: Floating point precision & financial math.
- `UTILS - Python Basics - Strings`: Ticker manipulation & news parsing.
- `UTILS - Python Basics - Control Flow`: Implementing trading logic & rules.
- `UTILS - Python Basics - Functions`: Building reusable quant libraries.

### Level 2: Data Structures & Algorithms
*Optimizing performance for high-frequency environments.*
- `UTILS - Data Structures`: Efficient use of Lists, Sets, Tuples, and Dictionaries.
- `UTILS - Algorithms - Sorting`: Algorithmic efficiency (Quicksort, Mergesort).
- `UTILS - Algorithms - Searching`: Binary search on time-series data.
- `UTILS - Algorithms - Graph`: Arbitrage detection using shortest paths.
- `UTILS - Algorithms - Dynamic Programming`: Optimizing execution paths.

### Level 3: Advanced Engineering
*Writing professional, production-ready code.*
- `UTILS - Advanced Python - AsyncIO`: Building high-throughput data pipelines.
- `UTILS - Advanced Python - OOP`: Designing scalable Trading Engines & Portfolio Managers.
- `UTILS - Advanced Python - Context Managers`: Handling database locks and atomic transactions.
- `UTILS - Advanced Python - Decorators`: Custom logging, timing, and error handling wrappers.
- `UTILS - Advanced Python - Error Handling`: Robust systems that never crash mid-trade.
- `UTILS - Advanced Python - Multiprocessing`: Parallel Monte Carlo, backtests, and parameter sweeps across all CPU cores.

### Level 4: Quantitative Methods
*The mathematics of the markets.*
- `UTILS - Quantitative Methods - Kalman Filter`: Dynamic hedge ratios & noise filtering.
- `UTILS - Quantitative Methods - Stochastic Processes`: Geometric Brownian Motion & Monte Carlo.
- `UTILS - Quantitative Methods - Statistics`: Hypothesis testing, stationarity, and cointegration. Includes **interactive tutorial** (`statistics_tutorial.py`) with quizzes covering Z-scores, correlation, and fat tails.
- `UTILS - Quantitative Methods - Regression`: Factor models & Alpha generation.
- `UTILS - Quantitative Methods - Linear Algebra`: Portfolio optimization & risk modelling.
- `UTILS - Quantitative Methods - Factor Models`: Fama-French 3-Factor model, factor regression, alpha decomposition, and performance attribution.
- `UTILS - Quantitative Methods - Performance Analysis`: Hurst Exponent, Omega Ratio, Tail Ratio, and Active Metrics.
- `UTILS - Quantitative Methods - GARCH`: EWMA and GARCH(1,1) volatility estimation, MLE fitting, multi-step forecasting.
- `UTILS - Quantitative Methods - Cointegration`: ADF unit-root test, Engle-Granger two-step, OU half-life, rolling z-score for pairs trading.

### Level 5: Strategies & Finance
*Applied quantitative finance.*
- `UTILS - Strategies - Pairs Trading`: Statistical arbitrage & mean reversion.
- `UTILS - Strategies - Momentum Trading`: Trend following & signal generation.
- `UTILS - Strategies - Mean Reversion`: Bollinger Band + RSI signals, Ornstein-Uhlenbeck process, and reversion-to-mean backtesting.
- `UTILS - Black-Scholes Option Pricing`: Greeks, implied volatility, & derivatives pricing. Includes **interactive tutorial** (`options_tutorial.py`) covering Black-Scholes, all five Greeks, and put-call parity.
- `UTILS - Finance - Volatility Calculator`: Parkinson, Garman-Klass, & EWMA estimators.
- `UTILS - Finance - Yield Curve`: Nelson-Siegel model fitting, forward rate extraction, and curve shape classification.
- `UTILS - Finance - Position Sizing`: Kelly Criterion, Fixed Fractional, Volatility Targeting, and Risk of Ruin.
- `UTILS - Portfolio Optimizer`: Efficient Frontier, Sharpe Ratio, & Markowitz optimization. Includes **interactive tutorial** (`portfolio_tutorial.py`) walking through MPT and portfolio construction.
- `UTILS - Risk Metrics`: Value at Risk (VaR), CVaR, Drawdown, & Sortino Ratio. Includes **interactive tutorial** (`risk_tutorial.py`) with worked examples and quizzes.
- `UTILS - Technical Indicators`: Custom implementations of RSI, MACD, Bollinger Bands.
- `UTILS - Strategies - Trend Following`: Donchian channel breakout, MA crossover, time-series momentum, ATR-based volatility position sizing.
- `UTILS - Finance - Performance Attribution`: Brinson-Hood-Beebower allocation/selection/interaction decomposition, information ratio, tracking error.
- `UTILS - Risk Metrics - Stress Testing`: Hypothetical and historical scenario engine (2008 GFC, 2020 COVID, 1987, dotcom, 2022), univariate sensitivity, reverse stress tests.

### Level 6: AI & Alternative Data
*Modern approaches to trading.*
- `UTILS - AI Development`: Basic market prediction models.
- `UTILS - Sentiment Analysis on News`: NLP for fundamental analysis.
- `UTILS - Websocket Connection`: Real-time market data streaming.

### Level 7: Market Microstructure
*Understanding order book dynamics and market impact.*
- `UTILS - Market Microstructure`: Order book implementation, spread analysis, and market impact models.
- `UTILS - High Frequency Trading`: Latency optimization, execution algorithms, and HFT strategies.

---

## Usage

### 1. Installation
Clone the repository and install the required dependencies.
```bash
git clone https://github.com/MeridianAlgo/Learn-Quant
pip install -r requirements.txt
```

### 2. Running a Module
Navigate to any directory and run the tutorial script.

**Example: Running the Momentum Strategy**
```bash
cd "UTILS - Strategies - Momentum Trading"
python momentum_strategy.py
```

**Example: Learning Performance Metrics**
```bash
cd "UTILS - Quantitative Methods - Performance Analysis"
python hurst_exponent.py
```

---

## Contributing
We believe in open-source knowledge. Contributions are welcome!
- **Found a bug?** Open an Issue.
- **Have a new strategy?** Fork the repo and submit a Pull Request.
- **Documentation improvements?** We love those too.

---

### License
This project is open-sourced under the MIT License.

---

**Learn-Quant v2.3.0**
*Quantitative Finance | Algorithmic Trading | Python Mastery*
**Maintained by MeridianAlgo**
