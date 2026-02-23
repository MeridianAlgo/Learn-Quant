# Learn-Quant: Master Quantitative Finance & Python (v1.8.0)

[![Lint](https://github.com/MeridianAlgo/Learn-Quant/actions/workflows/lint.yml/badge.svg)](https://github.com/MeridianAlgo/Learn-Quant/actions/workflows/lint.yml)

**Welcome to Learn-Quant!** Your all-in-one, comprehensive toolkit for mastering algorithmic trading, quantitative finance theory, and professional Python software engineering.

---

## Overview

Learn-Quant is a massive, curated collection of over 50+ self-contained modules designed to bridge the gap between academic theory and production-grade code. Whether you are a student, a software engineer moving into finance, or a trader learning to code, this repository provides the building blocks you need.

### Key Learning Outcomes
- **Master Quant Strategies**: Implement Pairs Trading, Momentum, Mean Reversion, and more.
- **Engineer Robust Systems**: Learn AsyncIO, Context Managers, Decorators, and advanced OOP.
- **Deep Dive into Math**: Kalman Filters, Stochastic Processes, Linear Algebra for Portfolio Theory.
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

### Level 4: Quantitative Methods
*The mathematics of the markets.*
- `UTILS - Quantitative Methods - Kalman Filter`: Dynamic hedge ratios & noise filtering.
- `UTILS - Quantitative Methods - Stochastic Processes`: Geometric Brownian Motion & Monte Carlo.
- `UTILS - Quantitative Methods - Statistics`: Hypothesis testing, stationarity, and cointegration.
- `UTILS - Quantitative Methods - Regression`: Factor models & Alpha generation.
- `UTILS - Quantitative Methods - Linear Algebra`: Portfolio optimization & risk modelling.

### Level 5: Strategies & Finance
*Applied quantitative finance.*
- `UTILS - Strategies - Pairs Trading`: Statistical arbitrage & mean reversion.
- `UTILS - Strategies - Momentum Trading`: Trend following & signal generation.
- `UTILS - Black-Scholes Option Pricing`: Greeks, implied volatility, & derivatives pricing.
- `UTILS - Finance - Volatility Calculator`: Parkinson, Garman-Klass, & EWMA estimators.
- `UTILS - Portfolio Optimizer`: Efficient Frontier, Sharpe Ratio, & Markowitz optimization.
- `UTILS - Risk Metrics`: Value at Risk (VaR), CVaR, Drawdown, & Sortino Ratio.
- `UTILS - Technical Indicators`: Custom implementations of RSI, MACD, Bollinger Bands.

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

**Example: Learning Context Managers**
```bash
cd "UTILS - Advanced Python - Context Managers"
python context_managers_tutorial.py
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

**Learn-Quant v1.8.0**
*Quantitative Finance | Algorithmic Trading | Python Mastery*
**Maintained by MeridianAlgo**
