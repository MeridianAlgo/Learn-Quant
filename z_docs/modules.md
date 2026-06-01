# All Modules

Complete index of all Learn-Quant lessons and utilities.

## Python Fundamentals

### [Python Basics - Comprehensions](Python Basics - Comprehensions.md)
**Python Basics – Comprehensions**

Comprehensions are Python's most elegant way to transform data—replacing loops with readable, performant one-liners. This module teaches **list, dict, set comprehensions**, **generator expressions**, and **functional tools** (`map`, `filter`, `reduce`, `accumulate`) used constantly in quantitative finance for data cleaning, signal generation, and portfolio calculations.

### [Python Basics - Control Flow](Python Basics - Control Flow.md)
**Python Basics – Control Flow**

Control flow structures (`if/elif/else`, `for`, `while`, comprehensions, `break`, `continue`) are the foundation of all algorithms. This module teaches how to make decisions, iterate through data, and build the logic patterns used in trading systems, backtests, and risk management tools.

### [Python Basics - Functions](Python Basics - Functions.md)
**Python Basics – Functions Utility**

This utility teaches Python functions - the building blocks of modular, reusable code. Learn to write efficient trading algorithms and financial tools using proper function design.

### [Python Basics - NumPy](Python Basics - NumPy.md)
**Python Basics – NumPy**

Covers the NumPy primitives that appear in virtually every quant codebase — from vectorised return calculations to portfolio variance via the quadratic form. All examples use realistic financial data so the connection between the NumPy API and actual quant work is immediate.

### [Python Basics - Numbers](Python Basics - Numbers.md)
**Python Basics – Numbers Utility**

After completing this lesson, you'll understand:

### [Python Basics - Pandas](Python Basics - Pandas.md)
**Python Basics – Pandas**

Covers the Pandas patterns that power real quant research pipelines — from building a synthetic OHLCV DataFrame through rolling indicators, resampling, groupby analysis, and a simple SMA-crossover backtest. Every example is grounded in price data so the link from Pandas API to practical quant work stays concrete.

### [Python Basics - Strings](Python Basics - Strings.md)
**Python Basics – Strings Utility**

This beginner-friendly utility introduces Python string fundamentals through hands-on examples. It is perfect for newcomers following the learning path in `Documentation/Programs/level1_fundamentals.py` and looking for extra practice manipulating text data.

## Data Structures

### [Data Structures - Arrays](Data Structures - Arrays.md)
**Arrays - Complete Guide to NumPy for Beginners and Beyond**

Welcome to the comprehensive guide to NumPy arrays! This utility is designed to help both beginners and experienced Python programmers master array operations for data analysis, scientific computing, and quantitative finance.

### [Data Structures - Dictionaries](Data Structures - Dictionaries.md)
**Dictionaries - Key-Value Data Structures for Financial Analysis**

This utility provides comprehensive Python dictionary operations essential for financial data organization, lookup tables, and key-value mappings. Dictionaries are the backbone of feature engineering and data lookup in quantitative finance.

### [Data Structures - Lists](Data Structures - Lists.md)
**Data Structures – Lists**

Lists are Python's **most fundamental data structure**—ordered, mutable collections used for storing time series data, portfolio holdings, transaction logs, and any sequence of values. Master list operations and you unlock efficient data processing essential for trading systems and quantitative analysis.

### [Data Structures - Tuples and Sets](Data Structures - Tuples and Sets.md)
**Data Structures – Tuples and Sets**

Tuples and Sets are fundamental Python data structures that complement Lists and Dictionaries. Understanding when to use them is key to writing efficient, Pythonic code for financial applications.

## Algorithms

### [Algorithms - Backtracking](Algorithms - Backtracking.md)
**Algorithms – Backtracking**

Backtracking is a general algorithmic technique for solving problems by building candidates incrementally and abandoning a candidate ("backtracking") as soon as it is determined to violate the problem constraints. It is a systematic form of exhaustive search that prunes the search space to avoid exploring clearly invalid paths.

### [Algorithms - Dynamic Programming](Algorithms - Dynamic Programming.md)
**Algorithms – Dynamic Programming**

Dynamic Programming (DP) is an algorithmic technique for solving problems by breaking them into overlapping subproblems, solving each subproblem once, and storing the result to avoid redundant computation. It converts exponential-time recursive solutions into polynomial-time ones.

### [Algorithms - Graph](Algorithms - Graph.md)
**Algorithms – Graph**

Graph algorithms operate on structures composed of vertices (nodes) and edges (connections). Many financial problems are naturally modelled as graphs: currency markets form weighted directed graphs, asset correlation matrices define undirected weighted graphs, and order routing networks are flow graphs.

### [Algorithms - Machine Learning](Algorithms - Machine Learning.md)
**Algorithms – Machine Learning**

This module implements fundamental machine learning algorithms from scratch using only NumPy — no scikit-learn or frameworks. Building these algorithms by hand is the most effective way to understand what happens inside the black boxes used in production trading systems.

### [Algorithms - Searching](Algorithms - Searching.md)
**Algorithms – Searching**

Searching algorithms find a target value within a data structure. The choice of algorithm determines whether a search takes O(n) time (checking every element) or O(log n) time (dividing the search space in half each step). In latency-sensitive financial systems, this difference is meaningful at scale.

### [Algorithms - Sorting](Algorithms - Sorting.md)
**Sorting Algorithms**

A comprehensive implementation of fundamental sorting algorithms with detailed explanations, complexity analysis, and performance comparisons.

### [Algorithms - String](Algorithms - String.md)
**Algorithms – String**

String algorithms handle efficient manipulation, searching, and analysis of text data. In quantitative finance, string processing is essential for parsing market data feeds, extracting information from news and filings, matching ticker symbols, and cleaning raw data from APIs.

### [Algorithms - Tree](Algorithms - Tree.md)
**Algorithms – Tree**

Tree data structures organise data hierarchically to enable efficient search, insertion, and deletion. Binary Search Trees (BSTs) and their balanced variants (AVL trees, Red-Black trees) are the foundation of many performance-critical systems in finance, including order book matching engines, index structures for time-series databases, and priority queues for event-driven simulations.

## Advanced Python

### [Advanced Python - AsyncIO](Advanced Python - AsyncIO.md)
**AsyncIO for High-Frequency Data**

In quantitative finance, speed is edge. Python's `asyncio` library allows for **concurrency**, letting your program handle multiple tasks (like fetching data from 10 different exchanges) at once, rather than waiting for one to finish before starting the next.

### [Advanced Python - Context Managers](Advanced Python - Context Managers.md)
**Advanced Python – Context Managers**

Context Managers are a powerful Python feature for resource management. They allow you to allocate and release resources precisely when you want to. The most common usage is the `with` statement.

### [Advanced Python - Decorators and Generators](Advanced Python - Decorators and Generators.md)
**Advanced Python – Decorators and Generators**

Decorators and Generators are powerful Python features that separate professional code from beginner scripts. Decorators allow you to modify function behavior cleanly, while Generators enable memory-efficient processing of large financial datasets.

### [Advanced Python - Error Handling](Advanced Python - Error Handling.md)
**Advanced Python – Error Handling**

Robust error handling is what separates a script that crashes overnight from a professional trading system that runs for years. This module teaches you how to anticipate, catch, and manage errors gracefully.

### [Advanced Python - Multiprocessing](Advanced Python - Multiprocessing.md)
**Advanced Python Multiprocessing**

Python Global Interpreter Lock prevents multiple threads from executing Python bytecode at the same time. This makes threads useless for intense algorithmic work. The multiprocessing module bypasses the lock entirely by spawning separate operating system processes. Each process has its own Python interpreter and memory space, enabling true parallelism across all processing cores.

### [Advanced Python - OOP](Advanced Python - OOP.md)
**Advanced Python – Object-Oriented Programming**

Object-Oriented Programming (OOP) is essential for building scalable, maintainable trading systems and financial applications. Learn to organize code using classes, objects, and OOP principles.

## Quantitative Methods

### [Quantitative Methods - Bootstrap](Quantitative Methods - Bootstrap.md)
**Bootstrap Resampling**

The bootstrap estimates the sampling distribution of **any** statistic by resampling the observed data with replacement — no normality assumption required. It is the honest way to put confidence intervals around backtest metrics like Sharpe ratio, mean return, or maximum drawdown.

### [Quantitative Methods - Cointegration](Quantitative Methods - Cointegration.md)
**Cointegration & Pairs Trading Foundations**

Cointegration: two non-stationary series whose **linear combination is stationary**. Backbone of statistical arbitrage and pairs trading.

### [Quantitative Methods - Copulas](Quantitative Methods - Copulas.md)
**Quantitative Methods - Copulas**

This module demonstrates the concept of Copulas, specifically the Gaussian Copula, used in quantitative finance to model the dependency structure between multivariate random variables.

### [Quantitative Methods - Factor Models](Quantitative Methods - Factor Models.md)
**Quantitative Methods – Factor Models**

Factor models explain asset returns as a linear combination of systematic **factors** plus a stock-specific residual. The **Fama-French 3-Factor Model (1992)** extended CAPM by adding two well-documented risk premia: the **Size premium** (SMB) and the **Value premium** (HML), dramatically improving the explanation of cross-sectional stock returns.

### [Quantitative Methods - GARCH](Quantitative Methods - GARCH.md)
**GARCH Volatility Models**

GARCH (Generalized Autoregressive Conditional Heteroskedasticity) captures **volatility clustering** — high-volatility days tend to follow high-volatility days. Used for risk forecasting, option pricing, and VaR.

### [Quantitative Methods - Interest Rate Models](Quantitative Methods - Interest Rate Models.md)
**Short Rate Interest Rate Models**

Continuous-time models for the evolution of the short (instantaneous) interest rate. Used for bond pricing, interest rate derivatives, and yield curve modeling.

### [Quantitative Methods - Kalman Filter](Quantitative Methods - Kalman Filter.md)
**Multi-Purpose Kalman Filter**

This module provides a pure Python implementation of a 1-Dimensional Kalman Filter. Kalman filters are recursive algorithms used to estimate the state of a linear dynamic system from a series of noisy measurements.

### [Quantitative Methods - Linear Algebra](Quantitative Methods - Linear Algebra.md)
**Quantitative Methods – Linear Algebra**

Linear algebra is the mathematical foundation for portfolio optimization, risk modeling, factor analysis, and quantitative finance. This utility teaches essential concepts through practical financial applications.

### [Quantitative Methods - Optimization](Quantitative Methods - Optimization.md)
**Quantitative Methods – Optimization**

Optimization is the mathematical engine behind modern finance. From finding the best portfolio weights to calibrating complex models, optimization techniques are essential for quantitative analysts.

### [Quantitative Methods - Performance Analysis](Quantitative Methods - Performance Analysis.md)
**Performance Analysis Utilities**

This module provides quantitative performance metrics to evaluate risk-adjusted returns and the quality of investment strategies. Beyond simple metrics like the Sharpe Ratio, these tools help quants analyze tail risk, active management skill, and the statistical properties of return series.

### [Quantitative Methods - Principal Component Analysis](Quantitative Methods - Principal Component Analysis.md)
**Principal Component Analysis (PCA)**

PCA finds the orthogonal directions that explain the most variance in a dataset. In finance it powers **yield-curve decomposition** (level/slope/curvature), **statistical factor extraction**, **dimensionality reduction**, and **covariance de-noising**.

### [Quantitative Methods - Regime Detection](Quantitative Methods - Regime Detection.md)
**Market Regime Detection**

Identifies distinct market states (bull/bear, low/high volatility) using statistical methods. Regime-aware strategies adapt parameters to the current market environment.

### [Quantitative Methods - Regression Analysis](Quantitative Methods - Regression Analysis.md)
**Quantitative Methods – Regression Analysis**

Regression analysis is the statistical "Swiss Army Knife" of quantitative finance. It allows you to quantify relationships between variables, such as how a stock moves relative to the market (Beta) or how factors drive returns.

### [Quantitative Methods - Statistics](Quantitative Methods - Statistics.md)
**Statistics - Essential Statistical Analysis for Quantitative Finance**

This utility provides comprehensive statistical analysis tools essential for quantitative finance, risk management, and investment analysis. Statistics forms the foundation for understanding financial data patterns, risk assessment, and predictive modeling.

### [Quantitative Methods - Stochastic Processes](Quantitative Methods - Stochastic Processes.md)
**Quantitative Methods – Stochastic Processes**

Stochastic processes are mathematical models for random systems evolving over time. In finance, they are used to model asset prices, interest rates, and volatility for pricing derivatives and managing risk.

### [Quantitative Methods - TVM](Quantitative Methods - TVM.md)
**Time Value of Money (TVM) - Core Financial Calculations**

This utility provides comprehensive Time Value of Money (TVM) calculations essential for financial analysis, investment evaluation, and capital budgeting. TVM is the foundation of quantitative finance and corporate finance.

### [Quantitative Methods - Time Series](Quantitative Methods - Time Series.md)
**Quantitative Methods – Time Series Utility**

This utility introduces core time-series techniques used in quantitative finance. It serves as a bridge between the intermediate and advanced curriculum (`Documentation/Programs/level3_financial.py` and `level4_advanced.py`) and gives you reusable helpers for analyzing historical price data.

## Options, Derivatives & Finance

### [Advanced Options Pricing](Advanced Options Pricing.md)
**Advanced Options Pricing**

This module covers advanced mathematical techniques for pricing financial derivatives. The focus is on models beyond the standard assumptions. Rather than assuming constant volatility, we explore dynamic and local volatility models. These models are crucial for correctly valuing exotic options and managing the risks of complex derivatives portfolios.

### [Black-Scholes Option Pricing](Black-Scholes Option Pricing.md)
**Black-Scholes Option Pricing Utility**

This module lets you price basic stock options (calls and puts) using the Black-Scholes formula, a foundation of modern financial analysis.

### [Bond Price and Yield](Bond Price and Yield.md)
**Bond Price and Yield Calculator**

This utility lets you calculate the fair price of a bond or estimate its yield to maturity (YTM), two of the most basic (and important!) ideas in investing.

### [CAPM](CAPM.md)
**CAPM (Capital Asset Pricing Model) Utility**

This module lets you calculate the expected return of any stock or portfolio according to CAPM, a core idea in modern finance for pricing risky assets.

### [Discounted Cash Flow (DCF)](Discounted Cash Flow (DCF).md)
**Discounted Cash Flow (DCF) Calculator**

This tool calculates the present value of a series of future cash flows—the basic principle behind valuing businesses, real estate, projects, and stocks!

### [Dividend Tracker](Dividend Tracker.md)
**Dividend Tracker Utility (NO API)**

**This utility does NOT use any external APIs.** All data is managed locally for learning and experimentation.

### [Finance - Beta Calculator](Finance - Beta Calculator.md)
**Beta Calculator – Comprehensive Guide**

**Beta** measures how much a stock or portfolio moves compared to the overall market.

### [Finance - Correlation Analysis](Finance - Correlation Analysis.md)
**Correlation Analysis**

Analyze correlations between financial instruments for portfolio construction and risk management.

### [Finance - Covariance Estimation](Finance - Covariance Estimation.md)
**Robust Covariance Estimation**

Sample covariance is noisy and often poorly conditioned with many assets. Shrinkage estimators blend sample covariance with a structured target for more stable portfolio optimization.

### [Finance - Credit Risk](Finance - Credit Risk.md)
**Merton Credit Risk Model**

The Merton (1974) structural credit model treats a firm's **equity as a call option on its assets**. Default occurs when asset value falls below debt face value at maturity.

### [Finance - Duration Convexity](Finance - Duration Convexity.md)
**Bond Duration, Convexity, and DV01**

Fixed income sensitivity measures that quantify how bond prices respond to changes in interest rates.

### [Finance - Exotic Options](Finance - Exotic Options.md)
**Exotic Options Pricing**

Monte Carlo pricing for path-dependent options that have no simple closed-form solution (or where the path matters, not just the terminal price).

### [Finance - Expected Shortfall](Finance - Expected Shortfall.md)
**Expected Shortfall (CVaR)**

Expected Shortfall (ES), also called Conditional Value at Risk (CVaR), measures the **expected loss given that losses exceed the VaR threshold**. It is a coherent risk measure — unlike VaR, it captures tail severity, not just frequency.

### [Finance - FX Tools](Finance - FX Tools.md)
**FX (Foreign Exchange) Tools**

Core analytics for foreign exchange markets: no-arbitrage pricing, option valuation, and cross-rate calculations.

### [Finance - Greeks Calculator](Finance - Greeks Calculator.md)
**Finance – Greeks Calculator**

The Options Greeks measure the sensitivity of an option's price to changes in underlying market parameters. They are the primary tools used by options traders and risk managers to understand, hedge, and monitor options positions.

### [Finance - Kelly Criterion](Finance - Kelly Criterion.md)
**Kelly Criterion Position Sizing**

The Kelly Criterion determines the **optimal fraction of capital to allocate** to maximize the long-run geometric growth rate of wealth.

### [Finance - Options Strategies](Finance - Options Strategies.md)
**Finance – Options Strategies**

Options strategies combine multiple option legs (calls and puts at different strikes and expiries) to create specific risk/reward profiles. Rather than taking a directional bet with a single option, multi-leg strategies allow traders to express nuanced views on direction, volatility, time decay, and risk limits.

### [Finance - Position Sizing](Finance - Position Sizing.md)
**Finance – Position Sizing**

**Position sizing is the most underrated skill in quantitative trading.** A strategy with a mediocre edge and excellent position sizing will outperform a brilliant strategy with reckless sizing. This module covers four fundamental frameworks every trader and quant must understand before risking real capital.

### [Finance - Transaction Cost Analysis](Finance - Transaction Cost Analysis.md)
**Transaction Cost Analysis (TCA)**

Tools for measuring execution quality and estimating market impact. TCA is essential for evaluating whether a strategy's theoretical alpha survives real-world trading costs.

### [Finance - Volatility Calculator](Finance - Volatility Calculator.md)
**Volatility Calculator**

Calculate various volatility metrics for financial instruments.

### [Finance - Yield Curve](Finance - Yield Curve.md)
**Finance – Yield Curve**

The yield curve is the most closely watched chart in global finance. It plots interest rates (yields) across different maturities for bonds of equal credit quality — most commonly US Treasury bonds. Its shape and movements drive pricing for virtually every financial asset, from mortgages to corporate bonds to equity discount rates.

### [Monte Carlo Simulation - JavaScript](Monte Carlo Simulation - JavaScript.md)
**Monte Carlo Simulation – JavaScript**

A pure JavaScript Monte Carlo engine for portfolio simulation and European option pricing via geometric Brownian motion (GBM). Implements correlated multi-asset paths using Cholesky decomposition, antithetic variates for variance reduction, and VaR/CVaR estimation from the simulated return distribution. No external dependencies — runs directly in Node.js.

### [Options Chain Simulator](Options Chain Simulator.md)
**Options Chain Simulator Utility (NO API)**

**This utility does NOT use any external APIs.** All calculations are done locally for learning and experimentation.

### [Options Pricing - JavaScript](Options Pricing - JavaScript.md)
**Options Pricing – JavaScript**

A pure JavaScript implementation of the Black-Scholes European options pricing model with all five Greeks and implied volatility via bisection. No external dependencies — runs directly in Node.js and can be imported as a module into any JS project.

### [Technical Indicators](Technical Indicators.md)
**Technical Indicators Calculator Utility (NO API)**

**This utility does NOT use any external APIs.** All calculations are done locally for learning and experimentation.

## Risk & Performance

### [Finance - Information Ratio](Finance - Information Ratio.md)
**Information Ratio & Active Management Metrics**

When a portfolio is judged **against a benchmark**, what matters is how much it beat the benchmark by — and how *reliably*. These are the core metrics of active management: active return, tracking error, Information Ratio, and the appraisal ratio.

### [Finance - Performance Attribution](Finance - Performance Attribution.md)
**Performance Attribution**

Brinson decomposition splits portfolio active return into **allocation** and **selection** effects — answering *"did we beat the benchmark by picking the right sectors or the right stocks?"*

### [Risk Metrics](Risk Metrics.md)
**Risk Metrics Summary Utility**

This module gives you quick, professional stats about risk in any list or array of investment returns. It's used by investors, analysts, and students everywhere!

### [Risk Metrics - Drawdown Analysis](Risk Metrics - Drawdown Analysis.md)
**Drawdown Analysis**

Comprehensive drawdown metrics for quantifying portfolio loss risk over time. Drawdown measures capture both the **depth** and **duration** of losses — dimensions VaR ignores.

### [Risk Metrics - Stress Testing](Risk Metrics - Stress Testing.md)
**Portfolio Stress Testing**

Stress tests answer: *"What happens if 2008 repeats?"* or *"How big a shock kills the portfolio?"* Required by Basel III, CCAR, and most institutional risk frameworks.

### [Sharpe and Sortino Ratio](Sharpe and Sortino Ratio.md)
**Sharpe and Sortino Ratio Calculator**

This utility offers easy-to-use Python functions to calculate Sharpe and Sortino ratios for financial returns. These ratios help you understand whether a series of investment returns is attractive on a risk-adjusted basis.

### [Value at Risk (VaR)](Value at Risk (VaR).md)
**Value at Risk (VaR) Calculator**

This utility lets you estimate the potential losses on a portfolio or investment using Value at Risk (VaR), one of the most important tools in financial risk management.

## Portfolio Management

### [Monte Carlo Portfolio Simulator](Monte Carlo Portfolio Simulator.md)
**Monte Carlo Portfolio Simulator**

This utility helps you forecast possible futures for a portfolio using random simulations—a key idea in finance, risk management, and statistics!

### [Portfolio Management](Portfolio Management.md)
**Portfolio Management Utilities**

This folder contains utilities for portfolio management, risk analysis, and investment optimization.

### [Portfolio Management - Black Litterman](Portfolio Management - Black Litterman.md)
**Black-Litterman Portfolio Optimization**

The Black-Litterman (1990) model addresses the instability of mean-variance optimization by blending **market equilibrium returns** with **investor views** using Bayesian updating.

### [Portfolio Management - Risk Parity](Portfolio Management - Risk Parity.md)
**Risk Parity Portfolio Construction**

Risk parity builds a portfolio where **every asset contributes the same amount of risk** to the total — not the same amount of capital. A naive 60/40 stock/bond portfolio is ~90% *equity risk* despite being only 60% equity *capital*; risk parity fixes that imbalance.

### [Portfolio Optimizer](Portfolio Optimizer.md)
**Portfolio Optimizer (Mean-Variance)**

This utility helps you find the best mix of assets for a portfolio, balancing risk and return using the foundation of Modern Portfolio Theory (MPT).

### [Portfolio Tracker](Portfolio Tracker.md)
**Portfolio Tracker Utility (USES yfinance API)**

**This utility uses the yfinance API to fetch current prices automatically.** All other calculations and data are managed locally for learning and experimentation.

## Strategies

### [Order Execution Simulator](Order Execution Simulator.md)
**Order Execution Simulator Utility (NO API)**

**This utility does NOT use any external APIs.** All trades and portfolio data are managed locally for learning and experimentation.

### [Strategies - Market Making](Strategies - Market Making.md)
**Avellaneda-Stoikov Market Making Model**

Implementation of the **Avellaneda-Stoikov (2008)** continuous-time market making model. A dealer posts bid/ask quotes to maximize expected PnL while penalizing inventory accumulation.

### [Strategies - Mean Reversion](Strategies - Mean Reversion.md)
**Strategies – Mean Reversion**

Mean reversion is the statistical tendency for an asset's price to return to its historical average after deviating from it. While Momentum strategies bet on *continuation*, Mean Reversion strategies bet on *reversal* — buying when something is "too cheap" and selling when it is "too expensive" relative to recent history.

### [Strategies - Momentum Trading](Strategies - Momentum Trading.md)
**Strategies – Momentum Trading**

Momentum trading is a strategy that capitalizes on the continuance of existing trends in the market. The core philosophy is "buy high, sell higher." If an asset's price is rising strongly, momentum traders assume it will continue to rise.

### [Strategies - Pairs Trading](Strategies - Pairs Trading.md)
**Pairs Trading Strategy**

This module demonstrates a statistical arbitrage strategy known as Pairs Trading. It identifies two assets that move together and trades the convergence of their spread. When the correlation weakens temporarily, executing trades on both assets allows for capturing profits as they revert to their historical relationship. This quantitative technique relies strictly on mathematical relationships rather than fundamental valuation.

### [Strategies - Statistical Arbitrage](Strategies - Statistical Arbitrage.md)
**Strategies - Statistical Arbitrage**

This module demonstrates a basic Statistical Arbitrage strategy, specifically pairs trading.

### [Strategies - Trend Following](Strategies - Trend Following.md)
**Trend Following**

Trend-following: ride momentum with discipline. Backbone of CTAs and managed-futures funds (AHL, Winton, Man, MLP). Profits from extended directional moves; pays for it during chop.

## AI & Machine Learning

### [AI Development](AI Development.md)
**Gemini API Chatbot**

This project provides simple command-line chatbots for Google's Gemini API in both Python and Node.js.

### [Learning Platform](Learning Platform.md)
**Interactive Python Learning Platform**

An all-in-one learning hub that delivers progressive Python lessons through both a guided CLI and a hostable Flask web interface. Lessons combine narrative walkthroughs, executable code examples, mini quizzes, and follow-up practice ideas geared toward aspiring quantitative developers.

### [Machine Learning - Random Forest](Machine Learning - Random Forest.md)
**Machine Learning - Random Forest**

This module provides a basic implementation of a Random Forest Predictor for quantitative finance. It uses scikit-learn's `RandomForestRegressor` to predict time series data or returns based on a set of features.

### [Machine Learning Time Series](Machine Learning Time Series.md)
**Machine Learning for Sequential Financial Data**

Applying incredibly sophisticated statistical and advanced computational matrix calculating algorithms to historical sequential asset prices explicitly enables quantitative researchers to discover heavily latent non linear correlation patterns. Standard basic linear techniques lack the internal theoretical mapping memory required to fully process continuous progression data natively. Therefore, explicit sequential data pattern prediction necessitates deeply specialized memory architectures uniquely capable of successfully retaining vast contextual numerical memory safely across thousands of chronologically independent market observations simultaneously.

### [Reinforcement Learning Q Learning](Reinforcement Learning Q Learning.md)
**Reinforcement Learning for Quantitative Finance**

This module extensively covers the core mathematical algorithms necessary to construct entirely autonomous quantitative execution agents. Rather than relying on rigid statistical parameters or explicit condition based trading logic, reinforcement learning allows an agent to discover the most optimal sequences of action through continuous simulated trial and error. The intelligent agent dynamically interprets complex environmental states and receives explicit scalar rewards or punitive penalties based directly upon its transactional profitability and risk management threshold maintenance. Over thousands of episodes, the model organically maps the market mechanics to develop a mathematically optimal trading policy without human intervention.

### [Sentiment Analysis on News](Sentiment Analysis on News.md)
**Sentiment Analysis on News Utility (NO API)**

**This utility does NOT use any external APIs.** All sentiment analysis is done locally using a simple rule-based approach for learning and experimentation.

## Market Microstructure

### [High Frequency Trading](High Frequency Trading.md)
**High Frequency Trading**

High Frequency Trading (HFT) encompasses algorithmic strategies that execute a large number of orders at extremely high speeds — typically microseconds to milliseconds. HFT firms compete primarily on latency: the fastest participant to react to new information captures the profit.

### [Market Microstructure](Market Microstructure.md)
**Market Microstructure**

Market microstructure studies how trading mechanisms — the rules, protocols, and participants in a market — affect price formation, liquidity, and transaction costs. Understanding microstructure is essential for designing realistic execution algorithms, building order books, estimating market impact, and analysing bid-ask spreads.

## Utilities & Tools

### [Core Utilities](Core Utilities.md)
**Core Utilities**

This folder contains core mathematical and date/time utilities that form the foundation for quantitative finance calculations.

### [Currency Converter](Currency Converter.md)
**Currency Converter Utility (NO API)**

**This utility does NOT use any external APIs.** All exchange rates are entered manually for learning and experimentation.

### [Data Processing](Data Processing.md)
**Data Processing Utilities**

This folder contains utilities for data processing, validation, and manipulation in financial applications.

### [Economic Calendar](Economic Calendar.md)
**Economic Calendar Simulator Utility (NO API)**

**This utility does NOT use any external APIs.** All data is managed locally for learning and experimentation.

### [Historical Data](Historical Data.md)
**Alpaca Historical Data Fetcher**

This Node.js script fetches historical bars (OHLCV data) for stocks or crypto from the Alpaca Market Data API. It prompts you for the symbol type, symbol, timeframe, and date range, then displays the results in JSON format.

### [Logging](Logging.md)
**Logging Utilities**

This project provides simple logging utilities in both Python and JavaScript. You can add, read, edit, and delete log entries using either language. All logs are stored in a file named `log.txt` in the same directory.

### [Market Data](Market Data.md)
**Market Data Utilities**

This folder contains utilities for processing, analyzing, and fetching market data for financial applications.

### [News Fetching](News Fetching.md)
**Google News Fetcher**

This utility provides a Google News headline scraper using the `google-news-json` package. It no longer requires any API keys, making it ideal for beginners who want to experiment with news-driven trading ideas or sentiment analysis without signing up for external services.

### [System Utilities](System Utilities.md)
**System Utilities**

This folder contains utilities for system-level operations, file management, and configuration in financial applications.

### [Websocket Connection](Websocket Connection.md)
**WebSocket Connection Utilities**

This project provides WebSocket clients for connecting to various financial data providers, including YFLive and Finnhub. These utilities are designed for real-time market data streaming and analysis.

