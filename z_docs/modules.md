# All Modules

Every Learn-Quant lesson, grouped by track. Each card links to the full write-up with runnable code and worked examples.

## :material-language-python: Python Fundamentals

*Core Python for financial analysis — start here if you are new to code.*

<div class="grid cards" markdown>

-   __[Python Basics - Comprehensions](Python Basics - Comprehensions.md)__

    <span class="lq-badge lq-beginner">Beginner</span><span class="lq-badge lq-lang">Python</span>

    Comprehensions are Python's most elegant way to transform data—replacing loops with readable, performant one-liners. This module teaches **list, dict, set comprehensions**, **generator expressions**, and **functional tools** (`map`, `filter`, `reduce`, `accumulate`) used constantly in quantitative finance for data cleaning, signal generation, and portfolio calculations.

-   __[Python Basics - Control Flow](Python Basics - Control Flow.md)__

    <span class="lq-badge lq-beginner">Beginner</span><span class="lq-badge lq-lang">Python</span>

    Control flow structures (`if/elif/else`, `for`, `while`, comprehensions, `break`, `continue`) are the foundation of all algorithms. This module teaches how to make decisions, iterate through data, and build the logic patterns used in trading systems, backtests, and risk management tools.

-   __[Python Basics - Dates and Times](Python Basics - Dates and Times.md)__

    <span class="lq-badge lq-beginner">Beginner</span><span class="lq-badge lq-lang">Python</span>

    Markets run on a calendar, not a clock. Interest accrues over **days**, options

-   __[Python Basics - Functions](Python Basics - Functions.md)__

    <span class="lq-badge lq-beginner">Beginner</span><span class="lq-badge lq-lang">Python</span>

    This utility teaches Python functions - the building blocks of modular, reusable code. Learn to write efficient trading algorithms and financial tools using proper function design.

-   __[Python Basics - NumPy](Python Basics - NumPy.md)__

    <span class="lq-badge lq-beginner">Beginner</span><span class="lq-badge lq-lang">Python</span>

    Covers the NumPy primitives that appear in virtually every quant codebase — from vectorised return calculations to portfolio variance via the quadratic form. All examples use realistic financial data so the connection between the NumPy API and actual quant work is immediate.

-   __[Python Basics - Numbers](Python Basics - Numbers.md)__

    <span class="lq-badge lq-beginner">Beginner</span><span class="lq-badge lq-lang">Python</span>

    After completing this lesson, you'll understand:

-   __[Python Basics - Pandas](Python Basics - Pandas.md)__

    <span class="lq-badge lq-beginner">Beginner</span><span class="lq-badge lq-lang">Python</span>

    Covers the Pandas patterns that power real quant research pipelines — from building a synthetic OHLCV DataFrame through rolling indicators, resampling, groupby analysis, and a simple SMA-crossover backtest. Every example is grounded in price data so the link from Pandas API to practical quant work stays concrete.

-   __[Python Basics - Strings](Python Basics - Strings.md)__

    <span class="lq-badge lq-beginner">Beginner</span><span class="lq-badge lq-lang">Python</span>

    This beginner-friendly utility introduces Python string fundamentals through hands-on examples. It is perfect for newcomers following the learning path in `Documentation/Programs/level1_fundamentals.py` and looking for extra practice manipulating text data.

</div>

## :material-database-outline: Data Structures

*The right container for the job: arrays, lists, dicts, sets on market data.*

<div class="grid cards" markdown>

-   __[Data Structures - Arrays](Data Structures - Arrays.md)__

    <span class="lq-badge lq-beginner">Beginner</span><span class="lq-badge lq-lang">Python</span>

    Welcome to the comprehensive guide to NumPy arrays! This utility is designed to help both beginners and experienced Python programmers master array operations for data analysis, scientific computing, and quantitative finance.

-   __[Data Structures - Dictionaries](Data Structures - Dictionaries.md)__

    <span class="lq-badge lq-beginner">Beginner</span><span class="lq-badge lq-lang">Python</span>

    This utility provides comprehensive Python dictionary operations essential for financial data organization, lookup tables, and key-value mappings. Dictionaries are the backbone of feature engineering and data lookup in quantitative finance.

-   __[Data Structures - Lists](Data Structures - Lists.md)__

    <span class="lq-badge lq-beginner">Beginner</span><span class="lq-badge lq-lang">Python</span>

    Lists are Python's **most fundamental data structure**—ordered, mutable collections used for storing time series data, portfolio holdings, transaction logs, and any sequence of values. Master list operations and you unlock efficient data processing essential for trading systems and quantitative analysis.

-   __[Data Structures - Tuples and Sets](Data Structures - Tuples and Sets.md)__

    <span class="lq-badge lq-beginner">Beginner</span><span class="lq-badge lq-lang">Python</span>

    Tuples and Sets are fundamental Python data structures that complement Lists and Dictionaries. Understanding when to use them is key to writing efficient, Pythonic code for financial applications.

</div>

## :material-sitemap-outline: Algorithms

*Classic computer-science algorithms applied to price and order data.*

<div class="grid cards" markdown>

-   __[Algorithms - Backtracking](Algorithms - Backtracking.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    Backtracking is a general algorithmic technique for solving problems by building candidates incrementally and abandoning a candidate ("backtracking") as soon as it is determined to violate the problem constraints. It is a systematic form of exhaustive search that prunes the search space to avoid exploring clearly invalid paths.

-   __[Algorithms - Dynamic Programming](Algorithms - Dynamic Programming.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    Dynamic Programming (DP) is an algorithmic technique for solving problems by breaking them into overlapping subproblems, solving each subproblem once, and storing the result to avoid redundant computation. It converts exponential-time recursive solutions into polynomial-time ones.

-   __[Algorithms - Graph](Algorithms - Graph.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    Graph algorithms operate on structures composed of vertices (nodes) and edges (connections). Many financial problems are naturally modelled as graphs: currency markets form weighted directed graphs, asset correlation matrices define undirected weighted graphs, and order routing networks are flow graphs.

-   __[Algorithms - Machine Learning](Algorithms - Machine Learning.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    This module implements fundamental machine learning algorithms from scratch using only NumPy — no scikit-learn or frameworks. Building these algorithms by hand is the most effective way to understand what happens inside the black boxes used in production trading systems.

-   __[Algorithms - Searching](Algorithms - Searching.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    Searching algorithms find a target value within a data structure. The choice of algorithm determines whether a search takes O(n) time (checking every element) or O(log n) time (dividing the search space in half each step). In latency-sensitive financial systems, this difference is meaningful at scale.

-   __[Algorithms - Sorting](Algorithms - Sorting.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    A comprehensive implementation of fundamental sorting algorithms with detailed explanations, complexity analysis, and performance comparisons.

-   __[Algorithms - String](Algorithms - String.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    String algorithms handle efficient manipulation, searching, and analysis of text data. In quantitative finance, string processing is essential for parsing market data feeds, extracting information from news and filings, matching ticker symbols, and cleaning raw data from APIs.

-   __[Algorithms - Tree](Algorithms - Tree.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    Tree data structures organise data hierarchically to enable efficient search, insertion, and deletion. Binary Search Trees (BSTs) and their balanced variants (AVL trees, Red-Black trees) are the foundation of many performance-critical systems in finance, including order book matching engines, index structures for time-series databases, and priority queues for event-driven simulations.

</div>

## :material-cog-outline: Advanced Python

*Production engineering: async, OOP, concurrency, resilient error handling.*

<div class="grid cards" markdown>

-   __[Advanced Python - AsyncIO](Advanced Python - AsyncIO.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    In quantitative finance, speed is edge. Python's `asyncio` library allows for **concurrency**, letting your program handle multiple tasks (like fetching data from 10 different exchanges) at once, rather than waiting for one to finish before starting the next.

-   __[Advanced Python - Context Managers](Advanced Python - Context Managers.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    Context Managers are a powerful Python feature for resource management. They allow you to allocate and release resources precisely when you want to. The most common usage is the `with` statement.

-   __[Advanced Python - Decorators and Generators](Advanced Python - Decorators and Generators.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    Decorators and Generators are powerful Python features that separate professional code from beginner scripts. Decorators allow you to modify function behavior cleanly, while Generators enable memory-efficient processing of large financial datasets.

-   __[Advanced Python - Error Handling](Advanced Python - Error Handling.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    Robust error handling is what separates a script that crashes overnight from a professional trading system that runs for years. This module teaches you how to anticipate, catch, and manage errors gracefully.

-   __[Advanced Python - Multiprocessing](Advanced Python - Multiprocessing.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    Python Global Interpreter Lock prevents multiple threads from executing Python bytecode at the same time. This makes threads useless for intense algorithmic work. The multiprocessing module bypasses the lock entirely by spawning separate operating system processes. Each process has its own Python interpreter and memory space, enabling true parallelism across all processing cores.

-   __[Advanced Python - OOP](Advanced Python - OOP.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    Object-Oriented Programming (OOP) is essential for building scalable, maintainable trading systems and financial applications. Learn to organize code using classes, objects, and OOP principles.

</div>

## :material-function-variant: Quantitative Methods

*The mathematics underpinning modern finance, implemented from first principles.*

<div class="grid cards" markdown>

-   __[Quantitative Methods - Bayesian Inference](Quantitative Methods - Bayesian Inference.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    A strategy wins 7 of its first 10 trades. Is its true win rate 70%? Almost

-   __[Quantitative Methods - Bootstrap](Quantitative Methods - Bootstrap.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    The bootstrap estimates the sampling distribution of **any** statistic by resampling the observed data with replacement — no normality assumption required. It is the honest way to put confidence intervals around backtest metrics like Sharpe ratio, mean return, or maximum drawdown.

-   __[Quantitative Methods - Cointegration](Quantitative Methods - Cointegration.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    Cointegration: two non-stationary series whose **linear combination is stationary**. Backbone of statistical arbitrage and pairs trading.

-   __[Quantitative Methods - Copulas](Quantitative Methods - Copulas.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    This module demonstrates the concept of Copulas, specifically the Gaussian Copula, used in quantitative finance to model the dependency structure between multivariate random variables.

-   __[Quantitative Methods - Extreme Value Theory](Quantitative Methods - Extreme Value Theory.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    Most risk models assume returns are normally distributed. They are not —

-   __[Quantitative Methods - Factor Models](Quantitative Methods - Factor Models.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    Factor models explain asset returns as a linear combination of systematic **factors** plus a stock-specific residual. The **Fama-French 3-Factor Model (1992)** extended CAPM by adding two well-documented risk premia: the **Size premium** (SMB) and the **Value premium** (HML), dramatically improving the explanation of cross-sectional stock returns.

-   __[Quantitative Methods - GARCH](Quantitative Methods - GARCH.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    GARCH (Generalized Autoregressive Conditional Heteroskedasticity) captures **volatility clustering** — high-volatility days tend to follow high-volatility days. Used for risk forecasting, option pricing, and VaR.

-   __[Quantitative Methods - Interest Rate Models](Quantitative Methods - Interest Rate Models.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    Continuous-time models for the evolution of the short (instantaneous) interest rate. Used for bond pricing, interest rate derivatives, and yield curve modeling.

-   __[Quantitative Methods - Kalman Filter](Quantitative Methods - Kalman Filter.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    This module provides a pure Python implementation of a 1-Dimensional Kalman Filter. Kalman filters are recursive algorithms used to estimate the state of a linear dynamic system from a series of noisy measurements.

-   __[Quantitative Methods - Linear Algebra](Quantitative Methods - Linear Algebra.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    Linear algebra is the mathematical foundation for portfolio optimization, risk modeling, factor analysis, and quantitative finance. This utility teaches essential concepts through practical financial applications.

-   __[Quantitative Methods - Numerical Methods](Quantitative Methods - Numerical Methods.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    Most of the formulas in finance cannot be solved with algebra. There is no

-   __[Quantitative Methods - Optimization](Quantitative Methods - Optimization.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    Optimization is the mathematical engine behind modern finance. From finding the best portfolio weights to calibrating complex models, optimization techniques are essential for quantitative analysts.

-   __[Quantitative Methods - Performance Analysis](Quantitative Methods - Performance Analysis.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    This module provides quantitative performance metrics to evaluate risk-adjusted returns and the quality of investment strategies. Beyond simple metrics like the Sharpe Ratio, these tools help quants analyze tail risk, active management skill, and the statistical properties of return series.

-   __[Quantitative Methods - Principal Component Analysis](Quantitative Methods - Principal Component Analysis.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    PCA finds the orthogonal directions that explain the most variance in a dataset. In finance it powers **yield-curve decomposition** (level/slope/curvature), **statistical factor extraction**, **dimensionality reduction**, and **covariance de-noising**.

-   __[Quantitative Methods - Regime Detection](Quantitative Methods - Regime Detection.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    Identifies distinct market states (bull/bear, low/high volatility) using statistical methods. Regime-aware strategies adapt parameters to the current market environment.

-   __[Quantitative Methods - Regression Analysis](Quantitative Methods - Regression Analysis.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    Regression analysis is the statistical "Swiss Army Knife" of quantitative finance. It allows you to quantify relationships between variables, such as how a stock moves relative to the market (Beta) or how factors drive returns.

-   __[Quantitative Methods - Statistics](Quantitative Methods - Statistics.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    This utility provides comprehensive statistical analysis tools essential for quantitative finance, risk management, and investment analysis. Statistics forms the foundation for understanding financial data patterns, risk assessment, and predictive modeling.

-   __[Quantitative Methods - Stochastic Processes](Quantitative Methods - Stochastic Processes.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    Stochastic processes are mathematical models for random systems evolving over time. In finance, they are used to model asset prices, interest rates, and volatility for pricing derivatives and managing risk.

-   __[Quantitative Methods - TVM](Quantitative Methods - TVM.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    This utility provides comprehensive Time Value of Money (TVM) calculations essential for financial analysis, investment evaluation, and capital budgeting. TVM is the foundation of quantitative finance and corporate finance.

-   __[Quantitative Methods - Time Series](Quantitative Methods - Time Series.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    This utility introduces core time-series techniques used in quantitative finance. It serves as a bridge between the intermediate and advanced curriculum (`Documentation/Programs/level3_financial.py` and `level4_advanced.py`) and gives you reusable helpers for analyzing historical price data.

</div>

## :material-chart-bell-curve: Options, Derivatives & Finance

*Pricing, Greeks, fixed income and valuation of financial instruments.*

<div class="grid cards" markdown>

-   __[Advanced Options Pricing](Advanced Options Pricing.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    This module covers advanced mathematical techniques for pricing financial derivatives. The focus is on models beyond the standard assumptions. Rather than assuming constant volatility, we explore dynamic and local volatility models. These models are crucial for correctly valuing exotic options and managing the risks of complex derivatives portfolios.

-   __[Black-Scholes Option Pricing](Black-Scholes Option Pricing.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    This module lets you price basic stock options (calls and puts) using the Black-Scholes formula, a foundation of modern financial analysis.

-   __[Bond Price and Yield](Bond Price and Yield.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    This utility lets you calculate the fair price of a bond or estimate its yield to maturity (YTM), two of the most basic (and important!) ideas in investing.

-   __[CAPM](CAPM.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    CAPM is the idea that won a Nobel Prize and still anchors how the industry

-   __[Discounted Cash Flow (DCF)](Discounted Cash Flow (DCF).md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    This tool calculates the present value of a series of future cash flows—the basic principle behind valuing businesses, real estate, projects, and stocks!

-   __[Dividend Tracker](Dividend Tracker.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    **This utility does NOT use any external APIs.** All data is managed locally for learning and experimentation.

-   __[Finance - Beta Calculator](Finance - Beta Calculator.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    **Beta** measures how much a stock or portfolio moves compared to the overall market.

-   __[Finance - Correlation Analysis](Finance - Correlation Analysis.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    Analyze correlations between financial instruments for portfolio construction and risk management.

-   __[Finance - Covariance Estimation](Finance - Covariance Estimation.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    Sample covariance is noisy and often poorly conditioned with many assets. Shrinkage estimators blend sample covariance with a structured target for more stable portfolio optimization.

-   __[Finance - Credit Risk](Finance - Credit Risk.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    The Merton (1974) structural credit model treats a firm's **equity as a call option on its assets**. Default occurs when asset value falls below debt face value at maturity.

-   __[Finance - Duration Convexity](Finance - Duration Convexity.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    Fixed income sensitivity measures that quantify how bond prices respond to changes in interest rates.

-   __[Finance - Exotic Options](Finance - Exotic Options.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    Monte Carlo pricing for path-dependent options that have no simple closed-form solution (or where the path matters, not just the terminal price).

-   __[Finance - Expected Shortfall](Finance - Expected Shortfall.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    Expected Shortfall (ES), also called Conditional Value at Risk (CVaR), measures the **expected loss given that losses exceed the VaR threshold**. It is a coherent risk measure — unlike VaR, it captures tail severity, not just frequency.

-   __[Finance - FX Tools](Finance - FX Tools.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    Core analytics for foreign exchange markets: no-arbitrage pricing, option valuation, and cross-rate calculations.

-   __[Finance - Greeks Calculator](Finance - Greeks Calculator.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    The Options Greeks measure the sensitivity of an option's price to changes in underlying market parameters. They are the primary tools used by options traders and risk managers to understand, hedge, and monitor options positions.

-   __[Finance - Implied Volatility Surface](Finance - Implied Volatility Surface.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    Black-Scholes turns *volatility into a price*. The market runs the formula

-   __[Finance - Kelly Criterion](Finance - Kelly Criterion.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    The Kelly Criterion determines the **optimal fraction of capital to allocate** to maximize the long-run geometric growth rate of wealth.

-   __[Finance - Options Strategies](Finance - Options Strategies.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    Options strategies combine multiple option legs (calls and puts at different strikes and expiries) to create specific risk/reward profiles. Rather than taking a directional bet with a single option, multi-leg strategies allow traders to express nuanced views on direction, volatility, time decay, and risk limits.

-   __[Finance - Position Sizing](Finance - Position Sizing.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    **Position sizing is the most underrated skill in quantitative trading.** A strategy with a mediocre edge and excellent position sizing will outperform a brilliant strategy with reckless sizing. This module covers four fundamental frameworks every trader and quant must understand before risking real capital.

-   __[Finance - Transaction Cost Analysis](Finance - Transaction Cost Analysis.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    Tools for measuring execution quality and estimating market impact. TCA is essential for evaluating whether a strategy's theoretical alpha survives real-world trading costs.

-   __[Finance - Volatility Calculator](Finance - Volatility Calculator.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    Calculate various volatility metrics for financial instruments.

-   __[Finance - Yield Curve](Finance - Yield Curve.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    The yield curve is the most closely watched chart in global finance. It plots interest rates (yields) across different maturities for bonds of equal credit quality — most commonly US Treasury bonds. Its shape and movements drive pricing for virtually every financial asset, from mortgages to corporate bonds to equity discount rates.

-   __[Monte Carlo Simulation - JavaScript](Monte Carlo Simulation - JavaScript.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">JavaScript</span>

    A pure JavaScript Monte Carlo engine for portfolio simulation and European option pricing via geometric Brownian motion (GBM). Implements correlated multi-asset paths using Cholesky decomposition, antithetic variates for variance reduction, and VaR/CVaR estimation from the simulated return distribution. No external dependencies — runs directly in Node.js.

-   __[Options Chain Simulator](Options Chain Simulator.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    **This utility does NOT use any external APIs.** All calculations are done locally for learning and experimentation.

-   __[Options Pricing - JavaScript](Options Pricing - JavaScript.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">JavaScript</span>

    A pure JavaScript implementation of the Black-Scholes European options pricing model with all five Greeks and implied volatility via bisection. No external dependencies — runs directly in Node.js and can be imported as a module into any JS project.

-   __[Technical Indicators](Technical Indicators.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python · JavaScript</span>

    **This utility does NOT use any external APIs.** All calculations are done locally for learning and experimentation.

</div>

## :material-shield-alert-outline: Risk & Performance

*Measure what can go wrong and how well a strategy actually performed.*

<div class="grid cards" markdown>

-   __[Finance - Information Ratio](Finance - Information Ratio.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    When a portfolio is judged **against a benchmark**, what matters is how much it beat the benchmark by — and how *reliably*. These are the core metrics of active management: active return, tracking error, Information Ratio, and the appraisal ratio.

-   __[Finance - Performance Attribution](Finance - Performance Attribution.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    Brinson decomposition splits portfolio active return into **allocation** and **selection** effects — answering *"did we beat the benchmark by picking the right sectors or the right stocks?"*

-   __[Risk Metrics](Risk Metrics.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    This module gives you quick, professional stats about risk in any list or array of investment returns. It's used by investors, analysts, and students everywhere!

-   __[Risk Metrics - Drawdown Analysis](Risk Metrics - Drawdown Analysis.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    Comprehensive drawdown metrics for quantifying portfolio loss risk over time. Drawdown measures capture both the **depth** and **duration** of losses — dimensions VaR ignores.

-   __[Risk Metrics - Stress Testing](Risk Metrics - Stress Testing.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    Stress tests answer: *"What happens if 2008 repeats?"* or *"How big a shock kills the portfolio?"* Required by Basel III, CCAR, and most institutional risk frameworks.

-   __[Sharpe and Sortino Ratio](Sharpe and Sortino Ratio.md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    This utility offers easy-to-use Python functions to calculate Sharpe and Sortino ratios for financial returns. These ratios help you understand whether a series of investment returns is attractive on a risk-adjusted basis.

-   __[Value at Risk (VaR)](Value at Risk (VaR).md)__

    <span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-lang">Python</span>

    **Value at Risk** is the single most widely quoted number in financial risk

</div>

## :material-briefcase-outline: Portfolio Management

*Construct, optimise and rebalance multi-asset portfolios.*

<div class="grid cards" markdown>

-   __[Monte Carlo Portfolio Simulator](Monte Carlo Portfolio Simulator.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    This utility helps you forecast possible futures for a portfolio using random simulations—a key idea in finance, risk management, and statistics!

-   __[Portfolio Management](Portfolio Management.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    This folder contains utilities for portfolio management, risk analysis, and investment optimization.

-   __[Portfolio Management - Black Litterman](Portfolio Management - Black Litterman.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    The Black-Litterman (1990) model addresses the instability of mean-variance optimization by blending **market equilibrium returns** with **investor views** using Bayesian updating.

-   __[Portfolio Management - Risk Parity](Portfolio Management - Risk Parity.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    Risk parity builds a portfolio where **every asset contributes the same amount of risk** to the total — not the same amount of capital. A naive 60/40 stock/bond portfolio is ~90% *equity risk* despite being only 60% equity *capital*; risk parity fixes that imbalance.

-   __[Portfolio Optimizer](Portfolio Optimizer.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    This utility helps you find the best mix of assets for a portfolio, balancing risk and return using the foundation of Modern Portfolio Theory (MPT).

-   __[Portfolio Tracker](Portfolio Tracker.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    **This utility uses the yfinance API to fetch current prices automatically.** All other calculations and data are managed locally for learning and experimentation.

</div>

## :material-trending-up: Strategies

*End-to-end trading strategies with signals, backtests and execution.*

<div class="grid cards" markdown>

-   __[Order Execution Simulator](Order Execution Simulator.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    **This utility does NOT use any external APIs.** All trades and portfolio data are managed locally for learning and experimentation.

-   __[Strategies - Backtesting Engine](Strategies - Backtesting Engine.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    A backtest answers one question: *if I had traded this rule, what would have

-   __[Strategies - Market Making](Strategies - Market Making.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    Implementation of the **Avellaneda-Stoikov (2008)** continuous-time market making model. A dealer posts bid/ask quotes to maximize expected PnL while penalizing inventory accumulation.

-   __[Strategies - Mean Reversion](Strategies - Mean Reversion.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    Mean reversion is the statistical tendency for an asset's price to return to its historical average after deviating from it. While Momentum strategies bet on *continuation*, Mean Reversion strategies bet on *reversal* — buying when something is "too cheap" and selling when it is "too expensive" relative to recent history.

-   __[Strategies - Momentum Trading](Strategies - Momentum Trading.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    Momentum trading is a strategy that capitalizes on the continuance of existing trends in the market. The core philosophy is "buy high, sell higher." If an asset's price is rising strongly, momentum traders assume it will continue to rise.

-   __[Strategies - Pairs Trading](Strategies - Pairs Trading.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    This module demonstrates a statistical arbitrage strategy known as Pairs Trading. It identifies two assets that move together and trades the convergence of their spread. When the correlation weakens temporarily, executing trades on both assets allows for capturing profits as they revert to their historical relationship. This quantitative technique relies strictly on mathematical relationships rather than fundamental valuation.

-   __[Strategies - Statistical Arbitrage](Strategies - Statistical Arbitrage.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    This module demonstrates a basic Statistical Arbitrage strategy, specifically pairs trading.

-   __[Strategies - Trend Following](Strategies - Trend Following.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    Trend-following: ride momentum with discipline. Backbone of CTAs and managed-futures funds (AHL, Winton, Man, MLP). Profits from extended directional moves; pays for it during chop.

</div>

## :material-robot-outline: AI & Machine Learning

*Data-driven models: random forests, deep learning, RL and NLP for markets.*

<div class="grid cards" markdown>

-   __[AI Development](AI Development.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python · JavaScript</span>

    Command-line chatbots for Google's Gemini API, implemented in both Python and Node.js. This module demonstrates how to integrate a hosted large language model into a simple interactive application.

-   __[Learning Platform](Learning Platform.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    An all-in-one learning hub that delivers progressive Python lessons through both a guided CLI and a hostable Flask web interface. Lessons combine narrative walkthroughs, executable code examples, mini quizzes, and follow-up practice ideas geared toward aspiring quantitative developers.

-   __[Machine Learning - Feature Engineering](Machine Learning - Feature Engineering.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    The dirty secret of quant machine learning: the model is rarely the bottleneck.

-   __[Machine Learning - K-Means Clustering](Machine Learning - K-Means Clustering.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    Given a few hundred stocks and their return characteristics, which ones behave

-   __[Machine Learning - Random Forest](Machine Learning - Random Forest.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    This module provides a basic implementation of a Random Forest Predictor for quantitative finance. It uses scikit-learn's `RandomForestRegressor` to predict time series data or returns based on a set of features.

-   __[Machine Learning Time Series](Machine Learning Time Series.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    Applying incredibly sophisticated statistical and advanced computational matrix calculating algorithms to historical sequential asset prices explicitly enables quantitative researchers to discover heavily latent non linear correlation patterns. Standard basic linear techniques lack the internal theoretical mapping memory required to fully process continuous progression data natively. Therefore, explicit sequential data pattern prediction necessitates deeply specialized memory architectures uniquely capable of successfully retaining vast contextual numerical memory safely across thousands of chronologically independent market observations simultaneously.

-   __[Reinforcement Learning Q Learning](Reinforcement Learning Q Learning.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    This module extensively covers the core mathematical algorithms necessary to construct entirely autonomous quantitative execution agents. Rather than relying on rigid statistical parameters or explicit condition based trading logic, reinforcement learning allows an agent to discover the most optimal sequences of action through continuous simulated trial and error. The intelligent agent dynamically interprets complex environmental states and receives explicit scalar rewards or punitive penalties based directly upon its transactional profitability and risk management threshold maintenance. Over thousands of episodes, the model organically maps the market mechanics to develop a mathematically optimal trading policy without human intervention.

-   __[Sentiment Analysis on News](Sentiment Analysis on News.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    **This utility does NOT use any external APIs.** All sentiment analysis is done locally using a simple rule-based approach for learning and experimentation.

</div>

## :material-pulse: Market Microstructure

*Order books, spreads and the low-latency mechanics of how trades happen.*

<div class="grid cards" markdown>

-   __[High Frequency Trading](High Frequency Trading.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    High Frequency Trading (HFT) encompasses algorithmic strategies that execute a large number of orders at extremely high speeds — typically microseconds to milliseconds. HFT firms compete primarily on latency: the fastest participant to react to new information captures the profit.

-   __[Market Microstructure](Market Microstructure.md)__

    <span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-lang">Python</span>

    Market microstructure studies how trading mechanisms — the rules, protocols, and participants in a market — affect price formation, liquidity, and transaction costs. Understanding microstructure is essential for designing realistic execution algorithms, building order books, estimating market impact, and analysing bid-ask spreads.

</div>

## :material-tools: Utilities & Tools

*The plumbing: data ingestion, logging, FX, calendars and helpers.*

<div class="grid cards" markdown>

-   __[Core Utilities](Core Utilities.md)__

    <span class="lq-badge lq-beginner">Beginner</span><span class="lq-badge lq-lang">Python</span>

    This folder contains core mathematical and date/time utilities that form the foundation for quantitative finance calculations.

-   __[Currency Converter](Currency Converter.md)__

    <span class="lq-badge lq-beginner">Beginner</span><span class="lq-badge lq-lang">Python</span>

    **This utility does NOT use any external APIs.** All exchange rates are entered manually for learning and experimentation.

-   __[Data Processing](Data Processing.md)__

    <span class="lq-badge lq-beginner">Beginner</span><span class="lq-badge lq-lang">Python</span>

    This folder contains utilities for data processing, validation, and manipulation in financial applications.

-   __[Economic Calendar](Economic Calendar.md)__

    <span class="lq-badge lq-beginner">Beginner</span><span class="lq-badge lq-lang">Python</span>

    **This utility does NOT use any external APIs.** All data is managed locally for learning and experimentation.

-   __[Historical Data](Historical Data.md)__

    <span class="lq-badge lq-beginner">Beginner</span><span class="lq-badge lq-lang">JavaScript</span>

    A Node.js script that fetches historical bars (OHLCV data) for stocks or crypto from the Alpaca Market Data API. It prompts interactively for the symbol type, symbol, timeframe, and date range, then prints the results as JSON.

-   __[Logging](Logging.md)__

    <span class="lq-badge lq-beginner">Beginner</span><span class="lq-badge lq-lang">Python · JavaScript</span>

    A pair of minimal, dependency-light logging utilities implemented in both Python and JavaScript. Each supports adding, reading, editing, and deleting log entries through an interactive command-line menu. All entries are persisted to a plain-text `log.txt` file in the working directory.

-   __[Market Data](Market Data.md)__

    <span class="lq-badge lq-beginner">Beginner</span><span class="lq-badge lq-lang">Python</span>

    This folder contains utilities for processing, analyzing, and fetching market data for financial applications.

-   __[News Fetching](News Fetching.md)__

    <span class="lq-badge lq-beginner">Beginner</span><span class="lq-badge lq-lang">JavaScript</span>

    This utility provides a Google News headline scraper using the `google-news-json` package. It no longer requires any API keys, making it ideal for beginners who want to experiment with news-driven trading ideas or sentiment analysis without signing up for external services.

-   __[System Utilities](System Utilities.md)__

    <span class="lq-badge lq-beginner">Beginner</span><span class="lq-badge lq-lang">Python</span>

    This folder contains utilities for system-level operations, file management, and configuration in financial applications.

-   __[Websocket Connection](Websocket Connection.md)__

    <span class="lq-badge lq-beginner">Beginner</span><span class="lq-badge lq-lang">Python</span>

    This project provides WebSocket clients for connecting to various financial data providers, including YFLive and Finnhub. These utilities are designed for real-time market data streaming and analysis.

</div>
