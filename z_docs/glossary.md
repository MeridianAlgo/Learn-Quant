# Glossary

Short definitions for terms that come up across the lessons, with a pointer to
where each one is implemented.

## Markets

Bid and ask
:   The best price a buyer will pay and a seller will accept. The gap is the
    spread. See [Market Microstructure](Market Microstructure.md).

Effective spread
:   Twice the distance between a trade price and the midquote at the time,
    which is what the trade actually cost. See
    [Market Microstructure - Liquidity Measures](Market Microstructure - Liquidity Measures.md).

Kyle's lambda
:   How far price moves per unit of net order flow. Higher means a thinner
    market. See
    [Market Microstructure - Liquidity Measures](Market Microstructure - Liquidity Measures.md).

Implementation shortfall
:   The difference between the price when you decided to trade and the price
    you actually got, across the whole order. See
    [Strategies - Optimal Execution](Strategies - Optimal Execution.md).

## Options

Derivative
:   A contract whose value depends on some underlying asset, such as an
    option on a stock. See [Black-Scholes Option Pricing](Black-Scholes Option Pricing.md).

Greeks
:   The sensitivities of an option price: delta, gamma, theta, vega, rho. See
    [Finance - Greeks Calculator](Finance - Greeks Calculator.md).

Implied volatility
:   The volatility that makes a pricing model match the market price of an
    option. See [Finance - Implied Volatility Surface](Finance - Implied Volatility Surface.md).

## Risk

Value at Risk
:   A loss level not expected to be exceeded over a horizon at a given
    confidence. See [Value at Risk (VaR)](Value at Risk (VaR).md).

Expected shortfall
:   The average loss on the days that are worse than VaR. See
    [Finance - Expected Shortfall](Finance - Expected Shortfall.md).

Drawdown
:   The fall from the highest point of an equity curve to a later low. See
    [Risk Metrics - Drawdown Analysis](Risk Metrics - Drawdown Analysis.md).

Sharpe ratio
:   Excess return divided by volatility. See
    [Sharpe and Sortino Ratio](Sharpe and Sortino Ratio.md).

## Statistics

Stationarity
:   A series whose mean, variance and autocorrelation do not drift over time.
    Many methods assume it. See [Quantitative Methods - Time Series](Quantitative Methods - Time Series.md).

Cointegration
:   Two wandering series whose spread does not wander. The basis of pairs
    trading. See [Quantitative Methods - Cointegration](Quantitative Methods - Cointegration.md).

GARCH
:   A model where today's volatility depends on yesterday's shock and
    yesterday's volatility, which produces volatility clustering. See
    [Quantitative Methods - GARCH](Quantitative Methods - GARCH.md).

Bandwidth
:   The width of the bump placed on each point in a kernel density estimate.
    It matters far more than the bump's shape. See
    [Quantitative Methods - Kernel Density Estimation](Quantitative Methods - Kernel Density Estimation.md).

Monte Carlo
:   Estimating something by simulating many random outcomes and averaging. See
    [Monte Carlo Portfolio Simulator](Monte Carlo Portfolio Simulator.md).

## Portfolios

Efficient frontier
:   The best achievable return for each level of risk. See
    [Portfolio Optimizer](Portfolio Optimizer.md).

Risk parity
:   Weighting assets so each contributes the same amount of risk. See
    [Portfolio Management - Risk Parity](Portfolio Management - Risk Parity.md).

Hierarchical risk parity
:   Risk parity applied down a tree of correlated clusters, which avoids
    inverting the covariance matrix. See
    [Portfolio Management - Hierarchical Risk Parity](Portfolio Management - Hierarchical Risk Parity.md).

Beta
:   How much an asset moves with the market. See [CAPM](CAPM.md) and
    [Finance - Beta Calculator](Finance - Beta Calculator.md).

Something missing? [Open an issue](https://github.com/MeridianAlgo/Learn-Quant/issues).
