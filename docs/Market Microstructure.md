# Market Microstructure

## Overview

Market microstructure studies how trading mechanisms — the rules, protocols, and participants in a market — affect price formation, liquidity, and transaction costs. Understanding microstructure is essential for designing realistic execution algorithms, building order books, estimating market impact, and analysing bid-ask spreads.

This module implements three core components: an order book, market impact models, and spread analysis tools.

## Key Concepts

### The Order Book
A real-time record of all outstanding limit orders at each price level, separated into bids (buyers) and asks (sellers).

- **Bid side**: price levels where buyers are willing to buy, sorted descending (best bid = highest price).
- **Ask side**: price levels where sellers are willing to sell, sorted ascending (best ask = lowest price).
- **Bid-ask spread**: best_ask - best_bid. The minimum cost of an immediate round-trip trade.
- **Market depth**: the total volume available at each price level; deeper books absorb large orders with less price impact.

### Order Types

| Type | Description | Guarantee |
|------|-------------|-----------|
| Market order | Execute immediately at best available price | Execution, not price |
| Limit order | Execute only at specified price or better | Price, not execution |
| Stop order | Becomes market order when price reaches trigger | Neither |

### Price Formation
Prices in a continuous auction market are formed by the interaction of market orders (demand) and limit orders (supply). Information arrives through order flow — the pattern of buys and sells over time.

### Bid-Ask Spread Components
The spread has three components:
1. **Order processing cost**: administrative cost of running the market.
2. **Inventory cost**: compensation for the dealer holding unwanted inventory risk.
3. **Adverse selection cost**: compensation for trading with potentially better-informed counterparties.

### Market Impact
When a large order is executed, it moves the price against the trader. Market impact has two components:
- **Temporary impact**: immediate price movement from consuming liquidity; reverses after execution.
- **Permanent impact**: lasting price change reflecting information content of the trade.

### Market Impact Models

| Model | Formula | Characteristics |
|-------|---------|----------------|
| Square-root model | impact ∝ sigma * sqrt(Q/ADV) | Standard industry model |
| Linear model | impact ∝ Q / ADV | Simpler, overestimates for large Q |
| Almgren-Chriss | optimal VWAP schedule | Minimises impact + timing risk |

where Q = trade size, ADV = average daily volume, sigma = volatility.

## Files
- `order_book.py`: Order dataclass with heap ordering, order book state with bid/ask sides, order matching engine.
- `market_impact.py`: Trade dataclass, abstract MarketImpactModel base class, square-root and linear impact implementations.
- `spread_analyzer.py`: Quote dataclass with bid/ask properties, spread time series analysis, spread statistics.
- `__init__.py`: Module-level exports.

## How to Run
```bash
# Run individual components
python order_book.py
python market_impact.py
python spread_analyzer.py
```

## Financial Applications

### 1. Smart Order Routing (SOR)
- Compare available liquidity and spread across multiple venues.
- Route each slice of a large order to the venue offering the best execution quality.

### 2. Execution Cost Analysis (TCA)
- Transaction Cost Analysis measures the difference between the decision price and the actual execution price.
- Market impact models quantify the avoidable and unavoidable components of execution cost.

### 3. Optimal Execution
- Almgren-Chriss model: find the trading schedule that minimises the total cost of executing a large order over a fixed horizon.
- Trade-off: execute slowly (less market impact) vs. execute quickly (less price risk from waiting).

### 4. Liquidity Risk Management
- Order book depth determines how much can be traded before the market moves significantly.
- Portfolios with large positions in illiquid securities face significant liquidation costs in stress scenarios.

### 5. Market Quality Research
- Monitor spread and depth over time to assess the impact of exchange rule changes or market events.
- Intraday spread patterns reveal optimal times of day for execution (spreads narrow mid-session).

## Best Practices

- **Never ignore transaction costs**: A strategy that looks profitable before costs may lose money after them. Always model realistic spreads and impact.
- **Estimate impact before trading**: Use the square-root model to estimate cost before submitting large orders. If estimated impact exceeds expected alpha, do not trade.
- **Use limit orders when possible**: Market orders consume liquidity and pay the spread; limit orders provide liquidity and can capture the spread.
- **Time large orders**: Spreads are widest at open/close; mid-session liquidity is deepest for most liquid securities.
- **Measure with real data**: Impact models are calibrated on historical data; performance varies significantly by asset class, market cap, and volatility regime.
