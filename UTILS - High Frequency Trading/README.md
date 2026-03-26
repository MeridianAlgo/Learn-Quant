# High Frequency Trading

## Overview

High Frequency Trading (HFT) encompasses algorithmic strategies that execute a large number of orders at extremely high speeds — typically microseconds to milliseconds. HFT firms compete primarily on latency: the fastest participant to react to new information captures the profit.

This module covers the engineering and quantitative foundations of HFT systems: latency measurement, optimisation techniques, market data processing, and execution algorithms.

## Key Concepts

### Latency
The time elapsed between an event (e.g., a market data update) and a corresponding action (e.g., an order submission).

| Latency source | Typical range | Optimisation lever |
|---------------|---------------|-------------------|
| Network (co-location) | 1–10 microseconds | Physical proximity to exchange |
| Kernel / OS | 10–100 microseconds | Kernel bypass (DPDK, RDMA) |
| Application logic | 1–50 microseconds | C++, lock-free data structures |
| Order processing | 5–50 microseconds | FPGA, low-latency order entry |

### Latency Measurement
Accurate measurement requires:
- **Hardware timestamps**: OS system calls (time.perf_counter) have jitter; hardware timestamping is nanosecond-accurate.
- **Percentile analysis**: Mean latency is misleading; P99 and P99.9 (tail latency) determine worst-case behaviour.
- **Warm-up period**: First execution is slow (cold cache, JIT). Discard initial measurements.

### HFT Strategy Categories

**Market Making**
- Post both a bid and an ask simultaneously.
- Profit from the bid-ask spread on each round-trip.
- Risk: adverse selection (trading against informed flow).

**Latency Arbitrage**
- Exploit temporary price discrepancies between venues before they are corrected.
- Requires being faster than competitors, not just faster than the market.

**Statistical Arbitrage**
- Short-term mean reversion or momentum exploited at high frequency.
- E.g., ETF/constituent arbitrage, correlated-asset pairs at tick frequency.

**Order Flow Toxicity**
- Detect informed versus uninformed order flow.
- Widen spreads or pull quotes when order flow appears toxic (using metrics like VPIN).

### Execution Algorithms

| Algorithm | Goal | Typical use |
|-----------|------|-------------|
| TWAP | Execute evenly over time | Minimise market impact on schedule |
| VWAP | Match volume-weighted average price | Benchmark execution quality |
| Implementation Shortfall | Minimise total trading cost | Optimal balance of urgency vs. impact |
| POV (Participation) | Track a % of market volume | Stealth large-order execution |

## Files
- `latency_optimizer.py`: Latency measurement utilities, statistical analysis (mean, median, P99), baseline comparison, and bottleneck identification.
- `__init__.py`: Module exports for LatencyOptimizer, MarketDataProcessor, ExecutionAlgorithms, and HFTStrategies.

## How to Run
```bash
python latency_optimizer.py
```

## Financial Applications

### 1. Market Making
- HFT market makers provide continuous two-sided quotes, improving market liquidity.
- Profitability depends on capturing the spread while managing adverse selection.

### 2. Arbitrage
- Exchange arbitrage: same asset priced differently on two venues.
- ETF arbitrage: ETF price deviating from NAV of underlying basket.
- Both opportunities exist for milliseconds — only accessible to low-latency participants.

### 3. Latency Measurement for Strategy Evaluation
- Even for non-HFT strategies, measuring execution latency identifies bottlenecks.
- A strategy that generates signals in 100ms but executes in 500ms misses short-lived opportunities.

### 4. Backtesting Realism
- Accurate backtests of intraday strategies must model order arrival latency and queue position.
- Ignoring latency leads to overestimated fill rates and unrealistically optimistic results.

## Best Practices

- **Profile before optimising**: Measure where time is actually spent before rewriting code. Premature optimisation is the root of many bugs.
- **Co-location matters**: For genuine HFT, physical proximity to the exchange matching engine is the single most impactful latency reduction.
- **Python for prototyping only**: Python's GIL and interpreter overhead make it unsuitable for production HFT. C++ with lock-free queues is standard.
- **Focus on tail latency**: Strategies that are fast on average but slow occasionally (high P99) will underperform in production — tail events are when execution matters most.
- **Understand queue position**: On limit order exchanges, earlier-submitted orders at the same price have priority. Latency advantage determines queue position.
