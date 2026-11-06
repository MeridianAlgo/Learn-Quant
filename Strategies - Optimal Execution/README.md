# Strategies, Optimal Execution

A signal tells you what to own. It does not tell you how to get there. Moving
a large position takes time, and the schedule you choose decides how much of
your edge goes to the market on the way in. This module solves the classic
version of that problem, from Almgren and Chriss's 2000 paper "Optimal
Execution of Portfolio Transactions".

## The trade off

Sell everything immediately and you pay a large impact cost, since your own
order pushes the price down. Sell slowly and impact is small, but the price
has more time to drift against you. One cost is predictable and the other is
risk. The model puts them on a single scale:

```
minimise   E[cost] + lambda * Var[cost]
```

`lambda` is risk aversion. Zero means you only care about the average cost,
and large values mean you will pay extra impact to get done quickly.

## The pieces

* `ExecutionProblem` holds the inputs and checks they make sense.
* `urgency(problem, lam)` gives the decay rate `kappa`. Its inverse is the
  half life of the trade.
* `optimal_trajectory(problem, lam)` gives shares still held at each step.
* `twap_trajectory(problem)` is the straight line, `lam = 0` case.
* `trade_list(trajectory)` converts holdings into shares sold per interval.
* `expected_cost` and `cost_variance` score any schedule, optimal or not.
* `efficient_frontier(problem, lams)` traces cost against risk.

## The model

Over `N` intervals of length `tau`, selling `n_k` shares in interval `k`:

* **Temporary impact** costs `eta * n_k / tau` per share. It depends on how
  fast you trade and vanishes afterwards.
* **Permanent impact** moves the price by `gamma * n_k` for good.
* **Volatility** `sigma` moves the price randomly in between.

The expected cost then becomes a sum of squared trade sizes, and the variance
becomes a sum of squared holdings. Minimising one plus `lambda` times the
other gives

```
x_j = X * sinh(kappa * (T - t_j)) / sinh(kappa * T)
```

When `kappa` is near zero the hyperbolic sine is nearly linear and you get
TWAP. As `kappa` grows the curve falls steeply at the start, so most of the
order is done in the first few intervals.

## Reading the demo

The demo uses the paper's own example: one million shares of a $50 stock with
30% annual volatility, sold over five days. Running it prints the efficient
frontier. The first step away from TWAP is the cheap one. Going from
`lambda = 0` to `1e-7` takes about $73,000 off the standard deviation of cost
and adds under $8,000 to the expected cost. Further along the curve each unit
of risk removed gets more expensive, and at `1e-5` you pay over three times
the TWAP cost to cut the risk by three quarters. Where to sit on that curve is
a business decision, but the flat start is why almost nobody executes a large
order with pure TWAP.

## Example

```python
from optimal_execution import ExecutionProblem, optimal_trajectory, trade_list

p = ExecutionProblem(shares=100_000, horizon=1, periods=8, sigma=0.5, eta=1e-5)
holdings = optimal_trajectory(p, risk_aversion=1e-5)
print(trade_list(holdings).round())
```

## Things to watch

* Linear impact is a simplification. Empirical impact grows closer to the
  square root of order size, which the
  [`Market Microstructure`](../Market%20Microstructure/) module covers.
* The schedule is static. It is decided at the start and never reacts to
  what the price does. Adaptive versions exist but have no closed form.
* `eta` and `gamma` have to be estimated, usually from your own fills. The
  liquidity estimators in
  [`Market Microstructure - Liquidity Measures`](../Market%20Microstructure%20-%20Liquidity%20Measures/)
  are a rough starting point.
* If `eta - gamma * tau / 2` is not positive the problem is ill posed and the
  constructor refuses it.

## Where to go next

* For simulating TWAP, VWAP and POV fills see
  [`Order Execution Simulator`](../Order%20Execution%20Simulator/).
* For measuring what an execution actually cost see
  [`Finance - Transaction Cost Analysis`](../Finance%20-%20Transaction%20Cost%20Analysis/).
* For the dynamic programming view of the same idea see
  [`Algorithms - Dynamic Programming`](../Algorithms%20-%20Dynamic%20Programming/).
