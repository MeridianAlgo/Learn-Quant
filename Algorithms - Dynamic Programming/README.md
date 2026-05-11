# Algorithms – Dynamic Programming

## Overview

Dynamic Programming (DP) is an algorithmic technique for solving problems by breaking them into overlapping subproblems, solving each subproblem once, and storing the result to avoid redundant computation. It converts exponential-time recursive solutions into polynomial-time ones.

In quantitative finance, DP is the foundation of many core models: binomial option pricing, optimal stopping (American options), portfolio rebalancing with transaction costs, and reinforcement learning-based trading agents.

## Key Concepts

### Two Approaches

**Top-Down (Memoisation)**
- Write the natural recursive solution.
- Cache (memoise) the result of each unique subproblem in a hash map.
- On a repeated call, return the cached result immediately.
- Lazy: only computes subproblems actually reached by the recursion.

**Bottom-Up (Tabulation)**
- Identify the smallest subproblems and fill a table iteratively.
- Build up to the full solution using previously stored answers.
- Usually more memory-efficient and avoids recursion overhead.

### Optimal Substructure
A problem has optimal substructure if an optimal solution to the problem contains optimal solutions to its subproblems. This is a necessary condition for DP.

### Overlapping Subproblems
Unlike divide-and-conquer (which solves non-overlapping subproblems), DP is valuable when the same subproblems recur many times in the naive recursive solution.

### Complexity Comparison

| Approach | Time | Space |
|----------|------|-------|
| Naive recursion (Fibonacci) | O(2^n) | O(n) stack |
| Memoised recursion | O(n) | O(n) cache |
| Bottom-up tabulation | O(n) | O(n) or O(1) |

## Files
- `dynamic_programming.py`: Classic DP problems (Fibonacci, longest common subsequence, knapsack) with memoisation and tabulation implementations, plus finance-relevant applications.

## How to Run
```bash
python dynamic_programming.py
```

## Financial Applications

### 1. Binomial Option Pricing
- The binomial tree is a DP table: each node's value is the discounted expected value of its two children.
- American options require an additional comparison with immediate exercise value at each node.

### 2. Optimal Stopping (American Options)
- DP solves: "at each time step, is it better to exercise now or wait?"
- The Longstaff-Schwartz algorithm uses regression + DP for Monte Carlo American option pricing.

### 3. Portfolio Rebalancing with Transaction Costs
- DP determines the optimal rebalancing policy over time: when the cost of rebalancing is outweighed by the expected improvement in portfolio allocation.

### 4. Optimal Trade Scheduling (VWAP/TWAP)
- Almgren-Chriss model: DP finds the optimal execution schedule that minimises market impact plus timing risk over a fixed horizon.

### 5. Regime-Switching Models
- Hidden Markov Model decoding (Viterbi algorithm) is a DP algorithm used to identify market regimes (bull/bear/volatile).

## Best Practices

- **Draw the recursion tree first**: Identify which subproblems overlap before coding — this confirms DP is applicable.
- **Define the state clearly**: A DP state must capture everything needed to solve all remaining subproblems from that point.
- **Optimise space**: Many DP tables only depend on the previous row — use a rolling array to reduce O(n*m) space to O(m).
- **Watch for floating-point accumulation**: In financial DP (option trees, execution models), rounding errors accumulate across many table entries — use sufficient precision.
