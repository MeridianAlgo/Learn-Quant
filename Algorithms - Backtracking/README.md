# Algorithms – Backtracking

## Overview

Backtracking is a general algorithmic technique for solving problems by building candidates incrementally and abandoning a candidate ("backtracking") as soon as it is determined to violate the problem constraints. It is a systematic form of exhaustive search that prunes the search space to avoid exploring clearly invalid paths.

Backtracking is relevant in quantitative finance wherever the solution space is combinatorial — generating all possible portfolios, enumerating multi-leg option strategies, or searching through trading rule combinations.

## Key Concepts

### Backtracking Pattern
The core idea is recursive exploration:
1. Make a choice (add an element to the current candidate).
2. Recurse to explore deeper.
3. Undo the choice (backtrack) and try the next alternative.

This produces a depth-first traversal of an implicit decision tree.

### Permutations vs Combinations

| Problem | Order matters? | Formula | Example |
|---------|---------------|---------|---------|
| Permutation | Yes | n! | Orderings of a trade sequence |
| Combination | No | C(n, k) | k-asset subsets from n candidates |
| Subset | No | 2^n | All possible portfolio compositions |

### Pruning
The efficiency advantage over naive enumeration: constraints are checked early to cut off entire subtrees of invalid solutions before they are fully expanded.

### Time Complexity
- Permutations: O(n! * n) — factorial growth, only practical for small n.
- Combinations C(n,k): O(C(n,k) * k) — much smaller when k << n.
- Constraint satisfaction: highly problem-dependent; pruning can make exponential problems tractable in practice.

## Files
- `backtracking_algorithms.py`: Implementations of permutation generation, combination enumeration, and constraint-satisfying search with finance-relevant examples.

## How to Run
```bash
python backtracking_algorithms.py
```

## Financial Applications

### 1. Portfolio Construction
- Enumerate all k-asset combinations from a universe of n candidates.
- Useful for small-universe exhaustive optimisation (e.g., picking 5 ETFs from 20).

### 2. Options Strategy Generation
- Generate all valid multi-leg option combinations (calls/puts, different strikes) for a given expiry.
- Backtracking prunes invalid structures (e.g., undefined max-loss profiles).

### 3. Scenario Tree Construction
- Build all paths through a binomial option pricing tree.
- Useful for path-dependent options where payoff depends on the sequence of prices.

### 4. Rule Mining
- Search through combinations of technical indicators or thresholds to find rules that satisfy backtesting constraints (minimum Sharpe, maximum drawdown, etc.).

## Best Practices

- **Know your n early**: For n > 15–20, full permutation enumeration becomes computationally infeasible. Use heuristic search (genetic algorithms, simulated annealing) instead.
- **Prune aggressively**: Apply constraints as early as possible in the recursion — this is where most of the performance improvement over brute force comes from.
- **Memoise when possible**: If subproblems repeat across branches, cache results (transitioning from backtracking to dynamic programming).
- **Prefer combinations over permutations**: For portfolio selection, the order of assets does not matter — use combinations to reduce the search space by n!/(n-k)!.
