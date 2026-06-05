<p class="lq-badges"><span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-cat">Algorithms</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Algorithms - Backtracking"
    python "backtracking_algorithms.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Algorithms%20-%20Backtracking)

---
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


---

## Continue in Algorithms

<div class="grid cards" markdown>

-   :material-sitemap-outline: __[Algorithms - Dynamic Programming](Algorithms - Dynamic Programming.md)__

    Dynamic Programming (DP) is an algorithmic technique for solving problems by breaking them into overlapping subproblems, solving each subproblem once, and storing the result to avoid redundant computation. It converts exponential-time recursive solutions into polynomial-time ones.

-   :material-sitemap-outline: __[Algorithms - Graph](Algorithms - Graph.md)__

    Graph algorithms operate on structures composed of vertices (nodes) and edges (connections). Many financial problems are naturally modelled as graphs: currency markets form weighted directed graphs, asset correlation matrices define undirected weighted graphs, and order routing networks are flow graphs.

-   :material-sitemap-outline: __[Algorithms - Machine Learning](Algorithms - Machine Learning.md)__

    This module implements fundamental machine learning algorithms from scratch using only NumPy — no scikit-learn or frameworks. Building these algorithms by hand is the most effective way to understand what happens inside the black boxes used in production trading systems.

-   :material-sitemap-outline: __[Algorithms - Searching](Algorithms - Searching.md)__

    Searching algorithms find a target value within a data structure. The choice of algorithm determines whether a search takes O(n) time (checking every element) or O(log n) time (dividing the search space in half each step). In latency-sensitive financial systems, this difference is meaningful at scale.

-   :material-sitemap-outline: __[Algorithms - Sorting](Algorithms - Sorting.md)__

    A comprehensive implementation of fundamental sorting algorithms with detailed explanations, complexity analysis, and performance comparisons.

-   :material-sitemap-outline: __[Algorithms - String](Algorithms - String.md)__

    String algorithms handle efficient manipulation, searching, and analysis of text data. In quantitative finance, string processing is essential for parsing market data feeds, extracting information from news and filings, matching ticker symbols, and cleaning raw data from APIs.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
