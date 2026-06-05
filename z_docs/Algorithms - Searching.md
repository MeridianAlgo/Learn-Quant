<p class="lq-badges"><span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-cat">Algorithms</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Algorithms - Searching"
    python "searching_algorithms.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Algorithms%20-%20Searching)

---
# Algorithms – Searching

## Overview

Searching algorithms find a target value within a data structure. The choice of algorithm determines whether a search takes O(n) time (checking every element) or O(log n) time (dividing the search space in half each step). In latency-sensitive financial systems, this difference is meaningful at scale.

## Key Concepts

### Linear Search
Scan every element sequentially until the target is found.

- **Time complexity**: O(n) worst and average case.
- **Space complexity**: O(1).
- Works on any data structure, sorted or unsorted.
- The only option when data is unsorted or when accessing elements is expensive.

### Binary Search
Exploit a sorted array to halve the search space at each step.

```
while low <= high:
    mid = (low + high) // 2
    if arr[mid] == target: return mid
    elif arr[mid] < target: low = mid + 1
    else: high = mid - 1
```

- **Time complexity**: O(log n) — searching 1 billion sorted elements takes at most 30 comparisons.
- **Space complexity**: O(1) iterative, O(log n) recursive.
- Prerequisite: data must be sorted.

### Binary Search Variants
- **Lower bound**: find the leftmost position where the target could be inserted.
- **Upper bound**: find the rightmost position.
- These are used to find ranges (e.g., all orders within a price band).

### Interpolation Search
An improvement over binary search for uniformly distributed data: estimate the position of the target based on its value rather than always checking the midpoint. Achieves O(log log n) on uniform data.

## Files
- `searching_algorithms.py`: Linear search, binary search (standard and variants), interpolation search, and exponential search with finance-relevant examples.

## How to Run
```bash
python searching_algorithms.py
```

## Financial Applications

### 1. Order Book Price Level Lookup
- Order books maintain sorted price levels (bids descending, asks ascending).
- Binary search finds the best matching price level in O(log n).
- Lower/upper bound variants find all orders within a price range efficiently.

### 2. Time-Series Data Access
- Historical price arrays are sorted by timestamp.
- Binary search retrieves the closing price for any specific date in O(log n) instead of scanning all records.

### 3. Strike Price Lookup
- Options chains list strikes in sorted order.
- Binary search finds the nearest-ATM (at-the-money) strike efficiently.

### 4. Percentile and Quantile Calculation
- Value at Risk (VaR) requires finding the p-th percentile of a sorted P&L distribution.
- Binary search locates the quantile boundary in O(log n).

### 5. Threshold Detection in Signal Arrays
- Given a sorted array of indicator values, find the first date the indicator crossed a threshold.
- Binary search (lower bound variant) solves this in O(log n).

## Best Practices

- **Keep data sorted when searches are frequent**: The O(n log n) cost of sorting once is paid back after ~log n binary searches.
- **Use `bisect` in Python**: The standard library `bisect` module provides optimised binary search (implemented in C). Prefer it over manual implementations in production.
- **Interpolation search for dense price grids**: When tick data is approximately uniform, interpolation search outperforms binary search in practice.
- **Know your data distribution**: Searching unsorted data? Linear search. Sorted and uniform? Interpolation. Sorted and unknown distribution? Binary.


---

## Continue in Algorithms

<div class="grid cards" markdown>

-   :material-sitemap-outline: __[Algorithms - Backtracking](Algorithms - Backtracking.md)__

    Backtracking is a general algorithmic technique for solving problems by building candidates incrementally and abandoning a candidate ("backtracking") as soon as it is determined to violate the problem constraints. It is a systematic form of exhaustive search that prunes the search space to avoid exploring clearly invalid paths.

-   :material-sitemap-outline: __[Algorithms - Dynamic Programming](Algorithms - Dynamic Programming.md)__

    Dynamic Programming (DP) is an algorithmic technique for solving problems by breaking them into overlapping subproblems, solving each subproblem once, and storing the result to avoid redundant computation. It converts exponential-time recursive solutions into polynomial-time ones.

-   :material-sitemap-outline: __[Algorithms - Graph](Algorithms - Graph.md)__

    Graph algorithms operate on structures composed of vertices (nodes) and edges (connections). Many financial problems are naturally modelled as graphs: currency markets form weighted directed graphs, asset correlation matrices define undirected weighted graphs, and order routing networks are flow graphs.

-   :material-sitemap-outline: __[Algorithms - Machine Learning](Algorithms - Machine Learning.md)__

    This module implements fundamental machine learning algorithms from scratch using only NumPy — no scikit-learn or frameworks. Building these algorithms by hand is the most effective way to understand what happens inside the black boxes used in production trading systems.

-   :material-sitemap-outline: __[Algorithms - Sorting](Algorithms - Sorting.md)__

    A comprehensive implementation of fundamental sorting algorithms with detailed explanations, complexity analysis, and performance comparisons.

-   :material-sitemap-outline: __[Algorithms - String](Algorithms - String.md)__

    String algorithms handle efficient manipulation, searching, and analysis of text data. In quantitative finance, string processing is essential for parsing market data feeds, extracting information from news and filings, matching ticker symbols, and cleaning raw data from APIs.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
