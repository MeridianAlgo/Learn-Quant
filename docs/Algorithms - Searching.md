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
