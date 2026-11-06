# Sorting Algorithms

<p class="lq-meta">Intermediate <span>/</span> Algorithms <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Algorithms%20-%20Sorting">source</a></p>


## Overview
A comprehensive implementation of fundamental sorting algorithms with detailed explanations, complexity analysis, and performance comparisons.

## Algorithms Included

### Basic Sorting (O(n²))
- **Bubble Sort**: Simple comparison-based algorithm, good for educational purposes
- **Selection Sort**: Finds minimum element repeatedly, minimal swaps
- **Insertion Sort**: Efficient for small or nearly-sorted datasets

### Advanced Sorting (O(n log n))
- **Merge Sort**: Stable divide-and-conquer algorithm
- **Quick Sort**: Efficient in-place sorting with good average performance
- **Heap Sort**: Uses binary heap structure, guaranteed O(n log n) performance

## Complexity Analysis

| Algorithm | Best | Average | Worst | Space | Stable |
|-----------|------|---------|-------|-------|--------|
| Bubble Sort | O(n) | O(n²) | O(n²) | O(1) | Yes |
| Selection Sort | O(n²) | O(n²) | O(n²) | O(1) | No |
| Insertion Sort | O(n) | O(n²) | O(n²) | O(1) | Yes |
| Merge Sort | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes |
| Quick Sort | O(n log n) | O(n log n) | O(n²) | O(log n) | No |
| Heap Sort | O(n log n) | O(n log n) | O(n log n) | O(1) | No |

## Usage Examples

```python
from sorting_algorithms import bubble_sort, quick_sort, merge_sort

data = [64, 34, 25, 12, 22, 11, 90]

# Basic sorting
sorted_bubble = bubble_sort(data)
sorted_quick = quick_sort(data)
sorted_merge = merge_sort(data)

# Performance comparison
from sorting_algorithms import compare_sorting_algorithms
results = compare_sorting_algorithms(data)
```

## Testing
Run the demonstration script to see all algorithms in action:

```bash
python sorting_algorithms.py
```

## Learning Points
- **Divide and Conquer**: Merge Sort and Quick Sort
- **In-place Sorting**: Quick Sort, Heap Sort, Selection Sort
- **Stability**: Why it matters for equal elements
- **Trade-offs**: Time vs Space complexity
- **Best Use Cases**: When to choose which algorithm

---

## Running it

```bash
cd "Algorithms - Sorting"
python "sorting_algorithms.py"
```

## More in Algorithms

- [Algorithms - Backtracking](Algorithms - Backtracking.md): Backtracking is a general algorithmic technique for solving problems by building candidates incrementally and abandoning a candidate ("backtracking") as soon as it is determined to violate the problem constraints.
- [Algorithms - Dynamic Programming](Algorithms - Dynamic Programming.md): Dynamic Programming (DP) is an algorithmic technique for solving problems by breaking them into overlapping subproblems, solving each subproblem once, and storing the result to avoid redundant computation.
- [Algorithms - Graph](Algorithms - Graph.md): Graph algorithms operate on structures composed of vertices (nodes) and edges (connections).
- [Algorithms - Machine Learning](Algorithms - Machine Learning.md): This module implements fundamental machine learning algorithms from scratch using only NumPy — no scikit-learn or frameworks.
- [Algorithms - Searching](Algorithms - Searching.md): Searching algorithms find a target value within a data structure.
- [Algorithms - String](Algorithms - String.md): String algorithms handle efficient manipulation, searching, and analysis of text data.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
