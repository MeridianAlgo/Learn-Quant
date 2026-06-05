<p class="lq-badges"><span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-cat">Algorithms</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Algorithms - Sorting"
    python "sorting_algorithms.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Algorithms%20-%20Sorting)

---
# Sorting Algorithms

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

-   :material-sitemap-outline: __[Algorithms - Searching](Algorithms - Searching.md)__

    Searching algorithms find a target value within a data structure. The choice of algorithm determines whether a search takes O(n) time (checking every element) or O(log n) time (dividing the search space in half each step). In latency-sensitive financial systems, this difference is meaningful at scale.

-   :material-sitemap-outline: __[Algorithms - String](Algorithms - String.md)__

    String algorithms handle efficient manipulation, searching, and analysis of text data. In quantitative finance, string processing is essential for parsing market data feeds, extracting information from news and filings, matching ticker symbols, and cleaning raw data from APIs.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
