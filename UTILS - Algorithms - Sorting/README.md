# Sorting Algorithms

## 📋 Overview
A comprehensive implementation of fundamental sorting algorithms with detailed explanations, complexity analysis, and performance comparisons.

## 🚀 Algorithms Included

### Basic Sorting (O(n²))
- **Bubble Sort**: Simple comparison-based algorithm, good for educational purposes
- **Selection Sort**: Finds minimum element repeatedly, minimal swaps
- **Insertion Sort**: Efficient for small or nearly-sorted datasets

### Advanced Sorting (O(n log n))
- **Merge Sort**: Stable divide-and-conquer algorithm
- **Quick Sort**: Efficient in-place sorting with good average performance
- **Heap Sort**: Uses binary heap structure, guaranteed O(n log n) performance

## 🧮 Complexity Analysis

| Algorithm | Best | Average | Worst | Space | Stable |
|-----------|------|---------|-------|-------|--------|
| Bubble Sort | O(n) | O(n²) | O(n²) | O(1) | Yes |
| Selection Sort | O(n²) | O(n²) | O(n²) | O(1) | No |
| Insertion Sort | O(n) | O(n²) | O(n²) | O(1) | Yes |
| Merge Sort | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes |
| Quick Sort | O(n log n) | O(n log n) | O(n²) | O(log n) | No |
| Heap Sort | O(n log n) | O(n log n) | O(n log n) | O(1) | No |

## 🎯 Usage Examples

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

## 🧪 Testing
Run the demonstration script to see all algorithms in action:

```bash
python sorting_algorithms.py
```

## 📚 Learning Points
- **Divide and Conquer**: Merge Sort and Quick Sort
- **In-place Sorting**: Quick Sort, Heap Sort, Selection Sort
- **Stability**: Why it matters for equal elements
- **Trade-offs**: Time vs Space complexity
- **Best Use Cases**: When to choose which algorithm
