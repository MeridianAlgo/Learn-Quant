"""
Sorting Algorithms Implementation
A comprehensive collection of sorting algorithms with detailed explanations and examples.
"""

from typing import List, TypeVar, Callable
import time
import random

T = TypeVar("T")


def bubble_sort(arr: List[T]) -> List[T]:
    """
    Bubble Sort: Simple but inefficient O(n²) sorting algorithm.
    Repeatedly steps through the list, compares adjacent elements and swaps them if out of order.

    Time Complexity: O(n²) worst/average, O(n) best (already sorted)
    Space Complexity: O(1)

    Args:
        arr: List of comparable elements

    Returns:
        Sorted list
    """
    n = len(arr)
    result = arr.copy()

    for i in range(n):
        # Flag to detect if any swap happened in this pass
        swapped = False

        # Last i elements are already in place
        for j in range(0, n - i - 1):
            if result[j] > result[j + 1]:
                result[j], result[j + 1] = result[j + 1], result[j]
                swapped = True

        # If no swaps occurred, array is sorted
        if not swapped:
            break

    return result


def selection_sort(arr: List[T]) -> List[T]:
    """
    Selection Sort: O(n²) algorithm that finds minimum element and places it at beginning.

    Time Complexity: O(n²) all cases
    Space Complexity: O(1)

    Args:
        arr: List of comparable elements

    Returns:
        Sorted list
    """
    n = len(arr)
    result = arr.copy()

    for i in range(n):
        # Find index of minimum element in unsorted portion
        min_idx = i
        for j in range(i + 1, n):
            if result[j] < result[min_idx]:
                min_idx = j

        # Swap minimum element with first unsorted element
        result[i], result[min_idx] = result[min_idx], result[i]

    return result


def insertion_sort(arr: List[T]) -> List[T]:
    """
    Insertion Sort: Efficient for small or nearly-sorted datasets. O(n²) worst case.
    Builds final sorted array one item at a time by inserting each element into proper position.

    Time Complexity: O(n²) worst/average, O(n) best (already sorted)
    Space Complexity: O(1)

    Args:
        arr: List of comparable elements

    Returns:
        Sorted list
    """
    result = arr.copy()

    for i in range(1, len(result)):
        key = result[i]
        j = i - 1

        # Move elements greater than key one position ahead
        while j >= 0 and result[j] > key:
            result[j + 1] = result[j]
            j -= 1

        result[j + 1] = key

    return result


def merge_sort(arr: List[T]) -> List[T]:
    """
    Merge Sort: Divide-and-conquer algorithm with O(n log n) complexity.
    Recursively divides array into halves, sorts them, then merges.

    Time Complexity: O(n log n) all cases
    Space Complexity: O(n)

    Args:
        arr: List of comparable elements

    Returns:
        Sorted list
    """

    def merge(left: List[T], right: List[T]) -> List[T]:
        """Merge two sorted lists into one sorted list."""
        merged = []
        i = j = 0

        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1

        # Add remaining elements
        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged

    # Base case: single element or empty list is already sorted
    if len(arr) <= 1:
        return arr.copy()

    # Divide array into halves
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    # Conquer: merge sorted halves
    return merge(left, right)


def quick_sort(arr: List[T]) -> List[T]:
    """
    Quick Sort: Efficient divide-and-conquer algorithm with O(n log n) average complexity.
    Picks pivot element and partitions array around it.

    Time Complexity: O(n log n) average, O(n²) worst (rare with good pivot selection)
    Space Complexity: O(log n) recursive stack

    Args:
        arr: List of comparable elements

    Returns:
        Sorted list
    """

    def partition(low: int, high: int) -> int:
        """Partition array around pivot, return pivot index."""
        pivot = result[high]  # Choose last element as pivot
        i = low - 1  # Index of smaller element

        for j in range(low, high):
            if result[j] <= pivot:
                i += 1
                result[i], result[j] = result[j], result[i]

        # Place pivot in correct position
        result[i + 1], result[high] = result[high], result[i + 1]
        return i + 1

    def quick_sort_recursive(low: int, high: int):
        """Recursive quick sort implementation."""
        if low < high:
            pi = partition(low, high)
            quick_sort_recursive(low, pi - 1)
            quick_sort_recursive(pi + 1, high)

    result = arr.copy()
    quick_sort_recursive(0, len(result) - 1)
    return result


def heap_sort(arr: List[T]) -> List[T]:
    """
    Heap Sort: Uses binary heap data structure for O(n log n) sorting.
    Builds max heap then repeatedly extracts maximum element.

    Time Complexity: O(n log n) all cases
    Space Complexity: O(1)

    Args:
        arr: List of comparable elements

    Returns:
        Sorted list
    """

    def heapify(n: int, i: int):
        """Maintain heap property for subtree rooted at index i."""
        largest = i  # Initialize largest as root
        left = 2 * i + 1
        right = 2 * i + 2

        # If left child exists and is larger than root
        if left < n and result[left] > result[largest]:
            largest = left

        # If right child exists and is larger than largest so far
        if right < n and result[right] > result[largest]:
            largest = right

        # If largest is not root, swap and continue heapifying
        if largest != i:
            result[i], result[largest] = result[largest], result[i]
            heapify(n, largest)

    result = arr.copy()
    n = len(result)

    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(n, i)

    # Extract elements one by one
    for i in range(n - 1, 0, -1):
        # Move current root to end
        result[0], result[i] = result[i], result[0]
        # Call heapify on reduced heap
        heapify(i, 0)

    return result


def compare_sorting_algorithms(arr: List[T], algorithms: List[Callable] = None) -> dict:
    """
    Compare performance of different sorting algorithms on the same dataset.

    Args:
        arr: Input array to sort
        algorithms: List of sorting functions to compare

    Returns:
        Dictionary with algorithm names and their execution times
    """
    if algorithms is None:
        algorithms = [
            bubble_sort,
            selection_sort,
            insertion_sort,
            merge_sort,
            quick_sort,
            heap_sort,
        ]

    results = {}

    for algorithm in algorithms:
        start_time = time.time()
        sorted_arr = algorithm(arr)
        end_time = time.time()

        # Verify sorting is correct
        assert sorted_arr == sorted(
            arr
        ), f"{algorithm.__name__} failed to sort correctly"

        results[algorithm.__name__] = {
            "time": end_time - start_time,
            "sorted": sorted_arr,
        }

    return results


def demonstrate_sorting():
    """Demonstrate all sorting algorithms with examples."""
    print("=== Sorting Algorithms Demonstration ===\n")

    # Test arrays
    small_array = [64, 34, 25, 12, 22, 11, 90]
    random_array = [random.randint(1, 100) for _ in range(15)]
    nearly_sorted = list(range(1, 21))
    nearly_sorted[5], nearly_sorted[10] = 50, 75  # Swap a few elements

    test_arrays = [
        ("Small Array", small_array),
        ("Random Array", random_array),
        ("Nearly Sorted", nearly_sorted),
    ]

    algorithms = [
        bubble_sort,
        selection_sort,
        insertion_sort,
        merge_sort,
        quick_sort,
        heap_sort,
    ]

    for name, arr in test_arrays:
        print(f"\n--- {name}: {arr} ---")

        for algorithm in algorithms:
            sorted_arr = algorithm(arr)
            print(f"{algorithm.__name__:15}: {sorted_arr}")

    # Performance comparison
    print("\n=== Performance Comparison ===")
    large_array = [random.randint(1, 1000) for _ in range(1000)]

    results = compare_sorting_algorithms(large_array)

    print(f"\nSorting array of {len(large_array)} elements:")
    for name, data in results.items():
        print(f"{name:15}: {data['time']:.6f} seconds")


if __name__ == "__main__":
    demonstrate_sorting()
