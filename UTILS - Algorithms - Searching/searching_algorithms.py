"""
Searching Algorithms Implementation
A comprehensive collection of searching algorithms with detailed explanations and examples.
"""

import math
import random
from typing import Callable, List, Optional, TypeVar

T = TypeVar("T")


def linear_search(arr: List[T], target: T) -> Optional[int]:
    """
    Linear Search: Simple sequential search through the array.
    Checks each element one by one until target is found or array ends.

    Time Complexity: O(n) all cases
    Space Complexity: O(1)

    Args:
        arr: List of elements to search through
        target: Element to find

    Returns:
        Index of target if found, None otherwise
    """
    for i, element in enumerate(arr):
        if element == target:
            return i
    return None


def binary_search(arr: List[T], target: T) -> Optional[int]:
    """
    Binary Search: Efficient search for sorted arrays using divide-and-conquer.
    Repeatedly divides search interval in half.

    Time Complexity: O(log n) all cases
    Space Complexity: O(1) iterative, O(log n) recursive

    Args:
        arr: Sorted list of elements to search through
        target: Element to find

    Returns:
        Index of target if found, None otherwise
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = left + (right - left) // 2  # Avoid overflow

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return None


def binary_search_recursive(
    arr: List[T], target: T, left: int = 0, right: int = None
) -> Optional[int]:
    """
    Recursive version of binary search.

    Args:
        arr: Sorted list of elements
        target: Element to find
        left: Left boundary of search interval
        right: Right boundary of search interval

    Returns:
        Index of target if found, None otherwise
    """
    if right is None:
        right = len(arr) - 1

    if left > right:
        return None

    mid = left + (right - left) // 2

    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)


def jump_search(arr: List[T], target: T) -> Optional[int]:
    """
    Jump Search: Optimized for sorted arrays by jumping ahead by fixed steps.
    Good for arrays where random access is expensive.

    Time Complexity: O(√n) all cases
    Space Complexity: O(1)

    Args:
        arr: Sorted list of elements to search through
        target: Element to find

    Returns:
        Index of target if found, None otherwise
    """
    n = len(arr)
    step = int(math.sqrt(n))
    prev = 0

    # Find the block where element could be present
    while prev < n and arr[min(step, n) - 1] < target:
        prev = step
        step += int(math.sqrt(n))
        if prev >= n:
            return None

    # Linear search in the identified block
    for i in range(prev, min(step, n)):
        if arr[i] == target:
            return i

    return None


def interpolation_search(arr: List[T], target: T) -> Optional[int]:
    """
    Interpolation Search: Improved binary search for uniformly distributed data.
    Estimates position based on target value relative to endpoints.

    Time Complexity: O(log log n) average for uniform distribution, O(n) worst
    Space Complexity: O(1)

    Args:
        arr: Sorted list of elements with uniform distribution
        target: Element to find

    Returns:
        Index of target if found, None otherwise
    """
    left, right = 0, len(arr) - 1

    while left <= right and target >= arr[left] and target <= arr[right]:
        if left == right:
            return left if arr[left] == target else None

        # Estimate position using interpolation formula
        pos = left + ((target - arr[left]) * (right - left)) // (arr[right] - arr[left])

        if arr[pos] == target:
            return pos
        elif arr[pos] < target:
            left = pos + 1
        else:
            right = pos - 1

    return None


def exponential_search(arr: List[T], target: T) -> Optional[int]:
    """
    Exponential Search: Efficient for unbounded or infinite sorted arrays.
    First finds range where target might be, then uses binary search.

    Time Complexity: O(log n) all cases
    Space Complexity: O(1)

    Args:
        arr: Sorted list of elements to search through
        target: Element to find

    Returns:
        Index of target if found, None otherwise
    """
    n = len(arr)

    # If target is at first position
    if arr[0] == target:
        return 0

    # Find range for binary search by repeated doubling
    i = 1
    while i < n and arr[i] <= target:
        i *= 2

    # Call binary search for the found range
    return binary_search_recursive(arr, target, i // 2, min(i, n - 1))


def fibonacci_search(arr: List[T], target: T) -> Optional[int]:
    """
    Fibonacci Search: Similar to binary search but uses Fibonacci numbers.
    Divides array into unequal parts based on Fibonacci sequence.

    Time Complexity: O(log n) all cases
    Space Complexity: O(1)

    Args:
        arr: Sorted list of elements to search through
        target: Element to find

    Returns:
        Index of target if found, None otherwise
    """

    def fibonacci_numbers(n: int) -> tuple:
        """Generate Fibonacci numbers up to n."""
        fib2 = 0  # (m-2)th Fibonacci number
        fib1 = 1  # (m-1)th Fibonacci number
        fib = fib2 + fib1  # mth Fibonacci number

        while fib < n:
            fib2 = fib1
            fib1 = fib
            fib = fib2 + fib1

        return fib, fib1, fib2

    n = len(arr)
    fib, fib1, fib2 = fibonacci_numbers(n)

    # Offset for eliminated range
    offset = -1

    while fib > 1:
        # Check if fib2 is a valid location
        i = min(offset + fib2, n - 1)

        if arr[i] < target:
            fib = fib1
            fib1 = fib2
            fib2 = fib - fib1
            offset = i
        elif arr[i] > target:
            fib = fib2
            fib1 = fib1 - fib2
            fib2 = fib - fib1
        else:
            return i

    # Check for remaining element
    if fib1 and offset + 1 < n and arr[offset + 1] == target:
        return offset + 1

    return None


def compare_searching_algorithms(
    arr: List[T], target: T, algorithms: List[Callable] = None
) -> dict:
    """
    Compare performance of different searching algorithms.

    Args:
        arr: Input array to search in
        target: Element to search for
        algorithms: List of search functions to compare

    Returns:
        Dictionary with algorithm names and their results
    """
    if algorithms is None:
        algorithms = [linear_search, binary_search, jump_search, interpolation_search]

    results = {}

    for algorithm in algorithms:
        # For algorithms requiring sorted arrays
        if algorithm in [
            binary_search,
            jump_search,
            interpolation_search,
            fibonacci_search,
            exponential_search,
        ]:
            search_arr = sorted(arr)
        else:
            search_arr = arr

        result = algorithm(search_arr, target)

        results[algorithm.__name__] = {
            "found": result is not None,
            "index": result,
            "array_size": len(search_arr),
        }

    return results


def demonstrate_searching():
    """Demonstrate all searching algorithms with examples."""
    print("=== Searching Algorithms Demonstration ===\n")

    # Test arrays
    sorted_array = list(range(1, 101))  # 1 to 100
    random_array = [random.randint(1, 100) for _ in range(50)]

    # Test targets
    test_cases = [
        ("Existing Element", sorted_array, 42),
        ("First Element", sorted_array, 1),
        ("Last Element", sorted_array, 100),
        ("Non-existent Element", sorted_array, 150),
        ("Random Array Search", random_array, random_array[25] if random_array else 0),
    ]

    algorithms = [
        linear_search,
        binary_search,
        binary_search_recursive,
        jump_search,
        interpolation_search,
        exponential_search,
        fibonacci_search,
    ]

    for case_name, arr, target in test_cases:
        print(f"\n--- {case_name}: Searching for {target} ---")

        for algorithm in algorithms:
            # Use sorted array for algorithms that require it
            if algorithm in [
                binary_search,
                binary_search_recursive,
                jump_search,
                interpolation_search,
                exponential_search,
                fibonacci_search,
            ]:
                search_arr = sorted(arr)
            else:
                search_arr = arr

            result = algorithm(search_arr, target)

            if result is not None:
                print(f"{algorithm.__name__:20}: Found at index {result}")
            else:
                print(f"{algorithm.__name__:20}: Not found")

    # Performance comparison
    print("\n=== Performance Comparison ===")
    large_array = list(range(1, 10001))  # 1 to 10000
    target = 7856

    results = compare_searching_algorithms(large_array, target)

    print(f"\nSearching for {target} in array of {len(large_array)} elements:")
    for name, data in results.items():
        status = "Found" if data["found"] else "Not found"
        index = f"at index {data['index']}" if data["found"] else ""
        print(f"{name:20}: {status} {index}")


if __name__ == "__main__":
    demonstrate_searching()
