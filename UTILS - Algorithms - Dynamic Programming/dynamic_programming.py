"""
Dynamic Programming Algorithms Implementation
A comprehensive collection of DP algorithms with detailed explanations and examples.
"""

import sys
from typing import List, Tuple


def fibonacci_memoization(n: int) -> int:
    """
    Fibonacci using memoization (top-down DP).
    Stores results of expensive function calls and returns cached result.

    Time Complexity: O(n)
    Space Complexity: O(n) for recursion stack + memoization

    Args:
        n: Fibonacci number to compute

    Returns:
        nth Fibonacci number
    """
    if n <= 1:
        return n

    memo = {}

    def fib_helper(k: int) -> int:
        if k in memo:
            return memo[k]
        if k <= 1:
            return k

        memo[k] = fib_helper(k - 1) + fib_helper(k - 2)
        return memo[k]

    return fib_helper(n)


def fibonacci_tabulation(n: int) -> int:
    """
    Fibonacci using tabulation (bottom-up DP).
    Builds table iteratively from base cases.

    Time Complexity: O(n)
    Space Complexity: O(n)

    Args:
        n: Fibonacci number to compute

    Returns:
        nth Fibonacci number
    """
    if n <= 1:
        return n

    dp = [0] * (n + 1)
    dp[0], dp[1] = 0, 1

    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]

    return dp[n]


def fibonacci_optimized(n: int) -> int:
    """
    Optimized Fibonacci using O(1) space.
    Only keeps track of last two values.

    Time Complexity: O(n)
    Space Complexity: O(1)

    Args:
        n: Fibonacci number to compute

    Returns:
        nth Fibonacci number
    """
    if n <= 1:
        return n

    prev, curr = 0, 1

    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr

    return curr


def coin_change_min_coins(coins: List[int], amount: int) -> int:
    """
    Coin Change Problem: Minimum number of coins to make amount.

    Time Complexity: O(n * m) where n = len(coins), m = amount
    Space Complexity: O(m)

    Args:
        coins: Available coin denominations
        amount: Target amount

    Returns:
        Minimum number of coins, or -1 if impossible
    """
    if amount == 0:
        return 0

    # dp[i] = minimum coins to make amount i
    dp = [sys.maxsize] * (amount + 1)
    dp[0] = 0

    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i and dp[i - coin] != sys.maxsize:
                dp[i] = min(dp[i], dp[i - coin] + 1)

    return dp[amount] if dp[amount] != sys.maxsize else -1


def coin_change_count_ways(coins: List[int], amount: int) -> int:
    """
    Coin Change Problem: Number of ways to make amount.

    Time Complexity: O(n * m)
    Space Complexity: O(m)

    Args:
        coins: Available coin denominations
        amount: Target amount

    Returns:
        Number of ways to make amount
    """
    if amount == 0:
        return 1

    dp = [0] * (amount + 1)
    dp[0] = 1

    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]

    return dp[amount]


def knapsack_01(weights: List[int], values: List[int], capacity: int) -> int:
    """
    0/1 Knapsack Problem: Maximum value with weight constraint.
    Each item can be taken at most once.

    Time Complexity: O(n * capacity)
    Space Complexity: O(n * capacity)

    Args:
        weights: Weight of each item
        values: Value of each item
        capacity: Maximum weight capacity

    Returns:
        Maximum achievable value
    """
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(
                    values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w]
                )
            else:
                dp[i][w] = dp[i - 1][w]

    return dp[n][capacity]


def knapsack_01_optimized(weights: List[int], values: List[int], capacity: int) -> int:
    """
    Optimized 0/1 Knapsack using O(capacity) space.

    Time Complexity: O(n * capacity)
    Space Complexity: O(capacity)
    """
    dp = [0] * (capacity + 1)

    for i in range(len(weights)):
        # Iterate backwards to avoid using same item multiple times
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], values[i] + dp[w - weights[i]])

    return dp[capacity]


def unbounded_knapsack(weights: List[int], values: List[int], capacity: int) -> int:
    """
    Unbounded Knapsack Problem: Items can be taken unlimited times.

    Time Complexity: O(n * capacity)
    Space Complexity: O(capacity)

    Args:
        weights: Weight of each item
        values: Value of each item
        capacity: Maximum weight capacity

    Returns:
        Maximum achievable value
    """
    dp = [0] * (capacity + 1)

    for w in range(1, capacity + 1):
        for i in range(len(weights)):
            if weights[i] <= w:
                dp[w] = max(dp[w], values[i] + dp[w - weights[i]])

    return dp[capacity]


def longest_common_subsequence(text1: str, text2: str) -> int:
    """
    Longest Common Subsequence (LCS).
    Find longest sequence that appears in both strings in order.

    Time Complexity: O(m * n)
    Space Complexity: O(m * n)

    Args:
        text1: First string
        text2: Second string

    Returns:
        Length of longest common subsequence
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def longest_common_subsequence_with_path(text1: str, text2: str) -> Tuple[int, str]:
    """
    LCS with actual subsequence reconstruction.

    Returns:
        Tuple of (length, subsequence)
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Reconstruct LCS
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if text1[i - 1] == text2[j - 1]:
            lcs.append(text1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return dp[m][n], "".join(reversed(lcs))


def longest_increasing_subsequence(nums: List[int]) -> int:
    """
    Longest Increasing Subsequence (LIS).
    Find longest subsequence with strictly increasing elements.

    Time Complexity: O(n²)
    Space Complexity: O(n)

    Args:
        nums: Input array

    Returns:
        Length of longest increasing subsequence
    """
    if not nums:
        return 0

    dp = [1] * len(nums)

    for i in range(1, len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)


def longest_increasing_subsequence_optimized(nums: List[int]) -> int:
    """
    Optimized LIS using patience sorting approach.

    Time Complexity: O(n log n)
    Space Complexity: O(n)
    """
    if not nums:
        return 0

    tails = []

    for num in nums:
        left, right = 0, len(tails)
        while left < right:
            mid = (left + right) // 2
            if tails[mid] < num:
                left = mid + 1
            else:
                right = mid

        if left == len(tails):
            tails.append(num)
        else:
            tails[left] = num

    return len(tails)


def edit_distance(word1: str, word2: str) -> int:
    """
    Edit Distance (Levenshtein Distance).
    Minimum operations to convert one string to another.
    Operations: insert, delete, replace.

    Time Complexity: O(m * n)
    Space Complexity: O(m * n)

    Args:
        word1: First string
        word2: Second string

    Returns:
        Minimum edit distance
    """
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,  # delete
                    dp[i][j - 1] + 1,  # insert
                    dp[i - 1][j - 1] + 1,  # replace
                )

    return dp[m][n]


def matrix_chain_multiplication(dimensions: List[int]) -> int:
    """
    Matrix Chain Multiplication: Minimum scalar multiplications needed.

    Time Complexity: O(n³)
    Space Complexity: O(n²)

    Args:
        dimensions: Dimensions of matrices [p0, p1, p2, ..., pn]
                   where matrix i has dimensions pi-1 x pi

    Returns:
        Minimum number of scalar multiplications
    """
    n = len(dimensions) - 1  # Number of matrices
    dp = [[0] * (n + 1) for _ in range(n + 1)]

    # chain_len is chain length
    for chain_len in range(2, n + 1):
        for i in range(1, n - chain_len + 2):
            j = i + chain_len - 1
            dp[i][j] = sys.maxsize

            for k in range(i, j):
                cost = (
                    dp[i][k]
                    + dp[k + 1][j]
                    + dimensions[i - 1] * dimensions[k] * dimensions[j]
                )
                dp[i][j] = min(dp[i][j], cost)

    return dp[1][n]


def palindrome_partitioning(s: str) -> int:
    """
    Palindrome Partitioning: Minimum cuts needed for palindrome substrings.

    Time Complexity: O(n²)
    Space Complexity: O(n²)

    Args:
        s: Input string

    Returns:
        Minimum number of cuts needed
    """
    n = len(s)

    # Precompute palindrome table
    is_palindrome = [[False] * n for _ in range(n)]
    for i in range(n):
        is_palindrome[i][i] = True

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                if length == 2 or is_palindrome[i + 1][j - 1]:
                    is_palindrome[i][j] = True

    # DP for minimum cuts
    dp = [0] * n
    for i in range(n):
        if is_palindrome[0][i]:
            dp[i] = 0
        else:
            dp[i] = sys.maxsize
            for j in range(i):
                if is_palindrome[j + 1][i]:
                    dp[i] = min(dp[i], dp[j] + 1)

    return dp[n - 1]


def demonstrate_dynamic_programming():
    """Demonstrate all DP algorithms with examples."""
    print("=== Dynamic Programming Algorithms Demonstration ===\n")

    # Fibonacci
    print("--- Fibonacci Numbers ---")
    for n in [10, 20, 30]:
        memo = fibonacci_memoization(n)
        tab = fibonacci_tabulation(n)
        opt = fibonacci_optimized(n)
        print(f"F({n}): Memoization={memo}, Tabulation={tab}, Optimized={opt}")
    print()

    # Coin Change
    print("--- Coin Change Problem ---")
    coins = [1, 3, 4]
    amount = 6
    min_coins = coin_change_min_coins(coins, amount)
    ways = coin_change_count_ways(coins, amount)
    print(f"Coins: {coins}, Amount: {amount}")
    print(f"Minimum coins: {min_coins}")
    print(f"Number of ways: {ways}")
    print()

    # Knapsack
    print("--- Knapsack Problems ---")
    weights = [1, 3, 4, 5]
    values = [1, 4, 5, 7]
    capacity = 7

    knapsack_01_val = knapsack_01(weights, values, capacity)
    knapsack_01_opt = knapsack_01_optimized(weights, values, capacity)
    unbounded_val = unbounded_knapsack(weights, values, capacity)

    print(f"Items (weights: {weights}, values: {values}), Capacity: {capacity}")
    print(f"0/1 Knapsack: {knapsack_01_val}")
    print(f"0/1 Knapsack (Optimized): {knapsack_01_opt}")
    print(f"Unbounded Knapsack: {unbounded_val}")
    print()

    # String Problems
    print("--- String DP Problems ---")
    text1, text2 = "ABCDGH", "AEDFHR"
    lcs_length, lcs_seq = longest_common_subsequence_with_path(text1, text2)
    print(f"LCS of '{text1}' and '{text2}': length={lcs_length}, sequence='{lcs_seq}'")

    nums = [10, 9, 2, 5, 3, 7, 101, 18]
    lis_length = longest_increasing_subsequence(nums)
    lis_opt = longest_increasing_subsequence_optimized(nums)
    print(f"LIS of {nums}: length={lis_length}, optimized={lis_opt}")

    word1, word2 = "horse", "ros"
    edit_dist = edit_distance(word1, word2)
    print(f"Edit distance between '{word1}' and '{word2}': {edit_dist}")

    s = "aab"
    min_cuts = palindrome_partitioning(s)
    print(f"Palindrome partitioning of '{s}': {min_cuts} cuts")
    print()

    # Matrix Chain Multiplication
    print("--- Matrix Chain Multiplication ---")
    dimensions = [10, 30, 5, 60]
    min_mult = matrix_chain_multiplication(dimensions)
    print(f"Matrix dimensions: {dimensions}")
    print(f"Minimum multiplications: {min_mult}")


if __name__ == "__main__":
    demonstrate_dynamic_programming()
