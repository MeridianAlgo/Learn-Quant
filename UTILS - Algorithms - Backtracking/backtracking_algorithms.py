"""
Backtracking Algorithms - Combinatorial Problems and Constraint Satisfaction
"""


def permutations(nums):
    """
    Generate all permutations of a list
    Time: O(n!)
    """
    result = []

    def backtrack(current, remaining):
        if not remaining:
            result.append(current[:])
            return

        for i in range(len(remaining)):
            current.append(remaining[i])
            backtrack(current, remaining[:i] + remaining[i + 1 :])
            current.pop()

    backtrack([], nums)
    return result


def combinations(n, k):
    """
    Generate all combinations of k numbers from 1 to n
    Time: O(C(n,k))
    """
    result = []

    def backtrack(start, current):
        if len(current) == k:
            result.append(current[:])
            return

        for i in range(start, n + 1):
            current.append(i)
            backtrack(i + 1, current)
            current.pop()

    backtrack(1, [])
    return result


def subsets(nums):
    """
    Generate all subsets (power set)
    Time: O(2^n)
    """
    result = []

    def backtrack(start, current):
        result.append(current[:])

        for i in range(start, len(nums)):
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()

    backtrack(0, [])
    return result


def solve_sudoku(board):
    """
    Solve 9x9 Sudoku puzzle
    Time: O(9^m) where m is number of empty cells
    """

    def is_valid(row, col, num):
        # Check row
        if num in board[row]:
            return False

        # Check column
        if num in [board[i][col] for i in range(9)]:
            return False

        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if board[i][j] == num:
                    return False

        return True

    def solve():
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    for num in range(1, 10):
                        if is_valid(i, j, num):
                            board[i][j] = num
                            if solve():
                                return True
                            board[i][j] = 0
                    return False
        return True

    solve()
    return board


def word_search(board, word):
    """
    Search for word in 2D board
    Time: O(m × n × 4^L) where L is word length
    """
    rows, cols = len(board), len(board[0])

    def backtrack(row, col, index):
        if index == len(word):
            return True

        if row < 0 or row >= rows or col < 0 or col >= cols or board[row][col] != word[index]:
            return False

        temp = board[row][col]
        board[row][col] = "#"  # Mark as visited

        found = (
            backtrack(row + 1, col, index + 1)
            or backtrack(row - 1, col, index + 1)
            or backtrack(row, col + 1, index + 1)
            or backtrack(row, col - 1, index + 1)
        )

        board[row][col] = temp  # Restore
        return found

    for i in range(rows):
        for j in range(cols):
            if backtrack(i, j, 0):
                return True
    return False


def generate_parentheses(n):
    """
    Generate all valid combinations of n pairs of parentheses
    Time: O(4^n / sqrt(n))
    """
    result = []

    def backtrack(current, open_count, close_count):
        if len(current) == 2 * n:
            result.append(current)
            return

        if open_count < n:
            backtrack(current + "(", open_count + 1, close_count)

        if close_count < open_count:
            backtrack(current + ")", open_count, close_count + 1)

    backtrack("", 0, 0)
    return result


if __name__ == "__main__":
    # Permutations
    nums = [1, 2, 3]
    print(f"Permutations of {nums}:")
    print(permutations(nums))

    # Combinations
    print("\nCombinations C(4,2):")
    print(combinations(4, 2))

    # Subsets
    nums = [1, 2, 3]
    print(f"\nSubsets of {nums}:")
    print(subsets(nums))

    # Generate Parentheses
    n = 3
    print(f"\nValid parentheses for n={n}:")
    print(generate_parentheses(n))

    # Word Search
    board = [["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]]
    word = "ABCCED"
    print(f"\nWord '{word}' found in board: {word_search(board, word)}")
