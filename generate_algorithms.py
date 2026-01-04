#!/usr/bin/env python3
"""Generate comprehensive algorithm data for the learning platform"""

algorithms = [
    # Sorting
    ("insertion-sort", "Insertion Sort", "Builds sorted array one item at a time", "Sorting", "Beginner", 15, "O(n²)"),
    ("selection-sort", "Selection Sort", "Finds minimum element and places it at beginning", "Sorting", "Beginner", 15, "O(n²)"),
    ("heap-sort", "Heap Sort", "Uses binary heap data structure for efficient sorting", "Sorting", "Advanced", 30, "O(n log n)"),
    ("counting-sort", "Counting Sort", "Non-comparison integer sorting algorithm", "Sorting", "Intermediate", 20, "O(n + k)"),
    ("radix-sort", "Radix Sort", "Sorts integers by processing digits", "Sorting", "Intermediate", 25, "O(d × (n + k))"),
    
    # Searching
    ("jump-search", "Jump Search", "Searches sorted array by jumping ahead by fixed steps", "Searching", "Intermediate", 15, "O(√n)"),
    ("interpolation-search", "Interpolation Search", "Improved binary search for uniformly distributed data", "Searching", "Intermediate", 20, "O(log log n)"),
    ("exponential-search", "Exponential Search", "Finds range then applies binary search", "Searching", "Intermediate", 15, "O(log n)"),
    
    # Graph
    ("dfs", "Depth-First Search", "Explores as far as possible along each branch", "Graph", "Beginner", 20, "O(V + E)"),
    ("bellman-ford", "Bellman-Ford Algorithm", "Shortest path with negative weights", "Graph", "Advanced", 30, "O(V × E)"),
    ("floyd-warshall", "Floyd-Warshall", "All-pairs shortest paths", "Graph", "Advanced", 35, "O(V³)"),
    ("prims", "Prim's Algorithm", "Minimum spanning tree", "Graph", "Intermediate", 25, "O(E log V)"),
    ("kruskals", "Kruskal's Algorithm", "Minimum spanning tree using union-find", "Graph", "Intermediate", 25, "O(E log E)"),
    ("topological-sort", "Topological Sort", "Linear ordering of directed acyclic graph", "Graph", "Intermediate", 20, "O(V + E)"),
    
    # Dynamic Programming
    ("lcs", "Longest Common Subsequence", "Find longest subsequence common to two sequences", "Dynamic Programming", "Intermediate", 25, "O(m × n)"),
    ("lis", "Longest Increasing Subsequence", "Find longest increasing subsequence", "Dynamic Programming", "Intermediate", 20, "O(n²)"),
    ("edit-distance", "Edit Distance", "Minimum edits to transform one string to another", "Dynamic Programming", "Intermediate", 30, "O(m × n)"),
    ("coin-change", "Coin Change Problem", "Minimum coins needed to make amount", "Dynamic Programming", "Beginner", 20, "O(n × amount)"),
    ("matrix-chain", "Matrix Chain Multiplication", "Optimal parenthesization for matrix multiplication", "Dynamic Programming", "Advanced", 35, "O(n³)"),
    
    # Data Structures
    ("stack", "Stack Implementation", "LIFO data structure", "Data Structures", "Beginner", 10, "O(1)"),
    ("queue", "Queue Implementation", "FIFO data structure", "Data Structures", "Beginner", 10, "O(1)"),
    ("linked-list", "Linked List", "Dynamic linear data structure", "Data Structures", "Beginner", 15, "O(n)"),
    ("binary-tree", "Binary Tree", "Hierarchical tree structure", "Data Structures", "Intermediate", 20, "O(log n)"),
    ("hash-table", "Hash Table", "Key-value pairs with fast lookup", "Data Structures", "Intermediate", 20, "O(1)"),
    ("heap", "Heap (Priority Queue)", "Complete binary tree for priority operations", "Data Structures", "Intermediate", 25, "O(log n)"),
    ("trie", "Trie (Prefix Tree)", "Tree for storing strings efficiently", "Data Structures", "Intermediate", 25, "O(m)"),
    
    # String Algorithms
    ("kmp", "KMP Pattern Matching", "Efficient string pattern matching", "String Algorithms", "Advanced", 30, "O(n + m)"),
    ("rabin-karp", "Rabin-Karp Algorithm", "String matching using hashing", "String Algorithms", "Intermediate", 25, "O(n + m)"),
    ("boyer-moore", "Boyer-Moore Algorithm", "Fast string searching", "String Algorithms", "Advanced", 30, "O(n/m)"),
    
    # Greedy Algorithms
    ("activity-selection", "Activity Selection", "Select maximum non-overlapping activities", "Greedy", "Beginner", 15, "O(n log n)"),
    ("huffman-coding", "Huffman Coding", "Optimal prefix-free encoding", "Greedy", "Intermediate", 30, "O(n log n)"),
    ("fractional-knapsack", "Fractional Knapsack", "Maximize value with fractional items", "Greedy", "Beginner", 15, "O(n log n)"),
    
    # Backtracking
    ("n-queens", "N-Queens Problem", "Place N queens on N×N chessboard", "Backtracking", "Advanced", 35, "O(n!)"),
    ("sudoku-solver", "Sudoku Solver", "Solve 9×9 Sudoku puzzle", "Backtracking", "Advanced", 40, "O(9^m)"),
    ("subset-sum", "Subset Sum", "Find subset with given sum", "Backtracking", "Intermediate", 25, "O(2^n)"),
]

print(f"Total algorithms: {len(algorithms)}")
for alg in algorithms[:5]:
    print(f"  - {alg[1]} ({alg[3]})")
