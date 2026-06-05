<p class="lq-badges"><span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-cat">Algorithms</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Algorithms - Tree"
    python "tree_algorithms.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Algorithms%20-%20Tree)

---
# Algorithms – Tree

## Overview

Tree data structures organise data hierarchically to enable efficient search, insertion, and deletion. Binary Search Trees (BSTs) and their balanced variants (AVL trees, Red-Black trees) are the foundation of many performance-critical systems in finance, including order book matching engines, index structures for time-series databases, and priority queues for event-driven simulations.

## Key Concepts

### Binary Tree
Each node has at most two children (left and right). The tree has no ordering constraint by itself.

### Binary Search Tree (BST)
A binary tree where for every node:
- All values in the **left** subtree are **less than** the node's value.
- All values in the **right** subtree are **greater than** the node's value.

This invariant makes search, insert, and delete O(h) where h is the tree height.

| Operation | Average case | Worst case (degenerate) |
|-----------|-------------|------------------------|
| Search | O(log n) | O(n) |
| Insert | O(log n) | O(n) |
| Delete | O(log n) | O(n) |

### AVL Tree (Balanced BST)
An AVL tree maintains a balance invariant: the heights of the left and right subtrees of any node differ by at most 1. After each insert or delete, rotations restore balance.

- Guaranteed O(log n) for all operations.
- Balance factor = height(left) - height(right); must be -1, 0, or +1.

### Tree Traversals

| Traversal | Order | Use case |
|-----------|-------|---------|
| Inorder | Left, Root, Right | Produces sorted output from BST |
| Preorder | Root, Left, Right | Serialise tree structure |
| Postorder | Left, Right, Root | Evaluate expression trees |
| Level-order (BFS) | Level by level | Shortest path, tree width |

### Heap
A complete binary tree where each parent is larger (max-heap) or smaller (min-heap) than its children. Supports O(1) access to the maximum/minimum and O(log n) insert/remove. Used to implement priority queues.

## Files
- `tree_algorithms.py`: TreeNode class, BST insert/search/delete, AVL tree with rotations, all traversal methods, and heap operations.

## How to Run
```bash
python tree_algorithms.py
```

## Financial Applications

### 1. Order Book Implementation
- Price levels in an order book are maintained as a sorted BST (or balanced variant).
- O(log n) insert of new orders and O(log n) cancellation by price level.
- Inorder traversal produces the full book sorted by price.

### 2. Priority Queue for Event-Driven Simulation
- A min-heap processes market events (orders, cancellations, trades) in chronological order.
- Used in backtesting engines to simulate accurate event sequencing.

### 3. Interval Trees for Option Strike Ranges
- An interval tree (augmented BST) answers "which strikes are in the range [K1, K2]?" in O(log n + k) time.
- Used in options market-making to identify relevant strikes quickly.

### 4. Expression Trees for Formula Evaluation
- Pricing model formulas can be parsed into binary expression trees for efficient repeated evaluation.
- Postorder traversal evaluates the formula bottom-up.

### 5. Time-Series Index Structures
- B-trees (generalised BSTs) underlie database indexes.
- Range queries like "all trades between 09:30 and 09:45" use the tree structure for O(log n + k) retrieval.

## Best Practices

- **Use Python's `heapq` for priority queues**: The standard library heap is implemented in C and is efficient. Prefer it over a hand-rolled tree unless you need full BST operations.
- **Balance matters in production**: An unbalanced BST degrades to a linked list (O(n) operations) on sorted input — always use AVL or Red-Black trees for production order books.
- **Consider `sortedcontainers.SortedList`**: For Python applications, this library provides O(log n) sorted operations with better constants than a hand-written AVL tree.
- **Understand traversal order**: Inorder traversal on a BST always produces a sorted sequence — use it for auditing the order book state.


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

-   :material-sitemap-outline: __[Algorithms - Sorting](Algorithms - Sorting.md)__

    A comprehensive implementation of fundamental sorting algorithms with detailed explanations, complexity analysis, and performance comparisons.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
