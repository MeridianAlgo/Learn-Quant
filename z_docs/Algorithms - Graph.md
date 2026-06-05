<p class="lq-badges"><span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-cat">Algorithms</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Algorithms - Graph"
    python "graph_algorithms.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Algorithms%20-%20Graph)

---
# Algorithms – Graph

## Overview

Graph algorithms operate on structures composed of vertices (nodes) and edges (connections). Many financial problems are naturally modelled as graphs: currency markets form weighted directed graphs, asset correlation matrices define undirected weighted graphs, and order routing networks are flow graphs.

Understanding graph algorithms allows quants to detect arbitrage opportunities, analyse portfolio interconnectedness, and model contagion in financial networks.

## Key Concepts

### Graph Representations

| Representation | Space | Edge lookup | Adjacency iteration | Best for |
|---------------|-------|-------------|---------------------|---------|
| Adjacency list | O(V+E) | O(degree) | O(degree) | Sparse graphs |
| Adjacency matrix | O(V^2) | O(1) | O(V) | Dense graphs |
| Edge list | O(E) | O(E) | O(E) | Sorting edges |

### Core Traversals

**BFS (Breadth-First Search)**
- Explores all neighbours at the current depth before moving deeper.
- Finds shortest paths in unweighted graphs.
- Time: O(V + E).

**DFS (Depth-First Search)**
- Explores as far as possible along a branch before backtracking.
- Used for cycle detection, topological sort, connected components.
- Time: O(V + E).

### Shortest Path Algorithms

| Algorithm | Graph type | Time complexity |
|-----------|-----------|----------------|
| BFS | Unweighted | O(V + E) |
| Dijkstra | Non-negative weights | O((V + E) log V) |
| Bellman-Ford | Any weights (detects neg cycles) | O(V * E) |
| Floyd-Warshall | All-pairs | O(V^3) |

### Negative Cycle Detection
Bellman-Ford can detect negative-weight cycles, which map directly to **triangular arbitrage** opportunities in currency markets.

## Files
- `graph_algorithms.py`: Graph data structure, BFS, DFS, Dijkstra's algorithm, and negative cycle detection with currency arbitrage examples.

## How to Run
```bash
python graph_algorithms.py
```

## Financial Applications

### 1. Triangular Arbitrage Detection
- Model a currency market as a directed graph: each currency is a vertex, each exchange rate is a weighted edge (weight = -log(rate)).
- A negative-weight cycle in this graph means a sequence of currency conversions that produces a profit with no risk.
- Bellman-Ford detects negative cycles in O(V * E).

### 2. Portfolio Correlation Networks
- Model assets as vertices and pairwise correlations as edges.
- Minimum Spanning Tree (MST) of the correlation graph reveals the dominant relationships and clusters.
- Used in risk decomposition and identifying systemic exposure.

### 3. Trade Routing and Smart Order Routing
- Model trading venues as vertices and available routing paths as edges.
- Shortest path algorithms find the optimal routing to minimise market impact or latency.

### 4. Contagion and Systemic Risk
- Model banks or funds as vertices and exposure/lending relationships as directed edges.
- Graph centrality measures (PageRank, betweenness) identify systemically important institutions.

### 5. Supply Chain Finance
- Model supplier/buyer relationships as a directed graph to evaluate credit risk propagation and working capital flows.

## Best Practices

- **Choose the right representation**: Use adjacency lists for sparse financial networks (most real-world cases); adjacency matrices only when edge queries dominate.
- **Handle negative weights carefully**: Dijkstra's fails with negative weights — use Bellman-Ford for arbitrage detection where log-rates can be negative.
- **Scale matters**: Real FX markets have ~180 currencies — O(V^3) Floyd-Warshall may be too slow; use Bellman-Ford from a single source instead.
- **Filter weak edges**: In correlation networks, only retain edges above a threshold (e.g., |correlation| > 0.5) to avoid noise-driven spurious connections.


---

## Continue in Algorithms

<div class="grid cards" markdown>

-   :material-sitemap-outline: __[Algorithms - Backtracking](Algorithms - Backtracking.md)__

    Backtracking is a general algorithmic technique for solving problems by building candidates incrementally and abandoning a candidate ("backtracking") as soon as it is determined to violate the problem constraints. It is a systematic form of exhaustive search that prunes the search space to avoid exploring clearly invalid paths.

-   :material-sitemap-outline: __[Algorithms - Dynamic Programming](Algorithms - Dynamic Programming.md)__

    Dynamic Programming (DP) is an algorithmic technique for solving problems by breaking them into overlapping subproblems, solving each subproblem once, and storing the result to avoid redundant computation. It converts exponential-time recursive solutions into polynomial-time ones.

-   :material-sitemap-outline: __[Algorithms - Machine Learning](Algorithms - Machine Learning.md)__

    This module implements fundamental machine learning algorithms from scratch using only NumPy — no scikit-learn or frameworks. Building these algorithms by hand is the most effective way to understand what happens inside the black boxes used in production trading systems.

-   :material-sitemap-outline: __[Algorithms - Searching](Algorithms - Searching.md)__

    Searching algorithms find a target value within a data structure. The choice of algorithm determines whether a search takes O(n) time (checking every element) or O(log n) time (dividing the search space in half each step). In latency-sensitive financial systems, this difference is meaningful at scale.

-   :material-sitemap-outline: __[Algorithms - Sorting](Algorithms - Sorting.md)__

    A comprehensive implementation of fundamental sorting algorithms with detailed explanations, complexity analysis, and performance comparisons.

-   :material-sitemap-outline: __[Algorithms - String](Algorithms - String.md)__

    String algorithms handle efficient manipulation, searching, and analysis of text data. In quantitative finance, string processing is essential for parsing market data feeds, extracting information from news and filings, matching ticker symbols, and cleaning raw data from APIs.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
