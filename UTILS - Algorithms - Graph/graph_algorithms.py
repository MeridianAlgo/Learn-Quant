"""
Graph Algorithms Implementation
A comprehensive collection of graph algorithms with detailed explanations and examples.
"""

import heapq
import math
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Union


class Graph:
    """Graph data structure supporting both directed and undirected graphs."""

    def __init__(self, directed: bool = False):
        self.directed = directed
        self.adj_list: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self.vertices: Set[str] = set()

    def add_edge(self, u: str, v: str, weight: int = 1) -> None:
        """Add edge between vertices u and v with given weight."""
        self.vertices.add(u)
        self.vertices.add(v)
        self.adj_list[u].append((v, weight))
        if not self.directed:
            self.adj_list[v].append((u, weight))

    def add_vertex(self, vertex: str) -> None:
        """Add isolated vertex."""
        self.vertices.add(vertex)

    def get_neighbors(self, vertex: str) -> List[Tuple[str, int]]:
        """Get neighbors of vertex with their weights."""
        return self.adj_list[vertex]

    def __str__(self) -> str:
        return f"Graph(vertices={len(self.vertices)}, directed={self.directed})"


def bfs(graph: Graph, start: str) -> Dict[str, Union[int, List[str]]]:
    """
    Breadth-First Search: Level-order traversal of graph.
    Explores all neighbors at current depth before moving to next level.

    Time Complexity: O(V + E)
    Space Complexity: O(V)

    Args:
        graph: Graph to traverse
        start: Starting vertex

    Returns:
        Dictionary with distances and paths from start vertex
    """
    visited = set()
    distances = {start: 0}
    parents = {start: None}
    queue = deque([start])
    visited.add(start)

    while queue:
        current = queue.popleft()

        for neighbor, _ in graph.get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                distances[neighbor] = distances[current] + 1
                parents[neighbor] = current
                queue.append(neighbor)

    # Reconstruct paths
    paths = {}
    for vertex in distances:
        path = []
        current = vertex
        while current is not None:
            path.append(current)
            current = parents[current]
        paths[vertex] = list(reversed(path))

    return {"distances": distances, "paths": paths, "visited": visited}


def dfs(graph: Graph, start: str) -> Dict[str, Union[int, List[str], List[str]]]:
    """
    Depth-First Search: Deep traversal of graph.
    Explores as far as possible along each branch before backtracking.

    Time Complexity: O(V + E)
    Space Complexity: O(V)

    Args:
        graph: Graph to traverse
        start: Starting vertex

    Returns:
        Dictionary with discovery times, finish times, and traversal order
    """
    visited = set()
    discovery_times = {}
    finish_times = {}
    traversal_order = []
    time = 0

    def dfs_recursive(vertex: str):
        nonlocal time
        visited.add(vertex)
        time += 1
        discovery_times[vertex] = time
        traversal_order.append(vertex)

        for neighbor, _ in graph.get_neighbors(vertex):
            if neighbor not in visited:
                dfs_recursive(neighbor)

        time += 1
        finish_times[vertex] = time

    dfs_recursive(start)

    return {
        "discovery_times": discovery_times,
        "finish_times": finish_times,
        "traversal_order": traversal_order,
        "visited": visited,
    }


def dijkstra(graph: Graph, start: str) -> Dict[str, Union[int, List[str]]]:
    """
    Dijkstra's Algorithm: Shortest path from single source to all vertices.
    Works only for graphs with non-negative edge weights.

    Time Complexity: O((V + E) log V) with min-heap
    Space Complexity: O(V)

    Args:
        graph: Weighted graph with non-negative weights
        start: Starting vertex

    Returns:
        Dictionary with shortest distances and paths from start vertex
    """
    distances = dict.fromkeys(graph.vertices, math.inf)
    distances[start] = 0
    parents = {start: None}
    visited = set()
    heap = [(0, start)]

    while heap:
        current_distance, current = heapq.heappop(heap)

        if current in visited:
            continue

        visited.add(current)

        for neighbor, weight in graph.get_neighbors(current):
            if neighbor not in visited:
                distance = current_distance + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    parents[neighbor] = current
                    heapq.heappush(heap, (distance, neighbor))

    # Reconstruct paths
    paths = {}
    for vertex in graph.vertices:
        if distances[vertex] != math.inf:
            path = []
            current = vertex
            while current is not None:
                path.append(current)
                current = parents[current]
            paths[vertex] = list(reversed(path))
        else:
            paths[vertex] = []

    return {
        "distances": distances,
        "paths": paths,
        "unreachable": {v for v, d in distances.items() if d == math.inf},
    }


def bellman_ford(graph: Graph, start: str) -> Dict[str, Union[int, List[str], bool]]:
    """
    Bellman-Ford Algorithm: Shortest paths with negative edge weight detection.
    Can handle negative weights and detect negative cycles.

    Time Complexity: O(V * E)
    Space Complexity: O(V)

    Args:
        graph: Weighted graph (can have negative weights)
        start: Starting vertex

    Returns:
        Dictionary with distances, paths, and negative cycle detection
    """
    distances = dict.fromkeys(graph.vertices, math.inf)
    distances[start] = 0
    parents = {start: None}

    # Relax edges V-1 times
    for _ in range(len(graph.vertices) - 1):
        for u in graph.vertices:
            for v, weight in graph.get_neighbors(u):
                if distances[u] != math.inf and distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    parents[v] = u

    # Check for negative cycles
    has_negative_cycle = False
    for u in graph.vertices:
        for v, weight in graph.get_neighbors(u):
            if distances[u] != math.inf and distances[u] + weight < distances[v]:
                has_negative_cycle = True
                break

    # Reconstruct paths
    paths = {}
    for vertex in graph.vertices:
        if distances[vertex] != math.inf:
            path = []
            current = vertex
            while current is not None:
                path.append(current)
                current = parents[current]
            paths[vertex] = list(reversed(path))
        else:
            paths[vertex] = []

    return {
        "distances": distances,
        "paths": paths,
        "has_negative_cycle": has_negative_cycle,
        "unreachable": {v for v, d in distances.items() if d == math.inf},
    }


def floyd_warshall(graph: Graph) -> Dict[str, Union[Dict[str, int], List[str]]]:
    """
    Floyd-Warshall Algorithm: All-pairs shortest paths.
    Computes shortest paths between all pairs of vertices.

    Time Complexity: O(V³)
    Space Complexity: O(V²)

    Args:
        graph: Weighted graph

    Returns:
        Dictionary with distance matrix and path matrix
    """
    vertices = list(graph.vertices)

    # Initialize distance matrix
    distances = {u: dict.fromkeys(vertices, math.inf) for u in vertices}
    for u in vertices:
        distances[u][u] = 0

    # Set direct edge distances
    for u in vertices:
        for v, weight in graph.get_neighbors(u):
            distances[u][v] = weight

    # Floyd-Warshall main algorithm
    for k in vertices:
        for i in vertices:
            for j in vertices:
                if distances[i][k] != math.inf and distances[k][j] != math.inf:
                    if distances[i][j] > distances[i][k] + distances[k][j]:
                        distances[i][j] = distances[i][k] + distances[k][j]

    return {"distances": distances, "vertices": vertices}


def kruskal_mst(graph: Graph) -> Dict[str, Union[List[Tuple[str, str, int]], int]]:
    """
    Kruskal's Algorithm: Minimum Spanning Tree using Union-Find.
    Finds MST for weighted undirected graphs.

    Time Complexity: O(E log E)
    Space Complexity: O(V + E)

    Args:
        graph: Weighted undirected graph

    Returns:
        Dictionary with MST edges and total weight
    """

    class UnionFind:
        def __init__(self):
            self.parent = {}
            self.rank = {}

        def make_set(self, vertex):
            self.parent[vertex] = vertex
            self.rank[vertex] = 0

        def find(self, vertex):
            if self.parent[vertex] != vertex:
                self.parent[vertex] = self.find(self.parent[vertex])
            return self.parent[vertex]

        def union(self, vertex1, vertex2):
            root1 = self.find(vertex1)
            root2 = self.find(vertex2)

            if root1 != root2:
                if self.rank[root1] < self.rank[root2]:
                    self.parent[root1] = root2
                elif self.rank[root1] > self.rank[root2]:
                    self.parent[root2] = root1
                else:
                    self.parent[root2] = root1
                    self.rank[root1] += 1

    if graph.directed:
        raise ValueError("Kruskal's algorithm requires undirected graph")

    # Collect all edges
    edges = []
    for u in graph.vertices:
        for v, weight in graph.get_neighbors(u):
            if u < v:  # Avoid duplicate edges
                edges.append((weight, u, v))

    # Sort edges by weight
    edges.sort()

    uf = UnionFind()
    for vertex in graph.vertices:
        uf.make_set(vertex)

    mst_edges = []
    total_weight = 0

    for weight, u, v in edges:
        if uf.find(u) != uf.find(v):
            uf.union(u, v)
            mst_edges.append((u, v, weight))
            total_weight += weight

    return {"edges": mst_edges, "total_weight": total_weight}


def prim_mst(graph: Graph, start: str) -> Dict[str, Union[List[Tuple[str, str, int]], int]]:
    """
    Prim's Algorithm: Minimum Spanning Tree using min-heap.
    Grows MST from a starting vertex.

    Time Complexity: O((V + E) log V)
    Space Complexity: O(V)

    Args:
        graph: Weighted undirected graph
        start: Starting vertex for MST

    Returns:
        Dictionary with MST edges and total weight
    """
    if graph.directed:
        raise ValueError("Prim's algorithm requires undirected graph")

    visited = set([start])
    mst_edges = []
    total_weight = 0
    heap = []

    # Add all edges from start vertex to heap
    for neighbor, weight in graph.get_neighbors(start):
        heapq.heappush(heap, (weight, start, neighbor))

    while heap and len(visited) < len(graph.vertices):
        weight, u, v = heapq.heappop(heap)

        if v not in visited:
            visited.add(v)
            mst_edges.append((u, v, weight))
            total_weight += weight

            for neighbor, w in graph.get_neighbors(v):
                if neighbor not in visited:
                    heapq.heappush(heap, (w, v, neighbor))

    return {"edges": mst_edges, "total_weight": total_weight}


def demonstrate_graph_algorithms():
    """Demonstrate all graph algorithms with examples."""
    print("=== Graph Algorithms Demonstration ===\n")

    # Create sample graph
    graph = Graph(directed=False)
    edges = [
        ("A", "B", 4),
        ("A", "C", 2),
        ("B", "C", 1),
        ("B", "D", 5),
        ("C", "D", 8),
        ("C", "E", 10),
        ("D", "E", 2),
        ("D", "F", 6),
        ("E", "F", 3),
    ]

    for u, v, weight in edges:
        graph.add_edge(u, v, weight)

    print(f"Graph: {graph}")
    print("Edges:", edges)
    print()

    # BFS
    print("--- Breadth-First Search (BFS) from A ---")
    bfs_result = bfs(graph, "A")
    print("Distances:", bfs_result["distances"])
    print("Paths:", bfs_result["paths"])
    print()

    # DFS
    print("--- Depth-First Search (DFS) from A ---")
    dfs_result = dfs(graph, "A")
    print("Traversal order:", dfs_result["traversal_order"])
    print("Discovery times:", dfs_result["discovery_times"])
    print()

    # Dijkstra
    print("--- Dijkstra's Shortest Paths from A ---")
    dijkstra_result = dijkstra(graph, "A")
    print("Shortest distances:", dijkstra_result["distances"])
    print("Shortest paths:", dijkstra_result["paths"])
    print()

    # MST
    print("--- Minimum Spanning Tree ---")
    kruskal_result = kruskal_mst(graph)
    prim_result = prim_mst(graph, "A")

    print("Kruskal's MST edges:", kruskal_result["edges"])
    print("Kruskal's MST weight:", kruskal_result["total_weight"])
    print("Prim's MST edges:", prim_result["edges"])
    print("Prim's MST weight:", prim_result["total_weight"])
    print()

    # Floyd-Warshall
    print("--- Floyd-Warshall All-Pairs Shortest Paths ---")
    fw_result = floyd_warshall(graph)
    print("Distance matrix:")
    for u in fw_result["vertices"]:
        row = [
            (f"{fw_result['distances'][u][v]:3}" if fw_result["distances"][u][v] != math.inf else " ∞")
            for v in fw_result["vertices"]
        ]
        print(f"{u}: {row}")


if __name__ == "__main__":
    demonstrate_graph_algorithms()
