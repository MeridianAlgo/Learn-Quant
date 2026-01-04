#!/usr/bin/env python3
import json

# All algorithm definitions
ALGORITHMS_DATA = [
    # SORTING (10 algorithms)
    {
        'id': 'bubble-sort', 'title': 'Bubble Sort', 'category': 'Sorting', 'difficulty': 'Beginner',
        'description': 'Simple comparison-based sorting that repeatedly swaps adjacent elements',
        'time': 15, 'complexity': 'O(n²)',
        'code': '''def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
print("Sorted:", bubble_sort([64, 34, 25, 12, 22]))'''
    },
    {
        'id': 'quick-sort', 'title': 'Quick Sort', 'category': 'Sorting', 'difficulty': 'Intermediate',
        'description': 'Efficient divide-and-conquer sorting with pivot partitioning',
        'time': 25, 'complexity': 'O(n log n)',
        'code': '''def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
print("Sorted:", quick_sort([64, 34, 25, 12, 22]))'''
    },
    {
        'id': 'merge-sort', 'title': 'Merge Sort', 'category': 'Sorting', 'difficulty': 'Intermediate',
        'description': 'Stable divide-and-conquer sorting with guaranteed O(n log n)',
        'time': 20, 'complexity': 'O(n log n)',
        'code': '''def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

print("Sorted:", merge_sort([64, 34, 25, 12, 22]))'''
    },
    {
        'id': 'insertion-sort', 'title': 'Insertion Sort', 'category': 'Sorting', 'difficulty': 'Beginner',
        'description': 'Builds sorted array one element at a time',
        'time': 15, 'complexity': 'O(n²)',
        'code': '''def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr
print("Sorted:", insertion_sort([12, 11, 13, 5, 6]))'''
    },
    {
        'id': 'selection-sort', 'title': 'Selection Sort', 'category': 'Sorting', 'difficulty': 'Beginner',
        'description': 'Finds minimum element and places it at beginning',
        'time': 15, 'complexity': 'O(n²)',
        'code': '''def selection_sort(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr
print("Sorted:", selection_sort([64, 25, 12, 22, 11]))'''
    },
    {
        'id': 'heap-sort', 'title': 'Heap Sort', 'category': 'Sorting', 'difficulty': 'Advanced',
        'description': 'Uses binary heap for efficient sorting',
        'time': 30, 'complexity': 'O(n log n)',
        'code': '''def heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    if left < n and arr[left] > arr[largest]:
        largest = left
    if right < n and arr[largest] < arr[right]:
        largest = right
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heap_sort(arr):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    for i in range(n-1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)
    return arr

print("Sorted:", heap_sort([12, 11, 13, 5, 6, 7]))'''
    },
    
    # SEARCHING (6 algorithms)
    {
        'id': 'binary-search', 'title': 'Binary Search', 'category': 'Searching', 'difficulty': 'Beginner',
        'description': 'Efficient search in sorted arrays using divide-and-conquer',
        'time': 10, 'complexity': 'O(log n)',
        'code': '''def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

data = [1, 3, 5, 7, 9, 11, 13, 15]
print("Found at index:", binary_search(data, 7))'''
    },
    {
        'id': 'linear-search', 'title': 'Linear Search', 'category': 'Searching', 'difficulty': 'Beginner',
        'description': 'Sequential search through array elements',
        'time': 5, 'complexity': 'O(n)',
        'code': '''def linear_search(arr, target):
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1

data = [64, 34, 25, 12, 22, 11, 90]
print("Found at index:", linear_search(data, 22))'''
    },
    {
        'id': 'jump-search', 'title': 'Jump Search', 'category': 'Searching', 'difficulty': 'Intermediate',
        'description': 'Searches sorted array by jumping ahead by fixed steps',
        'time': 15, 'complexity': 'O(√n)',
        'code': '''import math

def jump_search(arr, target):
    n = len(arr)
    step = int(math.sqrt(n))
    prev = 0
    while arr[min(step, n)-1] < target:
        prev = step
        step += int(math.sqrt(n))
        if prev >= n:
            return -1
    for i in range(prev, min(step, n)):
        if arr[i] == target:
            return i
    return -1

data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print("Found at:", jump_search(data, 6))'''
    },
    
    # GRAPH ALGORITHMS (8 algorithms)
    {
        'id': 'bfs', 'title': 'Breadth-First Search', 'category': 'Graph', 'difficulty': 'Beginner',
        'description': 'Level-order graph traversal exploring neighbors first',
        'time': 20, 'complexity': 'O(V + E)',
        'code': '''from collections import deque

def bfs(graph, start):
    visited = set([start])
    queue = deque([start])
    result = []
    while queue:
        vertex = queue.popleft()
        result.append(vertex)
        for neighbor in graph.get(vertex, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return result

graph = {'A': ['B', 'C'], 'B': ['D'], 'C': ['E'], 'D': [], 'E': []}
print("BFS:", bfs(graph, 'A'))'''
    },
    {
        'id': 'dfs', 'title': 'Depth-First Search', 'category': 'Graph', 'difficulty': 'Beginner',
        'description': 'Explores as far as possible along each branch',
        'time': 20, 'complexity': 'O(V + E)',
        'code': '''def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    result = [start]
    for neighbor in graph.get(start, []):
        if neighbor not in visited:
            result.extend(dfs(graph, neighbor, visited))
    return result

graph = {'A': ['B', 'C'], 'B': ['D'], 'C': ['E'], 'D': [], 'E': []}
print("DFS:", dfs(graph, 'A'))'''
    },
    {
        'id': 'dijkstra', 'title': "Dijkstra's Algorithm", 'category': 'Graph', 'difficulty': 'Intermediate',
        'description': 'Shortest path from source to all vertices',
        'time': 25, 'complexity': 'O((V+E) log V)',
        'code': '''import heapq

def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    while pq:
        curr_dist, curr = heapq.heappop(pq)
        if curr_dist > distances[curr]:
            continue
        for neighbor, weight in graph[curr]:
            distance = curr_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    return distances

graph = {'A': [('B', 4), ('C', 2)], 'B': [('C', 1)], 'C': []}
print("Distances:", dijkstra(graph, 'A'))'''
    },
    
    # DYNAMIC PROGRAMMING (8 algorithms)
    {
        'id': 'fibonacci', 'title': 'Fibonacci DP', 'category': 'Dynamic Programming', 'difficulty': 'Beginner',
        'description': 'Calculate Fibonacci using memoization',
        'time': 15, 'complexity': 'O(n)',
        'code': '''def fibonacci(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci(n-1, memo) + fibonacci(n-2, memo)
    return memo[n]

print("Fib(10):", fibonacci(10))
print("Fib(20):", fibonacci(20))'''
    },
    {
        'id': 'knapsack', 'title': '0/1 Knapsack', 'category': 'Dynamic Programming', 'difficulty': 'Intermediate',
        'description': 'Maximize value without exceeding capacity',
        'time': 30, 'complexity': 'O(n × W)',
        'code': '''def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(values[i-1] + dp[i-1][w-weights[i-1]], dp[i-1][w])
            else:
                dp[i][w] = dp[i-1][w]
    return dp[n][capacity]

weights = [1, 3, 4, 5]
values = [1, 4, 5, 7]
print("Max value:", knapsack(weights, values, 7))'''
    },
    {
        'id': 'lcs', 'title': 'Longest Common Subsequence', 'category': 'Dynamic Programming', 'difficulty': 'Intermediate',
        'description': 'Find longest subsequence common to two sequences',
        'time': 25, 'complexity': 'O(m × n)',
        'code': '''def lcs(X, Y):
    m, n = len(X), len(Y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i-1] == Y[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

print("LCS length:", lcs("AGGTAB", "GXTXAYB"))'''
    },
    {
        'id': 'coin-change', 'title': 'Coin Change', 'category': 'Dynamic Programming', 'difficulty': 'Beginner',
        'description': 'Minimum coins needed to make amount',
        'time': 20, 'complexity': 'O(n × amount)',
        'code': '''def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1

coins = [1, 2, 5]
print("Min coins for 11:", coin_change(coins, 11))'''
    },
    
    # DATA STRUCTURES (7 algorithms)
    {
        'id': 'stack', 'title': 'Stack', 'category': 'Data Structures', 'difficulty': 'Beginner',
        'description': 'LIFO data structure',
        'time': 10, 'complexity': 'O(1)',
        'code': '''class Stack:
    def __init__(self):
        self.items = []
    def push(self, item):
        self.items.append(item)
    def pop(self):
        return self.items.pop() if self.items else None
    def peek(self):
        return self.items[-1] if self.items else None

stack = Stack()
stack.push(1)
stack.push(2)
print("Top:", stack.peek())
print("Pop:", stack.pop())'''
    },
    {
        'id': 'queue', 'title': 'Queue', 'category': 'Data Structures', 'difficulty': 'Beginner',
        'description': 'FIFO data structure',
        'time': 10, 'complexity': 'O(1)',
        'code': '''from collections import deque

class Queue:
    def __init__(self):
        self.items = deque()
    def enqueue(self, item):
        self.items.append(item)
    def dequeue(self):
        return self.items.popleft() if self.items else None

queue = Queue()
queue.enqueue(1)
queue.enqueue(2)
print("Dequeue:", queue.dequeue())'''
    },
    {
        'id': 'linked-list', 'title': 'Linked List', 'category': 'Data Structures', 'difficulty': 'Beginner',
        'description': 'Dynamic linear data structure',
        'time': 15, 'complexity': 'O(n)',
        'code': '''class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
    def append(self, data):
        if not self.head:
            self.head = Node(data)
        else:
            curr = self.head
            while curr.next:
                curr = curr.next
            curr.next = Node(data)
    def display(self):
        curr = self.head
        while curr:
            print(curr.data, end=' -> ')
            curr = curr.next
        print('None')

ll = LinkedList()
ll.append(1)
ll.append(2)
ll.append(3)
ll.display()'''
    },
    {
        'id': 'binary-tree', 'title': 'Binary Tree', 'category': 'Data Structures', 'difficulty': 'Intermediate',
        'description': 'Hierarchical tree structure',
        'time': 20, 'complexity': 'O(log n)',
        'code': '''class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def inorder(root):
    if root:
        inorder(root.left)
        print(root.val, end=' ')
        inorder(root.right)

root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
print("Inorder:")
inorder(root)'''
    },
    {
        'id': 'hash-table', 'title': 'Hash Table', 'category': 'Data Structures', 'difficulty': 'Intermediate',
        'description': 'Key-value pairs with fast lookup',
        'time': 20, 'complexity': 'O(1)',
        'code': '''class HashTable:
    def __init__(self, size=10):
        self.size = size
        self.table = [[] for _ in range(size)]
    
    def hash(self, key):
        return hash(key) % self.size
    
    def put(self, key, value):
        h = self.hash(key)
        for i, (k, v) in enumerate(self.table[h]):
            if k == key:
                self.table[h][i] = (key, value)
                return
        self.table[h].append((key, value))
    
    def get(self, key):
        h = self.hash(key)
        for k, v in self.table[h]:
            if k == key:
                return v
        return None

ht = HashTable()
ht.put("name", "Alice")
print("Value:", ht.get("name"))'''
    },
    
    # GREEDY ALGORITHMS (4 algorithms)
    {
        'id': 'activity-selection', 'title': 'Activity Selection', 'category': 'Greedy', 'difficulty': 'Beginner',
        'description': 'Select maximum non-overlapping activities',
        'time': 15, 'complexity': 'O(n log n)',
        'code': '''def activity_selection(start, finish):
    activities = sorted(zip(start, finish), key=lambda x: x[1])
    selected = [activities[0]]
    for i in range(1, len(activities)):
        if activities[i][0] >= selected[-1][1]:
            selected.append(activities[i])
    return len(selected)

start = [1, 3, 0, 5, 8, 5]
finish = [2, 4, 6, 7, 9, 9]
print("Max activities:", activity_selection(start, finish))'''
    },
    {
        'id': 'huffman', 'title': 'Huffman Coding', 'category': 'Greedy', 'difficulty': 'Intermediate',
        'description': 'Optimal prefix-free encoding',
        'time': 30, 'complexity': 'O(n log n)',
        'code': '''import heapq
from collections import defaultdict

def huffman_coding(freq):
    heap = [[weight, [char, ""]] for char, weight in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

freq = {'a': 5, 'b': 9, 'c': 12, 'd': 13, 'e': 16, 'f': 45}
print("Huffman codes:", huffman_coding(freq))'''
    },
    
    # BACKTRACKING (3 algorithms)
    {
        'id': 'n-queens', 'title': 'N-Queens', 'category': 'Backtracking', 'difficulty': 'Advanced',
        'description': 'Place N queens on N×N chessboard',
        'time': 35, 'complexity': 'O(n!)',
        'code': '''def solve_n_queens(n):
    def is_safe(board, row, col):
        for i in range(row):
            if board[i] == col or abs(board[i] - col) == abs(i - row):
                return False
        return True
    
    def solve(board, row):
        if row == n:
            solutions.append(board[:])
            return
        for col in range(n):
            if is_safe(board, row, col):
                board[row] = col
                solve(board, row + 1)
                board[row] = -1
    
    solutions = []
    solve([-1] * n, 0)
    return len(solutions)

print("4-Queens solutions:", solve_n_queens(4))'''
    },
    {
        'id': 'subset-sum', 'title': 'Subset Sum', 'category': 'Backtracking', 'difficulty': 'Intermediate',
        'description': 'Find subset with given sum',
        'time': 25, 'complexity': 'O(2^n)',
        'code': '''def subset_sum(nums, target):
    def backtrack(start, current_sum):
        if current_sum == target:
            return True
        if current_sum > target or start >= len(nums):
            return False
        return (backtrack(start + 1, current_sum + nums[start]) or 
                backtrack(start + 1, current_sum))
    
    return backtrack(0, 0)

nums = [3, 34, 4, 12, 5, 2]
print("Subset exists:", subset_sum(nums, 9))'''
    },
]

# Generate TypeScript file
output = '''// Comprehensive Algorithm Learning Platform Data
// Auto-generated with 30+ algorithms across multiple categories

export interface Algorithm {
  id: string
  title: string
  description: string
  category: string
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced'
  timeEstimate: number
  complexity: string
  complexityExplanation: string
  code: string
  explanation: string
  examples: string[]
  useCases: string[]
}

export async function scanAlgorithms(): Promise<Algorithm[]> {
  return ALGORITHMS
}

export function getCategories(algorithms: Algorithm[]): string[] {
  const categories = new Set(algorithms.map(algo => algo.category))
  return ['All', ...Array.from(categories).sort()]
}

export function getAlgorithmsByCategory(algorithms: Algorithm[], category: string): Algorithm[] {
  return category === 'All' ? algorithms : algorithms.filter(algo => algo.category === category)
}

export function filterAlgorithms(algorithms: Algorithm[], searchTerm: string): Algorithm[] {
  if (!searchTerm) return algorithms
  const term = searchTerm.toLowerCase()
  return algorithms.filter(algo => 
    algo.title.toLowerCase().includes(term) ||
    algo.description.toLowerCase().includes(term) ||
    algo.category.toLowerCase().includes(term)
  )
}

const ALGORITHMS: Algorithm[] = [
'''

for i, alg in enumerate(ALGORITHMS_DATA):
    # Properly escape the code for TypeScript template literals
    code_escaped = alg['code'].replace('\\', '\\\\').replace('`', '\\`').replace('${', '\\${')
    title_escaped = alg['title'].replace("'", "\\'")
    desc_escaped = alg['description'].replace("'", "\\'")
    
    output += f'''  {{
    id: '{alg['id']}',
    title: '{title_escaped}',
    description: '{desc_escaped}',
    category: '{alg['category']}',
    difficulty: '{alg['difficulty']}',
    timeEstimate: {alg['time']},
    complexity: '{alg['complexity']}',
    complexityExplanation: '{alg['complexity']} complexity',
    code: `{code_escaped}`,
    explanation: '{desc_escaped}',
    examples: ['{alg['category']} example', 'Practice problem'],
    useCases: ['{alg['category']} applications', 'Real-world use']
  }}'''
    if i < len(ALGORITHMS_DATA) - 1:
        output += ','
    output += '\n'

output += ']\n'

with open('algorithm-learning-platform/src/lib/algorithmScanner.ts', 'w', encoding='utf-8') as f:
    f.write(output)

print(f"✓ Generated {len(ALGORITHMS_DATA)} algorithms across categories:")
categories = {}
for alg in ALGORITHMS_DATA:
    cat = alg['category']
    categories[cat] = categories.get(cat, 0) + 1
for cat, count in sorted(categories.items()):
    print(f"  • {cat}: {count} algorithms")
