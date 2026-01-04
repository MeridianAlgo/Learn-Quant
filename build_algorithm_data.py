#!/usr/bin/env python3
"""Build comprehensive algorithm data file"""

# Template for each algorithm
TEMPLATE = '''  {{
    id: '{id}',
    title: '{title}',
    description: '{description}',
    category: '{category}',
    difficulty: '{difficulty}',
    timeEstimate: {time},
    complexity: '{complexity}',
    complexityExplanation: '{complexity}',
    code: `{code}`,
    explanation: '{explanation}',
    examples: {examples},
    useCases: {use_cases}
  }}'''

# Generate code samples
def get_code(alg_type, name):
    codes = {
        'insertion-sort': '''def insertion_sort(arr):
    result = arr.copy()
    for i in range(1, len(result)):
        key = result[i]
        j = i - 1
        while j >= 0 and result[j] > key:
            result[j + 1] = result[j]
            j -= 1
        result[j + 1] = key
    return result

data = [12, 11, 13, 5, 6]
print("Sorted:", insertion_sort(data))''',
        
        'selection-sort': '''def selection_sort(arr):
    result = arr.copy()
    for i in range(len(result)):
        min_idx = i
        for j in range(i + 1, len(result)):
            if result[j] < result[min_idx]:
                min_idx = j
        result[i], result[min_idx] = result[min_idx], result[i]
    return result

data = [64, 25, 12, 22, 11]
print("Sorted:", selection_sort(data))''',
        
        'dfs': '''def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    print(start, end=' ')
    
    for neighbor in graph.get(start, []):
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    return visited

graph = {'A': ['B', 'C'], 'B': ['D', 'E'], 'C': ['F'], 'D': [], 'E': ['F'], 'F': []}
print("DFS traversal:")
dfs(graph, 'A')''',
        
        'stack': '''class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        return self.items.pop() if not self.is_empty() else None
    
    def peek(self):
        return self.items[-1] if not self.is_empty() else None
    
    def is_empty(self):
        return len(self.items) == 0

stack = Stack()
stack.push(1)
stack.push(2)
print("Top:", stack.peek())
print("Pop:", stack.pop())'''
    }
    return codes.get(alg_type, f'# {name} implementation\nprint("Algorithm: {name}")')

# Write the file
output = '''// Comprehensive algorithm collection for learning platform

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

# Add sample algorithms
algorithms = [
    ('insertion-sort', 'Insertion Sort', 'Builds sorted array one item at a time', 'Sorting', 'Beginner', 15, 'O(n²)', 'Efficient for small datasets', ['Small arrays', 'Nearly sorted'], ['Small data', 'Online sorting']),
    ('selection-sort', 'Selection Sort', 'Finds minimum and places at beginning', 'Sorting', 'Beginner', 15, 'O(n²)', 'Simple but inefficient', ['Teaching', 'Small data'], ['Educational', 'Simple sorting']),
    ('dfs', 'Depth-First Search', 'Explores as far as possible along branches', 'Graph', 'Beginner', 20, 'O(V + E)', 'Explores depth first', ['Maze solving', 'Path finding'], ['Graph traversal', 'Connectivity']),
    ('stack', 'Stack', 'LIFO data structure', 'Data Structures', 'Beginner', 10, 'O(1)', 'Last in first out', ['Undo operations', 'Expression evaluation'], ['Function calls', 'Backtracking']),
]

for alg in algorithms:
    code = get_code(alg[0], alg[1])
    output += TEMPLATE.format(
        id=alg[0],
        title=alg[1],
        description=alg[2],
        category=alg[3],
        difficulty=alg[4],
        time=alg[5],
        complexity=alg[6],
        explanation=alg[7],
        code=code,
        examples=str(alg[8]),
        use_cases=str(alg[9])
    ) + ',\n'

output += ']\n'

with open('algorithm-learning-platform/src/lib/algorithmScanner.ts', 'w', encoding='utf-8') as f:
    f.write(output)

print("✓ Generated algorithm scanner with", len(algorithms), "algorithms")
