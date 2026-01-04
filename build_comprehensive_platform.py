#!/usr/bin/env python3
"""Build comprehensive learning platform from all UTILS"""
import os
import re

def read_file_safe(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except:
        return ""

def extract_code_sample(content, max_lines=30):
    """Extract a meaningful code sample"""
    lines = content.split('\n')
    
    # Find first function or class
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('def ') or line.strip().startswith('class '):
            start_idx = i
            break
    
    # Get up to max_lines
    sample_lines = lines[start_idx:start_idx + max_lines]
    
    # Add example usage if exists
    if '__name__' in content:
        example_start = content.find('if __name__')
        if example_start > 0:
            example_lines = content[example_start:].split('\n')[:10]
            sample_lines.extend(['', '# Example usage:'] + example_lines)
    
    return '\n'.join(sample_lines[:max_lines])

def categorize_difficulty(category, file_name):
    """Determine difficulty based on category and content"""
    beginner_keywords = ['basics', 'tutorial', 'intro', 'simple', 'control flow', 'numbers', 'strings']
    advanced_keywords = ['advanced', 'stochastic', 'optimization', 'machine learning', 'monte carlo']
    
    cat_lower = category.lower()
    file_lower = file_name.lower()
    
    if any(k in cat_lower or k in file_lower for k in beginner_keywords):
        return 'Beginner'
    elif any(k in cat_lower or k in file_lower for k in advanced_keywords):
        return 'Advanced'
    else:
        return 'Intermediate'

def estimate_time(content):
    """Estimate learning time based on content length"""
    lines = len(content.split('\n'))
    if lines < 50:
        return 10
    elif lines < 150:
        return 20
    elif lines < 300:
        return 30
    else:
        return 45

def get_complexity_info(category):
    """Get complexity based on category"""
    complexity_map = {
        'Sorting': ('O(n log n)', 'Fast sorting - efficient for most data'),
        'Searching': ('O(log n)', 'Quick search - very efficient'),
        'Graph': ('O(V + E)', 'Linear in graph size - scales well'),
        'Dynamic Programming': ('O(n²)', 'Polynomial time - good for optimization'),
        'Data Structures': ('O(1) to O(n)', 'Depends on operation'),
        'Machine Learning': ('O(n × features)', 'Depends on data size'),
        'Finance': ('O(n)', 'Linear time - fast calculations'),
        'Statistics': ('O(n)', 'Single pass through data'),
        'Python Basics': ('O(1)', 'Constant time - instant'),
    }
    
    for key, value in complexity_map.items():
        if key.lower() in category.lower():
            return value
    
    return ('O(n)', 'Linear time - efficient')

def scan_all_utils():
    """Scan all UTILS folders"""
    algorithms = []
    base_dir = "."
    
    for folder in sorted(os.listdir(base_dir)):
        if not folder.startswith("UTILS -"):
            continue
            
        category = folder.replace("UTILS - ", "")
        folder_path = os.path.join(base_dir, folder)
        
        for file in os.listdir(folder_path):
            if not file.endswith('.py') or file.startswith('__'):
                continue
                
            file_path = os.path.join(folder_path, file)
            content = read_file_safe(file_path)
            
            if len(content) < 100:  # Skip tiny files
                continue
            
            # Extract info
            title = file.replace('.py', '').replace('_', ' ').title()
            code_sample = extract_code_sample(content)
            difficulty = categorize_difficulty(category, file)
            time_est = estimate_time(content)
            complexity, complexity_exp = get_complexity_info(category)
            
            # Create description
            desc_match = re.search(r'"""(.+?)"""', content, re.DOTALL)
            if desc_match:
                description = desc_match.group(1).strip().split('\n')[0][:100]
            else:
                description = f"Learn {title.lower()} with practical examples"
            
            # Get use cases from comments
            use_cases = []
            for line in content.split('\n')[:50]:
                if 'use case' in line.lower() or 'example' in line.lower():
                    use_cases.append(line.strip('# ').strip())
            
            if not use_cases:
                use_cases = [f"{category} applications", "Real-world problems", "Practice exercises"]
            
            algo_id = f"{category.lower().replace(' ', '-')}-{file.replace('.py', '').replace('_', '-')}"
            
            algorithms.append({
                'id': algo_id,
                'title': title,
                'description': description,
                'category': category,
                'difficulty': difficulty,
                'time': time_est,
                'complexity': complexity,
                'complexity_exp': complexity_exp,
                'code': code_sample,
                'use_cases': use_cases[:3]
            })
    
    return algorithms

def generate_typescript(algorithms):
    """Generate TypeScript file"""
    output = '''// Comprehensive Learning Platform - Auto-generated from UTILS
// Includes algorithms, finance tools, Python basics, and more

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
    
    for i, alg in enumerate(algorithms):
        # Escape for TypeScript - handle all special characters
        code_escaped = (alg['code']
            .replace('\\', '\\\\')
            .replace('`', '\\`')
            .replace('${', '\\${')
            .replace('\r', '')
        )
        
        # Clean title and description - remove all problematic chars
        title_escaped = alg['title'].replace("'", "").replace('"', '').replace('\n', ' ')
        desc_escaped = (alg['description']
            .replace("'", "")
            .replace('"', '')
            .replace('\n', ' ')
            .replace('\r', '')
        )
        
        # Simplify use cases to avoid escaping issues
        use_cases_simple = [
            f"Learn {alg['category']}",
            "Practical examples",
            "Real-world applications"
        ]
        
        output += f'''  {{
    id: '{alg['id']}',
    title: '{title_escaped}',
    description: '{desc_escaped}',
    category: '{alg['category']}',
    difficulty: '{alg['difficulty']}',
    timeEstimate: {alg['time']},
    complexity: '{alg['complexity']}',
    complexityExplanation: '{alg['complexity_exp']}',
    code: `{code_escaped}`,
    explanation: '{desc_escaped}',
    examples: ['{alg['category']} tutorial', 'Interactive lesson'],
    useCases: ['{use_cases_simple[0]}', '{use_cases_simple[1]}', '{use_cases_simple[2]}']
  }}'''
        
        if i < len(algorithms) - 1:
            output += ','
        output += '\n'
    
    output += ']\n'
    return output

if __name__ == "__main__":
    print("Scanning all UTILS folders...")
    algorithms = scan_all_utils()
    
    print(f"\n✓ Found {len(algorithms)} lessons")
    
    # Group by category
    categories = {}
    for alg in algorithms:
        cat = alg['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\n📚 Categories ({len(categories)}):")
    for cat, count in sorted(categories.items())[:20]:
        print(f"  • {cat}: {count} lessons")
    
    if len(categories) > 20:
        print(f"  ... and {len(categories) - 20} more categories")
    
    # Generate TypeScript
    print("\n📝 Generating TypeScript file...")
    ts_content = generate_typescript(algorithms)
    
    output_path = 'algorithm-learning-platform/src/lib/algorithmScanner.ts'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(ts_content)
    
    print(f"✓ Generated {output_path}")
    print(f"\n🎉 Platform ready with {len(algorithms)} lessons!")
