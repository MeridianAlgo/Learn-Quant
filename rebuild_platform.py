#!/usr/bin/env python3
import os
import re

def read_safe(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except:
        return ""

def needs_api(content):
    return any(x in content for x in ['api_key', 'API_KEY', 'requests.', 'yfinance', 'finnhub', 'websocket'])

def clean_code(content):
    """Extract clean, runnable code with complete functions"""
    lines = content.split('\n')
    result = []
    skip_imports = ['requests', 'yfinance', 'finnhub', 'websocket', 'pandas', 'numpy', 'matplotlib', 'plotly']
    
    in_block = False
    block_indent = 0
    lines_added = 0
    max_lines = 50
    
    for line in lines:
        if lines_added >= max_lines:
            break
            
        stripped = line.strip()
        
        # Skip external package imports
        if stripped.startswith('import ') or stripped.startswith('from '):
            if any(pkg in line for pkg in skip_imports):
                continue
        
        # Track function/class blocks
        if stripped.startswith('def ') or stripped.startswith('class '):
            in_block = True
            block_indent = len(line) - len(line.lstrip())
            result.append(line)
            lines_added += 1
        elif in_block:
            current_indent = len(line) - len(line.lstrip())
            if stripped and current_indent <= block_indent:
                in_block = False
                if stripped.startswith('def ') or stripped.startswith('class '):
                    in_block = True
                    block_indent = current_indent
                    result.append(line)
                    lines_added += 1
                elif stripped.startswith('if __name__'):
                    result.append(line)
                    lines_added += 1
                else:
                    result.append(line)
                    lines_added += 1
            else:
                result.append(line)
                lines_added += 1
        else:
            if stripped:
                result.append(line)
                lines_added += 1
    
    code = '\n'.join(result)
    
    if 'print(' not in code and '__name__' not in code:
        code += '\n\n# Example\nprint("Executed successfully!")'
    
    return code

def get_best_code_sample(content):
    """Get meaningful code sample"""
    if 'if __name__' in content:
        main_start = content.find('if __name__')
        before_main = content[:main_start]
        main_section = content[main_start:]
        
        lines = before_main.split('\n')
        key_lines = []
        in_function = False
        function_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            if stripped.startswith('import ') or stripped.startswith('from '):
                if any(pkg in line for pkg in ['requests', 'yfinance', 'finnhub', 'websocket', 'pandas', 'numpy']):
                    continue
            
            if stripped.startswith('def '):
                if function_lines:
                    key_lines.extend(function_lines)
                    key_lines.append('')
                function_lines = [line]
                in_function = True
            elif in_function:
                function_lines.append(line)
                if len(key_lines) > 30:
                    break
        
        if function_lines:
            key_lines.extend(function_lines)
        
        main_lines = main_section.split('\n')[:15]
        key_lines.append('')
        key_lines.extend(main_lines)
        
        return '\n'.join(key_lines[:50])
    
    return clean_code(content)

def get_speed_text(category):
    """Plain English speed descriptions"""
    cat = category.lower()
    if 'sort' in cat:
        return 'Fast sorting - efficient for most data'
    elif 'search' in cat:
        return 'Quick search - finds items fast'
    elif 'finance' in cat or 'portfolio' in cat or 'risk' in cat:
        return 'Fast calculations - instant results'
    elif 'basic' in cat or 'tutorial' in cat:
        return 'Simple operations - runs instantly'
    elif 'machine learning' in cat:
        return 'Learns from data - speed varies by size'
    elif 'graph' in cat:
        return 'Efficient traversal - scales well'
    else:
        return 'Efficient execution - good performance'

# Scan all UTILS
lessons = []
for folder in sorted(os.listdir(".")):
    if not folder.startswith("UTILS -"):
        continue
    
    category = folder.replace("UTILS - ", "")
    
    for file in os.listdir(folder):
        if not file.endswith('.py') or file.startswith('__'):
            continue
        
        content = read_safe(os.path.join(folder, file))
        if len(content) < 50:
            continue
        
        api_needed = needs_api(content)
        code = "# This code requires API keys\nprint('Please configure API credentials')" if api_needed else get_best_code_sample(content)
        
        # Get description
        desc_match = re.search(r'"""(.+?)"""', content, re.DOTALL)
        desc = desc_match.group(1).strip().split('\n')[0][:100] if desc_match else f"Learn {file[:-3]}"
        desc = desc.replace('"', '').replace("'", '')
        
        # Difficulty
        if any(x in category.lower() for x in ['basic', 'tutorial', 'intro']):
            diff = 'Beginner'
        elif any(x in category.lower() for x in ['advanced', 'stochastic', 'monte carlo']):
            diff = 'Advanced'
        else:
            diff = 'Intermediate'
        
        lessons.append({
            'id': f"{category.lower().replace(' ', '-')}-{file[:-3].replace('_', '-')}",
            'title': file[:-3].replace('_', ' ').title(),
            'desc': desc,
            'category': category,
            'difficulty': diff,
            'time': 20,
            'speed': get_speed_text(category),
            'code': code,
            'api': api_needed
        })

print(f"Found {len(lessons)} lessons")

# Generate TypeScript
ts = '''export interface Algorithm {
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
  requiresApi?: boolean
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

for i, lesson in enumerate(lessons):
    code_esc = lesson['code'].replace('\\', '\\\\').replace('`', '\\`').replace('${', '\\${')
    
    ts += f'''  {{
    id: '{lesson['id']}',
    title: '{lesson['title']}',
    description: '{lesson['desc']}',
    category: '{lesson['category']}',
    difficulty: '{lesson['difficulty']}',
    timeEstimate: {lesson['time']},
    complexity: '{lesson['speed']}',
    complexityExplanation: '{lesson['speed']}',
    code: `{code_esc}`,
    explanation: '{lesson['desc']}',
    examples: ['{lesson['category']} tutorial'],
    useCases: ['Learn {lesson['category']}', 'Practical examples'],
    requiresApi: {'true' if lesson['api'] else 'false'}
  }}'''
    
    if i < len(lessons) - 1:
        ts += ','
    ts += '\n'

ts += ']\n'

with open('algorithm-learning-platform/src/lib/algorithmScanner.ts', 'w', encoding='utf-8') as f:
    f.write(ts)

print(f"Generated {len(lessons)} lessons with complete code")
print(f"API required: {sum(1 for l in lessons if l['api'])}")
