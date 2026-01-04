#!/usr/bin/env python3
import os
import re

def read_file_safe(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except:
        return ""

def check_requires_api(content):
    api_indicators = ['api_key', 'API_KEY', 'requests.get', 'yfinance', 'finnhub', 'websocket']
    return any(indicator in content for indicator in api_indicators)

def fix_python_code(content, max_lines=25):
    lines = content.split('\n')
    result_lines = []
    
    for line in lines[:max_lines]:
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            if any(pkg in stripped for pkg in ['requests', 'yfinance', 'finnhub', 'websocket', 'pandas', 'numpy']):
                continue
        result_lines.append(line)
        if len(result_lines) > 20:
            break
    
    code_str = '\n'.join(result_lines)
    if 'print(' not in code_str:
        result_lines.append('\nprint("Code executed successfully!")')
    
    return '\n'.join(result_lines)

def get_plain_english_speed(category):
    if 'sort' in category.lower():
        return 'Fast sorting - works well with most data'
    elif 'search' in category.lower():
        return 'Quick search - finds items efficiently'
    elif 'finance' in category.lower() or 'portfolio' in category.lower():
        return 'Fast calculations - instant results'
    elif 'basic' in category.lower():
        return 'Simple operations - instant execution'
    else:
        return 'Efficient execution - good performance'

algorithms = []
for folder in sorted(os.listdir(".")):
    if not folder.startswith("UTILS -"):
        continue
    category = folder.replace("UTILS - ", "")
    for file in os.listdir(folder):
        if file.endswith('.py') and not file.startswith('__'):
            content = read_file_safe(os.path.join(folder, file))
            if len(content) < 100:
                continue
            
            requires_api = check_requires_api(content)
            code = "# Requires API key\nprint('API needed')" if requires_api else fix_python_code(content)
            
            algorithms.append({
                'id': f"{category.lower().replace(' ', '-')}-{file[:-3]}",
                'title': file[:-3].replace('_', ' ').title(),
                'category': category,
                'speed': get_plain_english_speed(category),
                'code': code,
                'api': requires_api
            })

print(f"Found {len(algorithms)} lessons")
