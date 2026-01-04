#!/usr/bin/env python3
"""Scan all UTILS folders and extract Python code"""
import os
import re

def scan_utils():
    utils = []
    base_dir = "."
    
    for folder in os.listdir(base_dir):
        if folder.startswith("UTILS -"):
            category = folder.replace("UTILS - ", "")
            folder_path = os.path.join(base_dir, folder)
            
            # Find Python files
            for file in os.listdir(folder_path):
                if file.endswith('.py') and not file.startswith('__'):
                    file_path = os.path.join(folder_path, file)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                            # Extract functions and classes
                            functions = re.findall(r'^def\s+(\w+)\s*\(', content, re.MULTILINE)
                            classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
                            
                            if functions or classes:
                                utils.append({
                                    'category': category,
                                    'file': file,
                                    'path': file_path,
                                    'functions': functions[:5],  # First 5
                                    'classes': classes[:3],  # First 3
                                    'content': content[:2000]  # First 2000 chars
                                })
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
    
    return utils

if __name__ == "__main__":
    utils = scan_utils()
    print(f"Found {len(utils)} Python utility files")
    
    # Group by category
    categories = {}
    for util in utils:
        cat = util['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(util)
    
    print(f"\nCategories: {len(categories)}")
    for cat, items in sorted(categories.items()):
        print(f"  {cat}: {len(items)} files")
