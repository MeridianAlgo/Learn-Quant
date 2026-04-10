import os
import shutil


def build_docs():
    docs_dir = 'docs'
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)

    if os.path.exists('README.md'):
        shutil.copy('README.md', os.path.join(docs_dir, 'index.md'))

    nav_items = ["  - Home: index.md"]

    dirs = [d for d in os.listdir('.') if os.path.isdir(d) and d.startswith('UTILS')]
    dirs.sort()

    for d in dirs:
        readme_path = os.path.join(d, 'README.md')
        if os.path.exists(readme_path):
            target_file = f"{d}.md"
            target_path = os.path.join(docs_dir, target_file)
            shutil.copy(readme_path, target_path)
            nav_items.append(f"  - '{d}': {target_file}")

    # Significantly Enhanced mkdocs.yml Configuration
    mkdocs_yaml = f"""site_name: Learn Quant
site_description: An advanced resource for learning quantitative finance, algorithmic strategies, and theoretical machine learning mechanisms.
theme:
  name: material
  features:
    - navigation.tabs
    - navigation.sections
    - navigation.top
    - search.suggest
    - search.highlight
    - search.share
    - content.code.copy
    - content.code.annotate
  palette:
    - scheme: default
      primary: indigo
      accent: deep purple
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - scheme: slate
      primary: indigo
      accent: deep purple
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
docs_dir: docs

markdown_extensions:
  - admonition
  - pymdownx.details
  - pymdownx.superfences
  - pymdownx.highlight:
      anchor_linenums: true
      line_spans: __span
      pygments_lang_class: true
  - pymdownx.inlinehilite
  - pymdownx.snippets
  - pymdownx.tabbed:
      alternate_style: true

plugins:
  - search

nav:
{chr(10).join(nav_items)}
"""
    with open('mkdocs.yml', 'w', encoding='utf-8') as f:
        f.write(mkdocs_yaml)

    print("Documentation structure built. State-of-the-art Material MkDocs ready.")

if __name__ == '__main__':
    build_docs()
