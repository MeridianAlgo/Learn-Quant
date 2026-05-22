import os
import shutil


def extract_description(readme_path):
    """Extract title and description from README.md"""
    try:
        with open(readme_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        title = ""
        description = ""
        for i, line in enumerate(lines):
            if line.startswith("# ") and not title:
                title = line.replace("# ", "").strip()
            elif title and i > 0 and line.strip() and not line.startswith("#"):
                description = line.strip()
                break
        return title, description
    except Exception:
        return "", ""


def build_docs():
    docs_dir = "docs"
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)

    # Copy root README as index
    if os.path.exists("README.md"):
        shutil.copy("README.md", os.path.join(docs_dir, "index.md"))

    # Collect all modules with their descriptions
    modules = []
    dirs = [
        d
        for d in os.listdir(".")
        if os.path.isdir(d)
        and d not in {".git", ".github", ".vscode", ".claude", "docs", ".pytest_cache", "tests"}
    ]

    for d in sorted(dirs):
        readme_path = os.path.join(d, "README.md")
        if os.path.exists(readme_path):
            title, description = extract_description(readme_path)
            target_file = f"{d}.md"
            target_path = os.path.join(docs_dir, target_file)
            shutil.copy(readme_path, target_path)
            modules.append({"dir": d, "title": title, "desc": description, "file": target_file})

    # Categorize modules
    categories = {
        "Python Fundamentals": [],
        "Advanced Python": [],
        "Algorithms": [],
        "Quantitative Finance": [],
        "Data Structures": [],
        "AI & Machine Learning": [],
        "Utilities": [],
        "Other": [],
    }

    for mod in modules:
        name = mod["dir"]
        if "Python Basics" in name:
            categories["Python Fundamentals"].append(mod)
        elif "Advanced Python" in name:
            categories["Advanced Python"].append(mod)
        elif "Algorithms" in name:
            categories["Algorithms"].append(mod)
        elif any(x in name for x in ["CAPM", "Black-Scholes", "Bond", "DCF", "Options", "Portfolio"]):
            categories["Quantitative Finance"].append(mod)
        elif "Data Structures" in name:
            categories["Data Structures"].append(mod)
        elif any(x in name for x in ["AI", "Machine Learning", "Neural", "Reinforcement"]):
            categories["AI & Machine Learning"].append(mod)
        elif any(x in name for x in ["Core Utilities", "UTILS", "Converter", "Tracker", "Calendar"]):
            categories["Utilities"].append(mod)
        else:
            categories["Other"].append(mod)

    # Build nav items for mkdocs
    nav_items = ["  - Home: index.md"]
    all_modules_page = "  - Modules: modules.md"
    nav_items.append(all_modules_page)

    for category, mods in categories.items():
        if mods:
            nav_items.append(f"  - {category}:")
            for mod in mods:
                nav_items.append(f"      - {mod['dir']}: {mod['file']}")

    # Create comprehensive modules page
    modules_content = "# All Modules\n\n"
    modules_content += "Complete index of all Learn-Quant lessons and utilities.\n\n"

    for category, mods in categories.items():
        if mods:
            modules_content += f"## {category}\n\n"
            for mod in mods:
                modules_content += f"### [{mod['dir']}]({mod['file']})\n"
                if mod["title"]:
                    modules_content += f"**{mod['title']}**\n\n"
                if mod["desc"]:
                    modules_content += f"{mod['desc']}\n\n"

    with open(os.path.join(docs_dir, "modules.md"), "w", encoding="utf-8") as f:
        f.write(modules_content)

    # Generate mkdocs.yml
    mkdocs_yaml = f"""site_name: Learn-Quant
site_description: Master quantitative finance, algorithmic trading, and professional Python engineering.
repo_url: https://github.com/MeridianAlgo/Learn-Quant
repo_name: MeridianAlgo/Learn-Quant

theme:
  name: material
  features:
    - navigation.tabs
    - navigation.sections
    - navigation.expand
    - navigation.top
    - search.suggest
    - search.highlight
    - search.share
    - content.code.copy
    - content.code.annotate
    - content.tooltips
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
  font:
    text: Roboto
    code: Roboto Mono

docs_dir: docs

markdown_extensions:
  - admonition
  - pymdownx.details
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format
  - pymdownx.highlight:
      anchor_linenums: true
      line_spans: __span
      pygments_lang_class: true
  - pymdownx.inlinehilite
  - pymdownx.snippets
  - pymdownx.tabbed:
      alternate_style: true
  - tables
  - attr_list

plugins:
  - search

nav:
{chr(10).join(nav_items)}
"""
    with open("mkdocs.yml", "w", encoding="utf-8") as f:
        f.write(mkdocs_yaml)

    print("✓ Documentation structure built with all modules indexed and categorized")


if __name__ == "__main__":
    build_docs()
