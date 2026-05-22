import os
import shutil


def extract_description(readme_path):
    """Extract title and first prose description from README.md"""
    try:
        with open(readme_path, encoding="utf-8") as f:
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


def categorize(name):
    if "Python Basics" in name:
        return "Python Fundamentals"
    if "Data Structures" in name:
        return "Data Structures"
    if "Algorithms" in name:
        return "Algorithms"
    if "Advanced Python" in name:
        return "Advanced Python"
    if "Quantitative Methods" in name:
        return "Quantitative Methods"
    if any(
        x in name
        for x in [
            "Black-Scholes",
            "Advanced Options",
            "Options Pricing",
            "Options Chain",
            "Greeks Calculator",
            "Exotic Options",
            "Options Strategies",
            "Technical Indicators",
            "Monte Carlo Simulation",
            "Bond Price",
            "Duration Convexity",
            "Yield Curve",
            "Discounted Cash Flow",
            "CAPM",
            "Beta Calculator",
            "Correlation Analysis",
            "Covariance Estimation",
            "Credit Risk",
            "Volatility Calculator",
            "Kelly Criterion",
            "Position Sizing",
            "Transaction Cost",
            "FX Tools",
            "Expected Shortfall",
            "Dividend Tracker",
        ]
    ):
        return "Options, Derivatives & Finance"
    if any(
        x in name
        for x in [
            "Risk Metrics",
            "Value at Risk",
            "Sharpe",
            "Performance Attribution",
        ]
    ):
        return "Risk & Performance"
    if any(
        x in name
        for x in [
            "Portfolio",
            "Monte Carlo Portfolio",
        ]
    ):
        return "Portfolio Management"
    if any(
        x in name
        for x in [
            "Strategies",
            "Order Execution",
        ]
    ):
        return "Strategies"
    if any(
        x in name
        for x in [
            "Machine Learning",
            "Reinforcement Learning",
            "AI Development",
            "Sentiment Analysis",
            "Learning Platform",
        ]
    ):
        return "AI & Machine Learning"
    if any(
        x in name
        for x in [
            "Market Microstructure",
            "High Frequency",
        ]
    ):
        return "Market Microstructure"
    if any(
        x in name
        for x in [
            "Market Data",
            "Historical Data",
            "News Fetching",
            "Websocket",
            "Core Utilities",
            "Data Processing",
            "Logging",
            "System Utilities",
            "Currency Converter",
            "Economic Calendar",
        ]
    ):
        return "Utilities & Tools"
    return "Other"


CATEGORY_ORDER = [
    "Python Fundamentals",
    "Data Structures",
    "Algorithms",
    "Advanced Python",
    "Quantitative Methods",
    "Options, Derivatives & Finance",
    "Risk & Performance",
    "Portfolio Management",
    "Strategies",
    "AI & Machine Learning",
    "Market Microstructure",
    "Utilities & Tools",
    "Other",
]


def build_docs():
    docs_dir = "docs"
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)

    if os.path.exists("README.md"):
        shutil.copy("README.md", os.path.join(docs_dir, "index.md"))

    modules = []
    skip = {".git", ".github", ".vscode", ".claude", "docs", ".pytest_cache", "tests", ".ruff_cache"}
    dirs = [d for d in os.listdir(".") if os.path.isdir(d) and d not in skip]

    for d in sorted(dirs):
        readme_path = os.path.join(d, "README.md")
        if os.path.exists(readme_path):
            title, description = extract_description(readme_path)
            target_file = f"{d}.md"
            shutil.copy(readme_path, os.path.join(docs_dir, target_file))
            modules.append(
                {
                    "dir": d,
                    "title": title,
                    "desc": description,
                    "file": target_file,
                    "category": categorize(d),
                }
            )

    categories = {cat: [] for cat in CATEGORY_ORDER}
    for mod in modules:
        categories[mod["category"]].append(mod)

    # Group the 13 fine-grained categories into 5 top-level tabs
    TAB_GROUPS = {
        "Home": [],
        "Foundations": ["Python Fundamentals", "Data Structures", "Algorithms", "Advanced Python"],
        "Quant Methods": ["Quantitative Methods"],
        "Finance & Options": [
            "Options, Derivatives & Finance",
            "Risk & Performance",
            "Portfolio Management",
            "Strategies",
        ],
        "AI & Tools": ["AI & Machine Learning", "Market Microstructure", "Utilities & Tools", "Other"],
    }

    nav_items = ["  - Home:", "      - Overview: index.md", "      - All Modules: modules.md"]

    for tab, cats in TAB_GROUPS.items():
        if tab == "Home":
            continue
        nav_items.append(f"  - {tab}:")
        for cat in cats:
            mods = categories.get(cat, [])
            if not mods:
                continue
            nav_items.append(f"      - {cat}:")
            for mod in mods:
                nav_items.append(f'          - "{mod["dir"]}": {mod["file"]}')

    modules_content = "# All Modules\n\nComplete index of all Learn-Quant lessons and utilities.\n\n"
    for cat in CATEGORY_ORDER:
        mods = categories[cat]
        if mods:
            modules_content += f"## {cat}\n\n"
            for mod in mods:
                modules_content += f"### [{mod['dir']}]({mod['file']})\n"
                if mod["title"]:
                    modules_content += f"**{mod['title']}**\n\n"
                if mod["desc"]:
                    modules_content += f"{mod['desc']}\n\n"

    with open(os.path.join(docs_dir, "modules.md"), "w", encoding="utf-8") as f:
        f.write(modules_content)

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
    - navigation.indexes
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

    total = sum(len(v) for v in categories.values())
    print(f"Docs built: {total} modules indexed across {sum(1 for v in categories.values() if v)} categories")
    for cat in CATEGORY_ORDER:
        if categories[cat]:
            print(f"  {cat}: {len(categories[cat])} modules")


if __name__ == "__main__":
    build_docs()
