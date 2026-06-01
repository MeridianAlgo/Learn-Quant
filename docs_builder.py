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
            "Information Ratio",
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
    docs_dir = "z_docs"
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)

    if os.path.exists("README.md"):
        shutil.copy("README.md", os.path.join(docs_dir, "index.md"))

    modules = []
    skip = {".git", ".github", ".vscode", ".claude", "z_docs", ".pytest_cache", "z_tests", ".ruff_cache"}
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

    def entries(cat, indent):
        pad = "  " * indent
        return [f'{pad}- "{m["dir"]}": {m["file"]}' for m in categories.get(cat, [])]

    nav_items = [
        "  - Home: index.md",
        "  - All Modules: modules.md",
        # ── Python Fundamentals ──────────────────────────────────────
        "  - Python Fundamentals:",
        *[f"      {e}" for e in entries("Python Fundamentals", 0)],
        # ── Python Advanced ──────────────────────────────────────────
        "  - Python Advanced:",
        "      - Advanced Python:",
        *[f"          {e}" for e in entries("Advanced Python", 0)],
        "      - Data Structures:",
        *[f"          {e}" for e in entries("Data Structures", 0)],
        "      - Algorithms:",
        *[f"          {e}" for e in entries("Algorithms", 0)],
        # ── Quant ────────────────────────────────────────────────────
        "  - Quant:",
        "      - Quantitative Methods:",
        *[f"          {e}" for e in entries("Quantitative Methods", 0)],
        "      - Finance & Options:",
        *[f"          {e}" for e in entries("Options, Derivatives & Finance", 0)],
        "      - Risk & Performance:",
        *[f"          {e}" for e in entries("Risk & Performance", 0)],
        "      - Portfolio Management:",
        *[f"          {e}" for e in entries("Portfolio Management", 0)],
        "      - Strategies:",
        *[f"          {e}" for e in entries("Strategies", 0)],
        # ── AI & Machine Learning ────────────────────────────────────
        "  - AI & Machine Learning:",
        *[f"      {e}" for e in entries("AI & Machine Learning", 0)],
        # ── Utilities & Tools ────────────────────────────────────────
        "  - Utilities & Tools:",
        *[f"      {e}" for e in entries("Market Microstructure", 0)],
        *[f"      {e}" for e in entries("Utilities & Tools", 0)],
        *[f"      {e}" for e in entries("Other", 0)],
    ]

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

docs_dir: z_docs

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
