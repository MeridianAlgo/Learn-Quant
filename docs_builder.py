"""Learn-Quant documentation builder.

Turns the repository of self-contained lesson folders into a polished
MkDocs Material site. For every module it:

* copies the module ``README.md`` into ``z_docs/``,
* augments the page with a metadata header (category, difficulty, language,
  "run this" command, GitHub source link) and a footer of related modules,

and on top of the module pages it hand-builds a rich landing page, a card-grid
module index, and several standalone guides (getting started, learning paths,
glossary, FAQ). Finally it regenerates ``mkdocs.yml`` so the navigation always
matches what is on disk.

Run it with ``python docs_builder.py``. The output is consumed by
``mkdocs build`` (see ``.github/workflows/pages.yml``).
"""

from __future__ import annotations

import os
import re
from urllib.parse import quote, unquote

REPO_URL = "https://github.com/MeridianAlgo/Learn-Quant"
DOCS_DIR = "z_docs"

# Directories that are not lesson modules.
SKIP_DIRS = {
    ".git",
    ".github",
    ".vscode",
    ".claude",
    "z_docs",
    ".pytest_cache",
    "z_tests",
    ".ruff_cache",
    "graphify-out",
    "__pycache__",
    "site",
}

# ---------------------------------------------------------------------------
# Category model: ordering, icon, one-line blurb and a default difficulty.
# ---------------------------------------------------------------------------
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

CATEGORY_META = {
    "Python Fundamentals": {
        "icon": ":material-language-python:",
        "blurb": "Core Python for financial analysis — start here if you are new to code.",
        "difficulty": "Beginner",
    },
    "Data Structures": {
        "icon": ":material-database-outline:",
        "blurb": "The right container for the job: arrays, lists, dicts, sets on market data.",
        "difficulty": "Beginner",
    },
    "Algorithms": {
        "icon": ":material-sitemap-outline:",
        "blurb": "Classic computer-science algorithms applied to price and order data.",
        "difficulty": "Intermediate",
    },
    "Advanced Python": {
        "icon": ":material-cog-outline:",
        "blurb": "Production engineering: async, OOP, concurrency, resilient error handling.",
        "difficulty": "Intermediate",
    },
    "Quantitative Methods": {
        "icon": ":material-function-variant:",
        "blurb": "The mathematics underpinning modern finance, implemented from first principles.",
        "difficulty": "Advanced",
    },
    "Options, Derivatives & Finance": {
        "icon": ":material-chart-bell-curve:",
        "blurb": "Pricing, Greeks, fixed income and valuation of financial instruments.",
        "difficulty": "Intermediate",
    },
    "Risk & Performance": {
        "icon": ":material-shield-alert-outline:",
        "blurb": "Measure what can go wrong and how well a strategy actually performed.",
        "difficulty": "Intermediate",
    },
    "Portfolio Management": {
        "icon": ":material-briefcase-outline:",
        "blurb": "Construct, optimise and rebalance multi-asset portfolios.",
        "difficulty": "Advanced",
    },
    "Strategies": {
        "icon": ":material-trending-up:",
        "blurb": "End-to-end trading strategies with signals, backtests and execution.",
        "difficulty": "Advanced",
    },
    "AI & Machine Learning": {
        "icon": ":material-robot-outline:",
        "blurb": "Data-driven models: random forests, deep learning, RL and NLP for markets.",
        "difficulty": "Advanced",
    },
    "Market Microstructure": {
        "icon": ":material-pulse:",
        "blurb": "Order books, spreads and the low-latency mechanics of how trades happen.",
        "difficulty": "Advanced",
    },
    "Utilities & Tools": {
        "icon": ":material-tools:",
        "blurb": "The plumbing: data ingestion, logging, FX, calendars and helpers.",
        "difficulty": "Beginner",
    },
    "Other": {
        "icon": ":material-shape-outline:",
        "blurb": "Additional modules.",
        "difficulty": "Intermediate",
    },
}

DIFFICULTY_CLASS = {
    "Beginner": "beginner",
    "Intermediate": "intermediate",
    "Advanced": "advanced",
}


# ---------------------------------------------------------------------------
# README parsing helpers.
# ---------------------------------------------------------------------------
def extract_description(readme_path):
    """Return ``(title, first_prose_paragraph)`` from a module README."""
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
            "Implied Volatility",
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
    if "Portfolio" in name or "Monte Carlo Portfolio" in name:
        return "Portfolio Management"
    if "Strategies" in name or "Order Execution" in name or "Backtest" in name:
        return "Strategies"
    if any(
        x in name
        for x in [
            "Machine Learning",
            "Reinforcement Learning",
            "AI Development",
            "Sentiment Analysis",
            "Feature Engineering",
            "Learning Platform",
        ]
    ):
        return "AI & Machine Learning"
    if "Market Microstructure" in name or "High Frequency" in name:
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
    # A generic "Finance - X" module we did not list explicitly still belongs
    # with the finance family rather than falling through to "Other".
    if name.startswith("Finance -"):
        return "Options, Derivatives & Finance"
    return "Other"


def detect_assets(module_dir):
    """Return ``(language, run_commands)`` for a module folder.

    ``run_commands`` is a list of shell snippets a learner can copy to run the
    module's primary script(s).
    """
    py_files, js_files = [], []
    for fname in sorted(os.listdir(module_dir)):
        lower = fname.lower()
        if lower.endswith(".py") and "__init__" not in lower:
            py_files.append(fname)
        elif lower.endswith(".js"):
            js_files.append(fname)

    def _pick_primary(files):
        # Prefer a tutorial entry point, otherwise the first file.
        tutorial = [f for f in files if "tutorial" in f.lower()]
        return (tutorial or files)[:1]

    commands = []
    if py_files:
        for f in _pick_primary(py_files):
            commands.append(f'python "{f}"')
    if js_files:
        for f in _pick_primary(js_files):
            commands.append(f'node "{f}"')

    if py_files and js_files:
        language = "Python · JavaScript"
    elif js_files:
        language = "JavaScript"
    elif py_files:
        language = "Python"
    else:
        language = "—"
    return language, commands


def rewrite_relative_links(body, module_dirs):
    """Turn GitHub-style ``../Module Folder/`` links into site ``Module.md`` links.

    Module READMEs cross-reference sibling lessons with relative folder links
    that work when browsing the repo on GitHub but 404 on the rendered site.
    This rewrites any such link that resolves to a known module folder so the
    "see also" navigation works in the docs too.
    """

    def repl(match):
        target = match.group(1).strip()
        # Strip a trailing README reference or slash, then url-decode.
        target = re.sub(r"/?(README\.md)?/?$", "", target)
        decoded = unquote(target)
        if decoded in module_dirs:
            return f"]({decoded}.md)"
        return match.group(0)

    return re.sub(r"\]\(\.\./([^)]+)\)", repl, body)


# ---------------------------------------------------------------------------
# Per-module page assembly.
# ---------------------------------------------------------------------------
def build_module_page(mod, siblings, module_dirs):
    """Return the augmented markdown for a single module page."""
    with open(os.path.join(mod["dir"], "README.md"), encoding="utf-8") as f:
        body = f.read()
    body = rewrite_relative_links(body, module_dirs)

    meta = CATEGORY_META[mod["category"]]
    diff = mod["difficulty"]
    diff_class = DIFFICULTY_CLASS.get(diff, "intermediate")
    source_url = f"{REPO_URL}/tree/main/{quote(mod['dir'])}"

    badges = (
        f'<span class="lq-badge lq-{diff_class}">{diff}</span>'
        f'<span class="lq-badge lq-cat">{mod["category"]}</span>'
        f'<span class="lq-badge lq-lang">{mod["language"]}</span>'
    )

    header = [f'<p class="lq-badges">{badges}</p>', ""]

    run_block = ""
    if mod["commands"]:
        joined = "\n".join([f'cd "{mod["dir"]}"'] + mod["commands"])
        run_block = f"\n```bash\n{joined}\n```\n"

    header.append('!!! tip "Run this module"')
    if run_block:
        for line in run_block.strip("\n").splitlines():
            header.append(f"    {line}")
    else:
        header.append("    This module is a reference/utility — import it from your own code.")
    header.append(f"\n    [:octicons-mark-github-16: View source on GitHub]({source_url})")
    header.append("")
    header.append("---")
    header.append("")

    # Footer: sibling modules in the same category for lateral navigation.
    related = [s for s in siblings if s["dir"] != mod["dir"]][:6]
    footer = ["", "", "---", "", f"## Continue in {mod['category']}", ""]
    if related:
        footer.append('<div class="grid cards" markdown>')
        footer.append("")
        for s in related:
            footer.append(f'-   {meta["icon"]} __[{s["dir"]}]({s["file"]})__')
            footer.append("")
            if s["desc"]:
                footer.append(f"    {s['desc']}")
                footer.append("")
        footer.append("</div>")
    footer.append("")
    footer.append(
        "[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } "
        "[:material-school-outline: Learning paths](learning-paths.md){ .md-button }"
    )
    footer.append("")

    return "\n".join(header) + body + "\n".join(footer)


# ---------------------------------------------------------------------------
# Standalone pages.
# ---------------------------------------------------------------------------
def build_index(stats, categories):
    cards = []
    for cat in CATEGORY_ORDER:
        mods = categories.get(cat, [])
        if not mods:
            continue
        meta = CATEGORY_META[cat]
        first = mods[0]["file"]
        cards.append(
            f'-   {meta["icon"]}{{ .lg .middle }} __{cat}__\n\n'
            f'    ---\n\n'
            f'    {meta["blurb"]}\n\n'
            f'    [:octicons-arrow-right-24: {len(mods)} modules]({first})\n'
        )
    cards_block = "<div class=\"grid cards\" markdown>\n\n" + "\n".join(cards) + "\n</div>"

    return f"""---
hide:
  - navigation
  - toc
---

<div class="lq-hero" markdown>

# Learn-Quant

### Master quantitative finance, algorithmic trading and professional Python — one runnable lesson at a time.

{stats["modules"]} self-contained modules · {stats["py"]} Python files · {stats["js"]} JavaScript modules · MIT licensed

[:material-rocket-launch-outline: Get started](getting-started.md){{ .md-button .md-button--primary }}
[:material-map-outline: Learning paths](learning-paths.md){{ .md-button }}
[:simple-github: Star on GitHub]({REPO_URL}){{ .md-button }}

</div>

## Why Learn-Quant?

<div class="grid cards" markdown>

-   :material-play-circle-outline:{{ .lg .middle }} __Every folder runs__

    ---

    No frameworks to install, no notebooks to wire up. Each module is a single
    folder you can run from the command line and read top to bottom.

-   :material-book-open-page-variant-outline:{{ .lg .middle }} __Theory *and* code__

    ---

    The maths is explained, then implemented from first principles — you see the
    formula and the line of code that makes it real.

-   :material-stairs:{{ .lg .middle }} __A real curriculum__

    ---

    Seven levels take you from `print("hello")` to GARCH volatility models,
    Black-Litterman allocation and reinforcement-learning agents.

-   :material-test-tube:{{ .lg .middle }} __Tested & linted__

    ---

    Modules ship with unit tests and pass `ruff` in CI, so the code you learn
    from is the code you can trust.

</div>

## Explore by topic

{cards_block}

## A guided path

```mermaid
flowchart LR
    A[Python Fundamentals] --> B[Data Structures & Algorithms]
    B --> C[Advanced Python]
    C --> D[Quantitative Methods]
    D --> E[Options & Finance]
    E --> F[Risk & Portfolio]
    F --> G[Strategies]
    G --> H[AI / ML & Microstructure]
```

New here? Follow the [recommended learning paths](learning-paths.md) — curated
sequences for beginners, options traders, quant researchers and ML engineers.

## Quick start

```bash
git clone {REPO_URL}
cd Learn-Quant
pip install -r requirements.txt

# run any module, e.g.
cd "Quantitative Methods - GARCH"
python garch.py
```

See the [Getting Started guide](getting-started.md) for the full setup, or jump
straight to the [complete module index](modules.md).
"""


def build_modules(categories):
    out = [
        "# All Modules",
        "",
        "Every Learn-Quant lesson, grouped by track. Each card links to the full"
        " write-up with runnable code and worked examples.",
        "",
    ]
    for cat in CATEGORY_ORDER:
        mods = categories.get(cat, [])
        if not mods:
            continue
        meta = CATEGORY_META[cat]
        out.append(f"## {meta['icon']} {cat}")
        out.append("")
        out.append(f"*{meta['blurb']}*")
        out.append("")
        out.append('<div class="grid cards" markdown>')
        out.append("")
        for m in mods:
            diff_class = DIFFICULTY_CLASS.get(m["difficulty"], "intermediate")
            out.append(f'-   __[{m["dir"]}]({m["file"]})__')
            out.append("")
            out.append(
                f'    <span class="lq-badge lq-{diff_class}">{m["difficulty"]}</span>'
                f'<span class="lq-badge lq-lang">{m["language"]}</span>'
            )
            out.append("")
            desc = m["desc"] or (m["title"] or m["dir"])
            out.append(f"    {desc}")
            out.append("")
        out.append("</div>")
        out.append("")
    return "\n".join(out)


def build_getting_started(stats):
    return f"""# Getting Started

Learn-Quant is a collection of **{stats["modules"]} self-contained modules**. There is no
package to install and no build step — you clone the repo and run whichever
lesson you want to learn from.

## 1. Install

```bash
git clone {REPO_URL}
cd Learn-Quant
python -m venv .venv
# Windows:  .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

A few JavaScript modules also need [Node.js](https://nodejs.org) (v18+).

!!! note "Requirements"
    The core stack is `numpy`, `pandas`, `scipy`, `scikit-learn` and
    `matplotlib`. Everything is pinned in `requirements.txt`; development tools
    (`ruff`, `pytest`) live in `requirements-dev.txt`.

## 2. Run a module

Every folder is independent. Change into it and run the main script:

```bash
cd "Black-Scholes Option Pricing"
python black_scholes.py
```

```bash
cd "Options Pricing - JavaScript"
node blackScholes.js
```

Modules whose file name ends in `_tutorial.py` are **interactive** — they walk
you through the concept with worked examples and quizzes.

## 3. How a module is laid out

```text
Quantitative Methods - GARCH/
├── README.md      ← the lesson: theory, formulas, usage, pitfalls
└── garch.py       ← the implementation with a runnable __main__ demo
```

Read the `README.md` for the *why*, then open the `.py` file for the *how*. The
two are written to be read side by side.

## 4. Suggested order

If you are working through the whole curriculum, follow the
[learning paths](learning-paths.md). In short:

1. **Python Fundamentals** → **Data Structures & Algorithms**
2. **Advanced Python** for production patterns
3. **Quantitative Methods** for the maths
4. **Options & Finance** → **Risk & Portfolio** → **Strategies**
5. **AI / ML** and **Market Microstructure** to specialise

## 5. Run the tests (optional)

```bash
pip install -r requirements-dev.txt
pytest z_tests -q
ruff check .
```

Ready? Head to the [module index](modules.md) or pick a
[learning path](learning-paths.md).
"""


def build_learning_paths():
    return """# Learning Paths

Pick the track that matches your goal. Each path is an ordered sequence of
modules — finish one before moving to the next.

## :material-school-outline: Complete Beginner → Quant Developer

The full curriculum, in order. Budget a few weeks and you will go from basic
syntax to pricing derivatives and backtesting strategies.

```mermaid
flowchart TD
    A[Python Basics - Numbers] --> B[Python Basics - Functions]
    B --> C[Python Basics - NumPy]
    C --> D[Python Basics - Pandas]
    D --> E[Data Structures - Arrays]
    E --> F[Algorithms - Sorting]
    F --> G[Advanced Python - OOP]
    G --> H[Quantitative Methods - Statistics]
    H --> I[Black-Scholes Option Pricing]
    I --> J[Risk Metrics]
    J --> K[Portfolio Optimizer]
    K --> L[Strategies - Pairs Trading]
```

## :material-chart-bell-curve: Options & Derivatives Trader

For traders who want to understand pricing and risk:

1. [Python Basics - NumPy](Python Basics - NumPy.md)
2. [Quantitative Methods - Stochastic Processes](Quantitative Methods - Stochastic Processes.md)
3. [Black-Scholes Option Pricing](Black-Scholes Option Pricing.md)
4. [Finance - Greeks Calculator](Finance - Greeks Calculator.md)
5. [Advanced Options Pricing](Advanced Options Pricing.md)
6. [Finance - Exotic Options](Finance - Exotic Options.md)
7. [Finance - Implied Volatility Surface](Finance - Implied Volatility Surface.md)
8. [Finance - Options Strategies](Finance - Options Strategies.md)

## :material-function-variant: Quant Researcher

For the statistically minded building signals and models:

1. [Quantitative Methods - Statistics](Quantitative Methods - Statistics.md)
2. [Quantitative Methods - Regression Analysis](Quantitative Methods - Regression Analysis.md)
3. [Quantitative Methods - Time Series](Quantitative Methods - Time Series.md)
4. [Quantitative Methods - GARCH](Quantitative Methods - GARCH.md)
5. [Quantitative Methods - Cointegration](Quantitative Methods - Cointegration.md)
6. [Quantitative Methods - Extreme Value Theory](Quantitative Methods - Extreme Value Theory.md)
7. [Strategies - Statistical Arbitrage](Strategies - Statistical Arbitrage.md)
8. [Strategies - Backtesting Engine](Strategies - Backtesting Engine.md)

## :material-robot-outline: ML / AI Engineer

For applying machine learning to markets:

1. [Python Basics - Pandas](Python Basics - Pandas.md)
2. [Algorithms - Machine Learning](Algorithms - Machine Learning.md)
3. [Machine Learning - Feature Engineering](Machine Learning - Feature Engineering.md)
4. [Machine Learning - Random Forest](Machine Learning - Random Forest.md)
5. [Machine Learning Time Series](Machine Learning Time Series.md)
6. [Reinforcement Learning Q Learning](Reinforcement Learning Q Learning.md)
7. [Sentiment Analysis on News](Sentiment Analysis on News.md)

## :material-briefcase-outline: Portfolio & Risk Manager

For allocation, risk budgeting and performance measurement:

1. [CAPM](CAPM.md)
2. [Finance - Covariance Estimation](Finance - Covariance Estimation.md)
3. [Portfolio Optimizer](Portfolio Optimizer.md)
4. [Portfolio Management - Risk Parity](Portfolio Management - Risk Parity.md)
5. [Portfolio Management - Black Litterman](Portfolio Management - Black Litterman.md)
6. [Risk Metrics](Risk Metrics.md)
7. [Value at Risk (VaR)](Value at Risk (VaR).md)
8. [Finance - Information Ratio](Finance - Information Ratio.md)

!!! tip
    Not sure where you sit? Start with
    [Python Basics - NumPy](Python Basics - NumPy.md) and
    [Quantitative Methods - Statistics](Quantitative Methods - Statistics.md) —
    they are the backbone every other path leans on.
"""


def build_glossary():
    return """# Glossary

A quick reference for the terms that show up across the modules. Each entry
points you to the lesson where the concept is implemented.

## Markets & instruments

`Bid / Ask`
:   The best price a buyer will pay (bid) and a seller will accept (ask). The
    gap between them is the **spread**. See *Market Microstructure*.

`Derivative`
:   A contract whose value derives from an underlying asset (e.g. an option on
    a stock). See *Black-Scholes Option Pricing*.

`Greeks`
:   Sensitivities of an option's price — Delta, Gamma, Theta, Vega, Rho. See
    *Finance - Greeks Calculator*.

`Implied Volatility`
:   The volatility that, fed into a pricing model, reproduces the market price
    of an option. See *Finance - Implied Volatility Surface*.

## Risk

`VaR (Value at Risk)`
:   The loss not expected to be exceeded over a horizon at a given confidence
    level. See *Value at Risk (VaR)*.

`Expected Shortfall / CVaR`
:   The average loss *given* that the VaR threshold is breached — a coherent
    tail-risk measure. See *Finance - Expected Shortfall*.

`Drawdown`
:   The peak-to-trough decline of an equity curve. See
    *Risk Metrics - Drawdown Analysis*.

`Sharpe Ratio`
:   Excess return per unit of total volatility. See
    *Sharpe and Sortino Ratio*.

## Statistics & models

`Stationarity`
:   A series whose statistical properties do not change over time — a
    prerequisite for many time-series methods. See
    *Quantitative Methods - Statistics*.

`Cointegration`
:   Two non-stationary series whose linear combination *is* stationary — the
    basis of pairs trading. See *Quantitative Methods - Cointegration*.

`GARCH`
:   A model for time-varying volatility that captures volatility clustering.
    See *Quantitative Methods - GARCH*.

`Monte Carlo`
:   Estimating outcomes by simulating many random paths. See
    *Monte Carlo Portfolio Simulator*.

## Portfolio

`Efficient Frontier`
:   The set of portfolios with the best return for each level of risk. See
    *Portfolio Optimizer*.

`Risk Parity`
:   Allocating so each asset contributes equally to portfolio risk. See
    *Portfolio Management - Risk Parity*.

`Beta`
:   Sensitivity of an asset's return to the market. See *CAPM* and
    *Finance - Beta Calculator*.

!!! info
    Missing a term? Open an issue on
    [GitHub](https://github.com/MeridianAlgo/Learn-Quant/issues) and we will add
    it.
"""


def build_faq():
    return """# FAQ & Contributing

## Frequently asked questions

??? question "Do I need to install Learn-Quant as a package?"
    No. It is a collection of standalone folders. Clone the repo, install the
    requirements once, and run any module directly.

??? question "Which Python version should I use?"
    Python 3.9+ works for every module. CI runs on 3.11.

??? question "Some modules import `scipy` / `scikit-learn` — are those required?"
    They are listed in `requirements.txt`. A handful of modules use them for
    optimisation or ML; the README of each module notes any extra dependency.

??? question "Can I use this code in my own project?"
    Yes — it is MIT licensed. It is written for learning, so audit and adapt
    before putting anything near real capital.

??? question "How do I run the JavaScript modules?"
    Install [Node.js](https://nodejs.org) 18+ and run `node <file>.js` inside
    the module folder.

??? question "Is any of this investment advice?"
    No. Everything here is educational. Markets carry risk; do your own research.

## Contributing

Contributions are very welcome.

- :material-bug-outline: **Found a bug?** Open an
  [issue](https://github.com/MeridianAlgo/Learn-Quant/issues).
- :material-lightbulb-on-outline: **Have a new module or strategy?** Fork the
  repo and open a pull request.
- :material-book-edit-outline: **Improving the docs?** Edit the relevant
  module `README.md` — the site rebuilds automatically.

### Module conventions

A good module folder contains:

1. A `README.md` with **theory → formula → usage → pitfalls**.
2. A single-purpose `.py` implementation with type hints and a runnable
   `__main__` demonstration.
3. A matching test in `z_tests/` (`test_<module>.py`).

Before opening a PR:

```bash
ruff check .
ruff format .
pytest z_tests -q
python docs_builder.py   # regenerate the site
```

Thanks for helping more people learn quant. :material-heart:
"""


# ---------------------------------------------------------------------------
# mkdocs.yml generation.
# ---------------------------------------------------------------------------
def build_mkdocs_yaml(nav_items):
    nav = "\n".join(nav_items)
    return f"""site_name: Learn-Quant
site_description: Master quantitative finance, algorithmic trading, and professional Python engineering — one runnable lesson at a time.
site_url: https://meridianalgo.github.io/Learn-Quant/
repo_url: {REPO_URL}
repo_name: MeridianAlgo/Learn-Quant
copyright: Copyright &copy; MeridianAlgo — MIT Licensed
edit_uri: edit/main/

theme:
  name: material
  language: en
  icon:
    repo: fontawesome/brands/github
    logo: material/chart-line
  features:
    - navigation.tabs
    - navigation.tabs.sticky
    - navigation.sections
    - navigation.indexes
    - navigation.top
    - navigation.footer
    - navigation.instant
    - navigation.instant.prefetch
    - navigation.tracking
    - toc.follow
    - search.suggest
    - search.highlight
    - search.share
    - content.code.copy
    - content.code.annotate
    - content.tooltips
  palette:
    - media: "(prefers-color-scheme)"
      toggle:
        icon: material/brightness-auto
        name: Switch to light mode
    - media: "(prefers-color-scheme: light)"
      scheme: default
      primary: indigo
      accent: deep purple
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - media: "(prefers-color-scheme: dark)"
      scheme: slate
      primary: indigo
      accent: deep purple
      toggle:
        icon: material/brightness-4
        name: Switch to system preference
  font:
    text: Inter
    code: JetBrains Mono

extra_css:
  - stylesheets/extra.css

extra:
  social:
    - icon: fontawesome/brands/github
      link: {REPO_URL}
      name: Learn-Quant on GitHub
  generator: false

docs_dir: z_docs

markdown_extensions:
  - admonition
  - attr_list
  - def_list
  - md_in_html
  - footnotes
  - tables
  - toc:
      permalink: true
      title: On this page
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
  - pymdownx.tasklist:
      custom_checkbox: true
  - pymdownx.emoji:
      emoji_index: !!python/name:material.extensions.emoji.twemoji
      emoji_generator: !!python/name:material.extensions.emoji.to_svg
  - pymdownx.keys
  - pymdownx.mark
  - pymdownx.caret
  - pymdownx.tilde

plugins:
  - search

nav:
{nav}
"""


EXTRA_CSS = """/* Learn-Quant docs theme extras */

:root {
  --lq-grad-1: #5e60ce;
  --lq-grad-2: #6930c3;
  --lq-grad-3: #7400b8;
}

/* ---- Hero ---- */
.lq-hero {
  text-align: center;
  padding: 3.5rem 1rem 2.5rem;
  margin: -1rem -0.8rem 2rem;
  border-radius: 0 0 1.25rem 1.25rem;
  background: linear-gradient(135deg, var(--lq-grad-1), var(--lq-grad-3));
  color: #fff;
}
.lq-hero h1 {
  color: #fff;
  font-weight: 800;
  font-size: 3rem;
  margin-bottom: 0.25rem;
  letter-spacing: -0.02em;
}
.lq-hero h3 {
  color: rgba(255, 255, 255, 0.95);
  font-weight: 500;
  max-width: 46rem;
  margin: 0 auto 1rem;
}
.lq-hero p { color: rgba(255, 255, 255, 0.85); }
.lq-hero .md-button {
  margin: 0.3rem;
  border-color: rgba(255, 255, 255, 0.9);
  color: #fff;
}
.lq-hero .md-button--primary {
  background: #fff;
  color: var(--lq-grad-3);
  border-color: #fff;
}
.lq-hero .md-button:hover {
  background: rgba(255, 255, 255, 0.15);
  border-color: #fff;
  color: #fff;
}
.lq-hero .md-button--primary:hover {
  background: rgba(255, 255, 255, 0.85);
  color: var(--lq-grad-3);
}

/* ---- Badges ---- */
.lq-badges { margin: 0 0 0.5rem; }
.lq-badge {
  display: inline-block;
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.03em;
  text-transform: uppercase;
  padding: 0.15rem 0.55rem;
  margin: 0.1rem 0.25rem 0.1rem 0;
  border-radius: 1rem;
  color: #fff;
  white-space: nowrap;
}
.lq-beginner { background: #2a9d8f; }
.lq-intermediate { background: #e09f3e; }
.lq-advanced { background: #e63946; }
.lq-cat { background: #5e60ce; }
.lq-lang { background: #495057; }

/* ---- Card grid polish ---- */
.md-typeset .grid.cards > ul > li,
.md-typeset .grid.cards > :is(ul, ol) > li {
  border-radius: 0.6rem;
  transition: border-color 0.2s, box-shadow 0.2s, transform 0.2s;
}
.md-typeset .grid.cards > ul > li:hover,
.md-typeset .grid.cards > :is(ul, ol) > li:hover {
  border-color: var(--lq-grad-1);
  box-shadow: 0 0.4rem 1.2rem rgba(94, 96, 206, 0.18);
  transform: translateY(-2px);
}

/* Slightly tighter hero on small screens */
@media screen and (max-width: 44.9em) {
  .lq-hero h1 { font-size: 2.1rem; }
  .lq-hero { padding: 2.5rem 0.75rem 1.75rem; }
}
"""


# ---------------------------------------------------------------------------
# Main build.
# ---------------------------------------------------------------------------
def build_docs():
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
    css_dir = os.path.join(DOCS_DIR, "stylesheets")
    os.makedirs(css_dir, exist_ok=True)

    # Discover modules.
    modules = []
    dirs = [d for d in os.listdir(".") if os.path.isdir(d) and d not in SKIP_DIRS]
    for d in sorted(dirs):
        readme_path = os.path.join(d, "README.md")
        if not os.path.exists(readme_path):
            continue
        title, description = extract_description(readme_path)
        category = categorize(d)
        language, commands = detect_assets(d)
        modules.append(
            {
                "dir": d,
                "title": title,
                "desc": description,
                "file": f"{d}.md",
                "category": category,
                "difficulty": CATEGORY_META[category]["difficulty"],
                "language": language,
                "commands": commands,
            }
        )

    categories = {cat: [] for cat in CATEGORY_ORDER}
    for mod in modules:
        categories[mod["category"]].append(mod)

    # Write augmented module pages.
    module_dirs = {m["dir"] for m in modules}
    for mod in modules:
        siblings = categories[mod["category"]]
        page = build_module_page(mod, siblings, module_dirs)
        with open(os.path.join(DOCS_DIR, mod["file"]), "w", encoding="utf-8") as f:
            f.write(page)

    # Stats for the landing page / guides.
    py_count = sum(
        1
        for d in dirs
        for f in os.listdir(d)
        if f.endswith(".py") and "__init__" not in f
    )
    js_count = sum(1 for d in dirs for f in os.listdir(d) if f.endswith(".js"))
    stats = {"modules": len(modules), "py": py_count, "js": js_count}

    # Standalone pages.
    pages = {
        "index.md": build_index(stats, categories),
        "modules.md": build_modules(categories),
        "getting-started.md": build_getting_started(stats),
        "learning-paths.md": build_learning_paths(),
        "glossary.md": build_glossary(),
        "faq.md": build_faq(),
        os.path.join("stylesheets", "extra.css"): EXTRA_CSS,
    }
    for name, content in pages.items():
        with open(os.path.join(DOCS_DIR, name), "w", encoding="utf-8") as f:
            f.write(content)

    # Navigation.
    def entries(cat, pad):
        return [f'{pad}- "{m["dir"]}": {m["file"]}' for m in categories.get(cat, [])]

    nav_items = [
        "  - Home: index.md",
        "  - Get Started:",
        "      - Getting Started: getting-started.md",
        "      - Learning Paths: learning-paths.md",
        "      - All Modules: modules.md",
        "      - Glossary: glossary.md",
        "      - FAQ & Contributing: faq.md",
        "  - Python Fundamentals:",
        *[f"      {e}" for e in entries("Python Fundamentals", "")],
        "  - Python Advanced:",
        "      - Advanced Python:",
        *[f"          {e}" for e in entries("Advanced Python", "")],
        "      - Data Structures:",
        *[f"          {e}" for e in entries("Data Structures", "")],
        "      - Algorithms:",
        *[f"          {e}" for e in entries("Algorithms", "")],
        "  - Quant:",
        "      - Quantitative Methods:",
        *[f"          {e}" for e in entries("Quantitative Methods", "")],
        "      - Finance & Options:",
        *[f"          {e}" for e in entries("Options, Derivatives & Finance", "")],
        "      - Risk & Performance:",
        *[f"          {e}" for e in entries("Risk & Performance", "")],
        "      - Portfolio Management:",
        *[f"          {e}" for e in entries("Portfolio Management", "")],
        "      - Strategies:",
        *[f"          {e}" for e in entries("Strategies", "")],
        "  - AI & Machine Learning:",
        *[f"      {e}" for e in entries("AI & Machine Learning", "")],
        "  - Utilities & Tools:",
        *[f"      {e}" for e in entries("Market Microstructure", "")],
        *[f"      {e}" for e in entries("Utilities & Tools", "")],
        *[f"      {e}" for e in entries("Other", "")],
    ]

    with open("mkdocs.yml", "w", encoding="utf-8") as f:
        f.write(build_mkdocs_yaml(nav_items))

    total = len(modules)
    used = sum(1 for v in categories.values() if v)
    print(f"Docs built: {total} modules indexed across {used} categories")
    print(f"  Python files: {py_count}  ·  JavaScript modules: {js_count}")
    for cat in CATEGORY_ORDER:
        if categories[cat]:
            print(f"  {cat}: {len(categories[cat])} modules")


if __name__ == "__main__":
    build_docs()
