"""Learn-Quant documentation builder.

Turns the repository of self-contained lesson folders into the MkDocs site.
For every module it:

* copies the module ``README.md`` into ``z_docs/``,
* adds a one line metadata strip under the title (level, track, language,
  source link), a "Running it" section and a list of other lessons in the
  same track,

and on top of the module pages it writes the home page, a single page index
of every module, and a few standalone guides (getting started, learning
paths, glossary, FAQ). Finally it regenerates ``mkdocs.yml`` so the
navigation always matches what is on disk.

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

# Lessons added in the latest release, shown on the home page.
RECENT = [
    "Portfolio Management - Hierarchical Risk Parity",
    "Quantitative Methods - Kernel Density Estimation",
    "Market Microstructure - Liquidity Measures",
    "Strategies - Optimal Execution",
]

# ---------------------------------------------------------------------------
# Tracks: ordering, a one line description and a default level.
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
        "blurb": "Numbers, strings, functions, NumPy and pandas, using market data as the running example.",
        "difficulty": "Beginner",
    },
    "Data Structures": {
        "blurb": "Arrays, lists, dicts and sets, and when each one is the right container.",
        "difficulty": "Beginner",
    },
    "Algorithms": {
        "blurb": "Sorting, searching, graphs and dynamic programming applied to prices and orders.",
        "difficulty": "Intermediate",
    },
    "Advanced Python": {
        "blurb": "Async, OOP, multiprocessing and error handling for code that has to keep running.",
        "difficulty": "Intermediate",
    },
    "Quantitative Methods": {
        "blurb": "Statistics, time series, stochastic processes and numerical methods, built from scratch.",
        "difficulty": "Advanced",
    },
    "Options, Derivatives & Finance": {
        "blurb": "Option pricing and Greeks, bonds and yield curves, valuation and sizing.",
        "difficulty": "Intermediate",
    },
    "Risk & Performance": {
        "blurb": "VaR, drawdowns and the ratios used to judge whether a track record is any good.",
        "difficulty": "Intermediate",
    },
    "Portfolio Management": {
        "blurb": "Mean variance, risk parity, Black Litterman and other ways to size a book.",
        "difficulty": "Advanced",
    },
    "Strategies": {
        "blurb": "Signals, backtests and execution for the standard strategy families.",
        "difficulty": "Advanced",
    },
    "AI & Machine Learning": {
        "blurb": "Classifiers, clustering, trees and reinforcement learning, with the leakage traps spelled out.",
        "difficulty": "Advanced",
    },
    "Market Microstructure": {
        "blurb": "Order books, spreads, liquidity and the cost of actually trading.",
        "difficulty": "Advanced",
    },
    "Utilities & Tools": {
        "blurb": "Data feeds, logging, FX conversion, calendars and other plumbing.",
        "difficulty": "Beginner",
    },
    "Other": {
        "blurb": "Lessons that do not fit a track yet.",
        "difficulty": "Intermediate",
    },
}


# ---------------------------------------------------------------------------
# README parsing helpers.
# ---------------------------------------------------------------------------
def _first_sentence(text, limit=220):
    text = re.sub(r"\*\*|__|`", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    ends = [m.start() for m in re.finditer(r"(?<=[.!?])\s", text)]
    sentence = text[: ends[0]] if ends else text
    # A very short opener ("You found an edge.") reads better with its follow up.
    if len(sentence) < 40 and len(ends) > 1:
        sentence = text[: ends[1]]
    if len(sentence) > limit:
        sentence = sentence[:limit].rsplit(" ", 1)[0] + "..."
    return sentence


def extract_description(readme_path):
    """Return ``(title, first_sentence_of_first_paragraph)`` from a module README.

    READMEs wrap their prose, so the whole first paragraph is collected before
    the first sentence is taken. Badges, lists, tables, quotes and code are
    skipped.
    """
    try:
        with open(readme_path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        return "", ""

    title = ""
    paragraphs = []
    current = []
    in_code = False

    def flush():
        if current:
            paragraphs.append(" ".join(current))
            current.clear()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            flush()
            continue
        if in_code:
            continue
        if stripped.startswith("# ") and not title:
            title = stripped[2:].strip()
            continue
        if not title:
            continue
        is_list = re.match(r"([-*+]|\d+\.)\s", stripped)
        if not stripped or is_list or stripped.startswith(("#", "[!", "!", "|", ">", "<", "---")):
            flush()
            continue
        current.append(stripped)
    flush()

    # Skip lead-ins that only introduce a list ("You'll learn:") and the
    # "no external APIs" disclaimer several utility READMEs open with.
    for para in paragraphs:
        sentence = _first_sentence(para)
        if sentence.rstrip().endswith(":") or "external API" in sentence:
            continue
        return title, sentence
    return title, ""


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
            "Calmar",
            "Omega",
            "Treynor",
            "Jensens Alpha",
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
        language = "Python, JavaScript"
    elif js_files:
        language = "JavaScript"
    elif py_files:
        language = "Python"
    else:
        language = "Reference"
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

    # One folder, "Value at Risk (VaR)", has brackets in its name, so the link
    # target is allowed to contain a balanced pair rather than stopping at the
    # first closing bracket it meets.
    return re.sub(r"\]\(\.\./((?:[^()]|\([^()]*\))+)\)", repl, body)


def _link(mod):
    return f"[{mod['dir']}]({mod['file']})"


def _cell(text):
    return text.replace("|", "\\|")


# Folder prefixes that just repeat the track name and can be dropped in tables.
_TRACK_PREFIXES = {
    "Python Basics",
    "Data Structures",
    "Algorithms",
    "Advanced Python",
    "Quantitative Methods",
    "Finance",
    "Machine Learning",
    "Strategies",
    "Portfolio Management",
    "Market Microstructure",
}


def _short_name(name):
    prefix, sep, rest = name.partition(" - ")
    return rest if sep and prefix in _TRACK_PREFIXES else name


# ---------------------------------------------------------------------------
# Per-module page assembly.
# ---------------------------------------------------------------------------
def build_module_page(mod, siblings, module_dirs):
    """Return the augmented markdown for a single module page."""
    with open(os.path.join(mod["dir"], "README.md"), encoding="utf-8") as f:
        body = f.read()
    body = rewrite_relative_links(body, module_dirs).rstrip() + "\n"

    source_url = f"{REPO_URL}/tree/main/{quote(mod['dir'])}"
    meta = (
        f'<p class="lq-meta">{mod["difficulty"]} <span>/</span> {mod["category"]} '
        f'<span>/</span> {mod["language"]} <span>/</span> <a href="{source_url}">source</a></p>'
    )

    # Put the metadata strip directly under the README's own title.
    lines = body.splitlines()
    h1 = next((i for i, line in enumerate(lines) if line.startswith("# ")), None)
    if h1 is None:
        lines = [meta, ""] + lines
    else:
        lines = lines[: h1 + 1] + ["", meta, ""] + lines[h1 + 1 :]

    tail = ["", "---", "", "## Running it", ""]
    if mod["commands"]:
        tail += ["```bash", f'cd "{mod["dir"]}"', *mod["commands"], "```"]
    else:
        tail.append("Nothing to run directly. Import the module from your own code.")
    tail.append("")

    related = [s for s in siblings if s["dir"] != mod["dir"]][:6]
    if related:
        tail += [f"## More in {mod['category']}", ""]
        for s in related:
            tail.append(f"- {_link(s)}" + (f": {s['desc']}" if s["desc"] else ""))
        tail.append("")
    tail.append("[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).")
    tail.append("")

    return "\n".join(lines + tail)


# ---------------------------------------------------------------------------
# Standalone pages.
# ---------------------------------------------------------------------------
def build_index(stats, categories, by_dir):
    track_rows = ["| Track | Lessons | Covers |", "|---|---:|---|"]
    for cat in CATEGORY_ORDER:
        mods = categories.get(cat, [])
        if not mods:
            continue
        track_rows.append(f"| [{cat}]({mods[0]['file']}) | {len(mods)} | {CATEGORY_META[cat]['blurb']} |")
    tracks = "\n".join(track_rows)

    recent = "\n".join(
        f"- {_link(by_dir[d])}" + (f": {by_dir[d]['desc']}" if by_dir[d]["desc"] else "") for d in RECENT if d in by_dir
    )

    return f"""---
hide:
  - navigation
  - toc
---

# Learn-Quant

<p class="lq-lede">{stats["modules"]} small, runnable lessons on quantitative finance and the
Python it is written in. Each folder covers one idea. The README explains it,
the code implements it without leaning on a library to do the hard part, and
a test checks the code does what the README says.</p>

<p class="lq-meta">{stats["py"]} Python files <span>/</span> {stats["js"]} JavaScript files <span>/</span> MIT licence <span>/</span> <a href="{REPO_URL}">GitHub</a></p>

## Where to start

- **New to Python.** Begin with [Python Basics - Numbers](Python Basics - Numbers.md)
  and work down the Python Fundamentals tab.
- **Comfortable with Python, new to finance.** Read
  [Quantitative Methods - Statistics](Quantitative Methods - Statistics.md), then
  [Black-Scholes Option Pricing](Black-Scholes Option Pricing.md).
- **Looking for one topic.** Press <kbd>/</kbd> to search, or scan the
  [module index](modules.md).

There are also [reading orders](learning-paths.md) for a few common goals.

## Tracks

{tracks}

## Recently added

{recent}

## Running a lesson

```bash
git clone {REPO_URL}
cd Learn-Quant
pip install -r requirements.txt

cd "Quantitative Methods - GARCH"
python garch.py
```

The [getting started page](getting-started.md) covers virtual environments,
the JavaScript lessons and running the tests.
"""


def build_modules(categories):
    out = [
        "# All modules",
        "",
        "Every lesson on one page, grouped by track. The level is a rough guide,",
        "not a gate.",
        "",
    ]
    for cat in CATEGORY_ORDER:
        mods = categories.get(cat, [])
        if not mods:
            continue
        out += [f"## {cat}", "", CATEGORY_META[cat]["blurb"], ""]
        out += ["| Module | Level | Language | About |", "|---|---|---|---|"]
        for m in mods:
            desc = m["desc"] or m["title"] or m["dir"]
            out.append(
                f"| [{_short_name(m['dir'])}]({m['file']}) | {m['difficulty']} | {m['language']} | {_cell(desc)} |"
            )
        out.append("")
    return "\n".join(out)


def build_getting_started(stats):
    return f"""# Getting started

Learn-Quant is {stats["modules"]} separate folders. There is no package to
install and nothing to build. Clone it, install the requirements once, and run
whichever lesson you want.

## Install

```bash
git clone {REPO_URL}
cd Learn-Quant
python -m venv .venv
# Windows:      .venv\\Scripts\\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

The JavaScript lessons need [Node.js](https://nodejs.org) 18 or later and
nothing else.

The Python requirements are `numpy`, `pandas`, `scipy`, `scikit-learn` and
`matplotlib`, plus a couple of small extras. Linting and test tools are in
`requirements-dev.txt`.

## Run a lesson

Change into a folder and run its main file:

```bash
cd "Black-Scholes Option Pricing"
python black_scholes.py
```

```bash
cd "Options Pricing - JavaScript"
node blackScholes.js
```

Files ending in `_tutorial.py` are interactive and will ask you questions as
they go.

## What is in a folder

```text
Quantitative Methods - GARCH/
├── README.md      the explanation: idea, formula, usage, caveats
└── garch.py       the code, with a demo under if __name__ == "__main__"
```

Read the README first, then open the code next to it. They are written to be
read together.

## A rough order

If you want to go through everything, the tabs along the top are already in a
sensible order:

1. Python Fundamentals, then data structures and algorithms
2. Advanced Python
3. Quantitative Methods
4. Options and finance, then risk, portfolios and strategies
5. Machine learning and market microstructure

The [learning paths](learning-paths.md) page has shorter routes for specific
goals.

## Tests

```bash
pip install -r requirements-dev.txt
pytest z_tests -q
ruff check .
```
"""


def build_learning_paths():
    return """# Learning paths

Shorter routes through the material for a few common goals. Each list is in
reading order. If something in a lesson is unfamiliar, the earlier lessons in
its track usually cover it.

## From zero to a first backtest

1. [Python Basics - Numbers](Python Basics - Numbers.md)
2. [Python Basics - Functions](Python Basics - Functions.md)
3. [Python Basics - NumPy](Python Basics - NumPy.md)
4. [Python Basics - Pandas](Python Basics - Pandas.md)
5. [Data Structures - Arrays](Data Structures - Arrays.md)
6. [Advanced Python - OOP](Advanced Python - OOP.md)
7. [Quantitative Methods - Statistics](Quantitative Methods - Statistics.md)
8. [Black-Scholes Option Pricing](Black-Scholes Option Pricing.md)
9. [Risk Metrics](Risk Metrics.md)
10. [Portfolio Optimizer](Portfolio Optimizer.md)
11. [Strategies - Pairs Trading](Strategies - Pairs Trading.md)
12. [Strategies - Backtesting Engine](Strategies - Backtesting Engine.md)

## Options and derivatives

1. [Python Basics - NumPy](Python Basics - NumPy.md)
2. [Quantitative Methods - Stochastic Processes](Quantitative Methods - Stochastic Processes.md)
3. [Black-Scholes Option Pricing](Black-Scholes Option Pricing.md)
4. [Finance - Greeks Calculator](Finance - Greeks Calculator.md)
5. [Options Pricing - Binomial Tree](Options Pricing - Binomial Tree.md)
6. [Advanced Options Pricing](Advanced Options Pricing.md)
7. [Finance - Exotic Options](Finance - Exotic Options.md)
8. [Finance - Implied Volatility Surface](Finance - Implied Volatility Surface.md)
9. [Finance - Options Strategies](Finance - Options Strategies.md)

## Quant research

1. [Quantitative Methods - Statistics](Quantitative Methods - Statistics.md)
2. [Quantitative Methods - Hypothesis Testing](Quantitative Methods - Hypothesis Testing.md)
3. [Quantitative Methods - Regression Analysis](Quantitative Methods - Regression Analysis.md)
4. [Quantitative Methods - Kernel Density Estimation](Quantitative Methods - Kernel Density Estimation.md)
5. [Quantitative Methods - Time Series](Quantitative Methods - Time Series.md)
6. [Quantitative Methods - GARCH](Quantitative Methods - GARCH.md)
7. [Quantitative Methods - Cointegration](Quantitative Methods - Cointegration.md)
8. [Quantitative Methods - Extreme Value Theory](Quantitative Methods - Extreme Value Theory.md)
9. [Strategies - Statistical Arbitrage](Strategies - Statistical Arbitrage.md)
10. [Quantitative Methods - Bootstrap](Quantitative Methods - Bootstrap.md)

## Machine learning on market data

1. [Python Basics - Pandas](Python Basics - Pandas.md)
2. [Algorithms - Machine Learning](Algorithms - Machine Learning.md)
3. [Machine Learning - Feature Engineering](Machine Learning - Feature Engineering.md)
4. [Machine Learning - Cross Validation](Machine Learning - Cross Validation.md)
5. [Machine Learning - Logistic Regression](Machine Learning - Logistic Regression.md)
6. [Machine Learning - Random Forest](Machine Learning - Random Forest.md)
7. [Machine Learning Time Series](Machine Learning Time Series.md)
8. [Reinforcement Learning Q Learning](Reinforcement Learning Q Learning.md)

## Portfolio and risk

1. [CAPM](CAPM.md)
2. [Finance - Covariance Estimation](Finance - Covariance Estimation.md)
3. [Portfolio Optimizer](Portfolio Optimizer.md)
4. [Portfolio Management - Risk Parity](Portfolio Management - Risk Parity.md)
5. [Portfolio Management - Hierarchical Risk Parity](Portfolio Management - Hierarchical Risk Parity.md)
6. [Portfolio Management - Black Litterman](Portfolio Management - Black Litterman.md)
7. [Value at Risk (VaR)](Value at Risk (VaR).md)
8. [Finance - Expected Shortfall](Finance - Expected Shortfall.md)
9. [Finance - Information Ratio](Finance - Information Ratio.md)

## Execution and trading costs

1. [Market Microstructure](Market Microstructure.md)
2. [Market Microstructure - Liquidity Measures](Market Microstructure - Liquidity Measures.md)
3. [Finance - Transaction Cost Analysis](Finance - Transaction Cost Analysis.md)
4. [Strategies - Optimal Execution](Strategies - Optimal Execution.md)
5. [Order Execution Simulator](Order Execution Simulator.md)
6. [Strategies - Market Making](Strategies - Market Making.md)

If you are not sure where you fit, [Python Basics - NumPy](Python Basics - NumPy.md)
and [Quantitative Methods - Statistics](Quantitative Methods - Statistics.md)
are used by nearly everything else.
"""


def build_glossary():
    return """# Glossary

Short definitions for terms that come up across the lessons, with a pointer to
where each one is implemented.

## Markets

Bid and ask
:   The best price a buyer will pay and a seller will accept. The gap is the
    spread. See [Market Microstructure](Market Microstructure.md).

Effective spread
:   Twice the distance between a trade price and the midquote at the time,
    which is what the trade actually cost. See
    [Market Microstructure - Liquidity Measures](Market Microstructure - Liquidity Measures.md).

Kyle's lambda
:   How far price moves per unit of net order flow. Higher means a thinner
    market. See
    [Market Microstructure - Liquidity Measures](Market Microstructure - Liquidity Measures.md).

Implementation shortfall
:   The difference between the price when you decided to trade and the price
    you actually got, across the whole order. See
    [Strategies - Optimal Execution](Strategies - Optimal Execution.md).

## Options

Derivative
:   A contract whose value depends on some underlying asset, such as an
    option on a stock. See [Black-Scholes Option Pricing](Black-Scholes Option Pricing.md).

Greeks
:   The sensitivities of an option price: delta, gamma, theta, vega, rho. See
    [Finance - Greeks Calculator](Finance - Greeks Calculator.md).

Implied volatility
:   The volatility that makes a pricing model match the market price of an
    option. See [Finance - Implied Volatility Surface](Finance - Implied Volatility Surface.md).

## Risk

Value at Risk
:   A loss level not expected to be exceeded over a horizon at a given
    confidence. See [Value at Risk (VaR)](Value at Risk (VaR).md).

Expected shortfall
:   The average loss on the days that are worse than VaR. See
    [Finance - Expected Shortfall](Finance - Expected Shortfall.md).

Drawdown
:   The fall from the highest point of an equity curve to a later low. See
    [Risk Metrics - Drawdown Analysis](Risk Metrics - Drawdown Analysis.md).

Sharpe ratio
:   Excess return divided by volatility. See
    [Sharpe and Sortino Ratio](Sharpe and Sortino Ratio.md).

## Statistics

Stationarity
:   A series whose mean, variance and autocorrelation do not drift over time.
    Many methods assume it. See [Quantitative Methods - Time Series](Quantitative Methods - Time Series.md).

Cointegration
:   Two wandering series whose spread does not wander. The basis of pairs
    trading. See [Quantitative Methods - Cointegration](Quantitative Methods - Cointegration.md).

GARCH
:   A model where today's volatility depends on yesterday's shock and
    yesterday's volatility, which produces volatility clustering. See
    [Quantitative Methods - GARCH](Quantitative Methods - GARCH.md).

Bandwidth
:   The width of the bump placed on each point in a kernel density estimate.
    It matters far more than the bump's shape. See
    [Quantitative Methods - Kernel Density Estimation](Quantitative Methods - Kernel Density Estimation.md).

Monte Carlo
:   Estimating something by simulating many random outcomes and averaging. See
    [Monte Carlo Portfolio Simulator](Monte Carlo Portfolio Simulator.md).

## Portfolios

Efficient frontier
:   The best achievable return for each level of risk. See
    [Portfolio Optimizer](Portfolio Optimizer.md).

Risk parity
:   Weighting assets so each contributes the same amount of risk. See
    [Portfolio Management - Risk Parity](Portfolio Management - Risk Parity.md).

Hierarchical risk parity
:   Risk parity applied down a tree of correlated clusters, which avoids
    inverting the covariance matrix. See
    [Portfolio Management - Hierarchical Risk Parity](Portfolio Management - Hierarchical Risk Parity.md).

Beta
:   How much an asset moves with the market. See [CAPM](CAPM.md) and
    [Finance - Beta Calculator](Finance - Beta Calculator.md).

Something missing? [Open an issue](https://github.com/MeridianAlgo/Learn-Quant/issues).
"""


def build_faq():
    return """# FAQ and contributing

### Do I need to install Learn-Quant as a package?

No. It is a folder of folders. Install the requirements once and run any
lesson directly.

### Which Python version?

3.9 or later. CI runs 3.11.

### Can I use the code in my own project?

Yes, it is MIT licensed. It is written to be read, not to be fast or battle
tested, so check it before it goes anywhere near real money.

### How do I run the JavaScript lessons?

Install Node.js 18 or later and run `node <file>.js` in the lesson's folder.
They have no dependencies.

### Is this investment advice?

No. It is teaching material.

## Contributing

Bug reports and pull requests are welcome on
[GitHub](https://github.com/MeridianAlgo/Learn-Quant). To change a page on
this site, edit the `README.md` in the lesson's folder. The site is rebuilt
from those on every push to `main`.

A new lesson should have:

1. A `README.md` that explains the idea, gives the formula, shows usage and
   lists the caveats.
2. One focused `.py` (or `.js`) file with a demo under `__main__`.
3. A test in `z_tests/` named `test_<module>.py`.

Before opening a pull request:

```bash
ruff check .
ruff format .
pytest z_tests -q
python docs_builder.py
```
"""


# ---------------------------------------------------------------------------
# mkdocs.yml generation.
# ---------------------------------------------------------------------------
def build_mkdocs_yaml(nav_items):
    nav = "\n".join(nav_items)
    return f"""site_name: Learn-Quant
site_description: Runnable lessons in quantitative finance and Python. One folder per topic, each with an explanation, an implementation and a test.
site_url: https://meridianalgo.github.io/Learn-Quant/
repo_url: {REPO_URL}
repo_name: MeridianAlgo/Learn-Quant
copyright: MeridianAlgo, MIT licence
edit_uri: edit/main/

theme:
  name: material
  language: en
  icon:
    repo: fontawesome/brands/github
    logo: material/sigma
  features:
    - navigation.tabs
    - navigation.sections
    - navigation.indexes
    - navigation.top
    - navigation.footer
    - navigation.instant
    - navigation.tracking
    - toc.follow
    - search.suggest
    - search.highlight
    - content.code.copy
  palette:
    - media: "(prefers-color-scheme: light)"
      scheme: default
      primary: custom
      accent: custom
      toggle:
        icon: material/weather-night
        name: Dark theme
    - media: "(prefers-color-scheme: dark)"
      scheme: slate
      primary: custom
      accent: custom
      toggle:
        icon: material/weather-sunny
        name: Light theme
  font:
    text: IBM Plex Sans
    code: IBM Plex Mono

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


EXTRA_CSS = """/* Learn-Quant site styles.
   Plain documentation look: paper background, ink text, one accent colour,
   serif headings, thin rules instead of shadows and cards. */

@import url("https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&display=swap");

:root,
[data-md-color-scheme="default"] {
  --lq-paper: #fbfaf6;
  --lq-ink: #1f1e1b;
  --lq-muted: #6d695f;
  --lq-rule: #e3dfd4;
  --lq-accent: #0f6b4c;
  --lq-code: #f2efe7;
  --lq-field: rgba(31, 30, 27, 0.06);

  --md-primary-fg-color: var(--lq-paper);
  --md-primary-fg-color--light: var(--lq-paper);
  --md-primary-fg-color--dark: var(--lq-paper);
  --md-primary-bg-color: var(--lq-ink);
  --md-primary-bg-color--light: var(--lq-muted);
  --md-accent-fg-color: var(--lq-accent);
  --md-accent-fg-color--transparent: rgba(15, 107, 76, 0.08);
  --md-default-bg-color: var(--lq-paper);
  --md-typeset-a-color: var(--lq-accent);
  --md-code-bg-color: var(--lq-code);
  --md-footer-bg-color: var(--lq-paper);
  --md-footer-bg-color--dark: var(--lq-paper);
  --md-footer-fg-color: var(--lq-ink);
  --md-footer-fg-color--light: var(--lq-muted);
  --md-footer-fg-color--lighter: var(--lq-muted);
}

[data-md-color-scheme="slate"] {
  --lq-paper: #161614;
  --lq-ink: #e6e3dc;
  --lq-muted: #9d998f;
  --lq-rule: #2d2c28;
  --lq-accent: #63c29b;
  --lq-code: #1f1f1c;
  --lq-field: rgba(230, 227, 220, 0.07);

  --md-hue: 45;
  --md-primary-fg-color: var(--lq-paper);
  --md-primary-fg-color--light: var(--lq-paper);
  --md-primary-fg-color--dark: var(--lq-paper);
  --md-primary-bg-color: var(--lq-ink);
  --md-primary-bg-color--light: var(--lq-muted);
  --md-accent-fg-color: var(--lq-accent);
  --md-accent-fg-color--transparent: rgba(99, 194, 155, 0.1);
  --md-default-bg-color: var(--lq-paper);
  --md-typeset-a-color: var(--lq-accent);
  --md-code-bg-color: var(--lq-code);
  --md-footer-bg-color: var(--lq-paper);
  --md-footer-bg-color--dark: var(--lq-paper);
  --md-footer-fg-color: var(--lq-ink);
  --md-footer-fg-color--light: var(--lq-muted);
  --md-footer-fg-color--lighter: var(--lq-muted);
}

/* ---- Header and tabs: flat, ruled ---- */
.md-header,
.md-header--shadow {
  box-shadow: none;
  border-bottom: 1px solid var(--lq-rule);
}
.md-tabs {
  border-bottom: 1px solid var(--lq-rule);
}
.md-tabs__link {
  font-size: 0.68rem;
  opacity: 0.75;
}
.md-tabs__link--active,
.md-tabs__link:hover {
  opacity: 1;
  color: var(--lq-accent);
}
.md-header__title {
  font-family: "Source Serif 4", Georgia, serif;
  font-weight: 600;
  font-size: 0.95rem;
}

/* Search field sits on a light header, so give it a light fill. */
@media screen and (min-width: 60em) {
  .md-search__form,
  .md-search__form:hover {
    background-color: var(--lq-field);
    border-radius: 3px;
  }
  .md-search__input {
    color: var(--lq-ink);
  }
}
.md-search__input::placeholder,
.md-search__input ~ .md-search__icon {
  color: var(--lq-muted);
}

/* ---- Type ---- */
.md-typeset {
  font-size: 0.8rem;
  line-height: 1.7;
}
.md-typeset h1,
.md-typeset h2,
.md-typeset h3 {
  font-family: "Source Serif 4", Georgia, serif;
  color: var(--lq-ink);
  letter-spacing: -0.005em;
}
.md-typeset h1 {
  font-weight: 700;
  font-size: 2rem;
  margin-bottom: 0.4em;
}
.md-typeset h2 {
  font-weight: 600;
  font-size: 1.3rem;
  margin-top: 2em;
  padding-top: 0.9em;
  border-top: 1px solid var(--lq-rule);
}
.md-typeset h3 {
  font-weight: 600;
  font-size: 1.05rem;
}
.md-typeset a {
  text-decoration: underline;
  text-decoration-thickness: 1px;
  text-underline-offset: 0.18em;
  text-decoration-color: color-mix(in srgb, var(--lq-accent) 35%, transparent);
}
.md-typeset a:hover {
  text-decoration-color: currentColor;
}
.md-typeset .headerlink {
  text-decoration: none;
}

/* ---- Metadata strip under each title ---- */
.md-typeset .lq-meta {
  font-family: var(--md-code-font-family);
  font-size: 0.66rem;
  color: var(--lq-muted);
  margin: -0.6em 0 1.6em;
}
.md-typeset .lq-meta span {
  margin: 0 0.35em;
  opacity: 0.6;
}
.md-typeset .lq-meta a {
  color: inherit;
}

.md-typeset .lq-lede {
  font-size: 1.02rem;
  line-height: 1.6;
  max-width: 36rem;
  margin-top: 0;
}

/* ---- Code ---- */
.md-typeset pre > code,
.md-typeset code {
  border-radius: 2px;
}
.md-typeset pre > code {
  border: 1px solid var(--lq-rule);
}

/* ---- Tables: rules, no boxes ---- */
.md-typeset table:not([class]) {
  border: none;
  border-radius: 0;
  font-size: 0.74rem;
  box-shadow: none;
}
.md-typeset table:not([class]) th {
  background: transparent;
  color: var(--lq-ink);
  font-weight: 600;
  border-bottom: 1.5px solid var(--lq-ink);
}
.md-typeset table:not([class]) td {
  border-top: 1px solid var(--lq-rule);
  vertical-align: top;
}
.md-typeset table:not([class]) tr:hover {
  background: transparent;
  box-shadow: none;
}
.md-typeset table:not([class]) td:first-child {
  min-width: 11rem;
}
.md-typeset table:not([class]) td:nth-child(2),
.md-typeset table:not([class]) td:nth-child(3) {
  white-space: nowrap;
}

/* ---- Admonitions and buttons, toned down ---- */
.md-typeset .admonition,
.md-typeset details {
  border-width: 0 0 0 3px;
  border-radius: 0;
  box-shadow: none;
  background: var(--lq-code);
  font-size: 0.76rem;
}
.md-typeset .md-button {
  border-radius: 2px;
  font-weight: 500;
}

/* ---- Definition lists in the glossary ---- */
.md-typeset dt {
  font-weight: 600;
  margin-top: 1.1em;
}

/* ---- Footer ---- */
.md-footer {
  border-top: 1px solid var(--lq-rule);
}
.md-footer-meta {
  background: var(--lq-paper);
}

@media screen and (max-width: 44.9em) {
  .md-typeset h1 { font-size: 1.6rem; }
  .md-typeset table:not([class]) td:first-child { min-width: 0; }
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
    by_dir = {m["dir"]: m for m in modules}

    # Write augmented module pages.
    module_dirs = set(by_dir)
    for mod in modules:
        siblings = categories[mod["category"]]
        page = build_module_page(mod, siblings, module_dirs)
        with open(os.path.join(DOCS_DIR, mod["file"]), "w", encoding="utf-8") as f:
            f.write(page)

    # Stats for the landing page / guides.
    py_count = sum(1 for d in dirs for f in os.listdir(d) if f.endswith(".py") and "__init__" not in f)
    js_count = sum(1 for d in dirs for f in os.listdir(d) if f.endswith(".js"))
    stats = {"modules": len(modules), "py": py_count, "js": js_count}

    # Standalone pages.
    pages = {
        "index.md": build_index(stats, categories, by_dir),
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
    def entries(cat):
        return [f'- "{m["dir"]}": {m["file"]}' for m in categories.get(cat, [])]

    nav_items = [
        "  - Home: index.md",
        "  - Start here:",
        "      - Getting started: getting-started.md",
        "      - Learning paths: learning-paths.md",
        "      - All modules: modules.md",
        "      - Glossary: glossary.md",
        "      - FAQ and contributing: faq.md",
        "  - Python:",
        *[f"      {e}" for e in entries("Python Fundamentals")],
        "  - Engineering:",
        "      - Advanced Python:",
        *[f"          {e}" for e in entries("Advanced Python")],
        "      - Data Structures:",
        *[f"          {e}" for e in entries("Data Structures")],
        "      - Algorithms:",
        *[f"          {e}" for e in entries("Algorithms")],
        "  - Quant:",
        "      - Quantitative Methods:",
        *[f"          {e}" for e in entries("Quantitative Methods")],
        "      - Options and Finance:",
        *[f"          {e}" for e in entries("Options, Derivatives & Finance")],
        "      - Risk and Performance:",
        *[f"          {e}" for e in entries("Risk & Performance")],
        "      - Portfolio Management:",
        *[f"          {e}" for e in entries("Portfolio Management")],
        "      - Strategies:",
        *[f"          {e}" for e in entries("Strategies")],
        "  - Machine Learning:",
        *[f"      {e}" for e in entries("AI & Machine Learning")],
        "  - Markets and Tools:",
        "      - Market Microstructure:",
        *[f"          {e}" for e in entries("Market Microstructure")],
        "      - Utilities:",
        *[f"          {e}" for e in entries("Utilities & Tools")],
        *[f"          {e}" for e in entries("Other")],
    ]

    with open("mkdocs.yml", "w", encoding="utf-8") as f:
        f.write(build_mkdocs_yaml(nav_items))

    total = len(modules)
    used = sum(1 for v in categories.values() if v)
    print(f"Docs built: {total} modules indexed across {used} tracks")
    print(f"  Python files: {py_count}, JavaScript files: {js_count}")
    for cat in CATEGORY_ORDER:
        if categories[cat]:
            print(f"  {cat}: {len(categories[cat])} modules")


if __name__ == "__main__":
    build_docs()
