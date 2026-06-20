# Getting Started

Learn-Quant is a collection of **118 self-contained modules**. There is no
package to install and no build step — you clone the repo and run whichever
lesson you want to learn from.

## 1. Install

```bash
git clone https://github.com/MeridianAlgo/Learn-Quant
cd Learn-Quant
python -m venv .venv
# Windows:  .venv\Scripts\activate
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
