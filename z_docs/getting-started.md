# Getting started

Learn-Quant is 138 separate folders. There is no package to
install and nothing to build. Clone it, install the requirements once, and run
whichever lesson you want.

## Install

```bash
git clone https://github.com/MeridianAlgo/Learn-Quant
cd Learn-Quant
python -m venv .venv
# Windows:      .venv\Scripts\activate
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
