"""
Essential Libraries
-------------------
A working quant leans on a small set of libraries for almost everything. A few
of them ship with Python itself and are always available. The rest are third
party packages you install once and then reuse across every project. Knowing
what each one is for, and reaching for the right one without thinking, is a large
part of being productive.

This lesson lists the ten imports you will see most often, gives a one line
description of each through library_overview, checks whether each is available
with is_available, and runs a tiny demonstration of every one in the __main__
block so you can see them in action.
"""

from __future__ import annotations

import importlib.util


def library_overview() -> dict[str, str]:
    """Return the ten core imports mapped to a short description of each.

    The dict has exactly ten entries. The first six are third party packages you
    install. The last four ship with Python in the standard library.
    """
    return {
        "numpy": "Fast numerical arrays and vectorized math, the base of the stack.",
        "pandas": "Labeled tables and time series for loading and shaping data.",
        "matplotlib": "Plotting library for turning numbers into charts.",
        "scipy": "Scientific computing such as statistics, optimization, and integration.",
        "sklearn": "Machine learning models for regression, classification, and clustering.",
        "statsmodels": "Statistical models and econometrics with rich result summaries.",
        "datetime": "Standard library dates, times, and durations.",
        "math": "Standard library basic math functions and constants.",
        "random": "Standard library random number generation.",
        "json": "Standard library reading and writing of JSON text.",
    }


def is_available(name: str) -> bool:
    """Return True if the named import can be located on this machine.

    Uses find_spec so it does not actually run the module code.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _demo_numpy() -> str:
    import numpy as np

    arr = np.array([1.0, 2.0, 3.0, 4.0])
    return f"numpy mean of {arr.tolist()} is {arr.mean()}"


def _demo_pandas() -> str:
    import pandas as pd

    df = pd.DataFrame({"price": [100, 101, 99], "volume": [10, 12, 9]})
    return f"pandas built a DataFrame with shape {df.shape}"


def _demo_matplotlib() -> str:
    import matplotlib

    matplotlib.use("Agg")  # headless backend, no window needed
    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.close(fig)
    return "matplotlib created and closed a figure object"


def _demo_scipy() -> str:
    from scipy import stats

    return f"scipy normal cdf at 0 is {stats.norm.cdf(0.0)}"


def _demo_sklearn() -> str:
    from sklearn.linear_model import LinearRegression

    model = LinearRegression().fit([[0], [1], [2]], [0, 2, 4])
    return f"sklearn fit a line with slope {round(float(model.coef_[0]), 2)}"


def _demo_statsmodels() -> str:
    import statsmodels

    return f"statsmodels is ready at version {statsmodels.__version__}"


def _demo_datetime() -> str:
    from datetime import date, timedelta

    settle = date(2024, 1, 1) + timedelta(days=2)
    return f"datetime added two days to get {settle.isoformat()}"


def _demo_math() -> str:
    import math

    return f"math sqrt of pi is {round(math.sqrt(math.pi), 4)}"


def _demo_random() -> str:
    import random

    random.seed(0)
    return f"random drew the value {round(random.random(), 4)} after seeding"


def _demo_json() -> str:
    import json

    text = json.dumps({"ok": True})
    back = json.loads(text)
    return f"json round tripped {text} back to {back}"


DEMOS = {
    "numpy": _demo_numpy,
    "pandas": _demo_pandas,
    "matplotlib": _demo_matplotlib,
    "scipy": _demo_scipy,
    "sklearn": _demo_sklearn,
    "statsmodels": _demo_statsmodels,
    "datetime": _demo_datetime,
    "math": _demo_math,
    "random": _demo_random,
    "json": _demo_json,
}


def run_demo(name: str) -> str:
    """Run the small demonstration for one library and return its result line.

    If the library is missing, return a plain message instead of raising, so the
    overall demo never crashes.
    """
    if not is_available(name):
        return f"{name} is not installed, skipping its demo"
    try:
        return DEMOS[name]()
    except Exception as exc:  # keep the tour going even if one demo misbehaves
        return f"{name} demo failed with {type(exc).__name__}"


if __name__ == "__main__":
    print("Essential Libraries")
    print("=" * 40)

    print("\nThe ten core imports and what each one does")
    for name, desc in library_overview().items():
        mark = "available" if is_available(name) else "missing"
        print(f"  {name:12s} [{mark:9s}] {desc}")

    print("\nA tiny live demonstration of each one")
    for name in library_overview():
        print(f"  {run_demo(name)}")
