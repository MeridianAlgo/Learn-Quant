# Python Basics, Essential Libraries

<p class="lq-meta">Beginner <span>/</span> Python Fundamentals <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Python%20Basics%20-%20Essential%20Libraries">source</a></p>


A working quant leans on a small set of libraries for almost everything. A few of
them ship with Python and are always available. The rest are third party packages
that you install once and then reuse on every project. Learning what each one is
for, and reaching for the right one by reflex, is a big part of being fast and
comfortable in Python.

This lesson walks through the ten imports you will meet most often and explains
what each one does and where a quant tends to use it.

## The third party packages

These six are installed separately. They form the scientific stack that the rest
of this repo is built on.

* **numpy** gives you fast numerical arrays and vectorized math. It is the
  foundation almost everything else sits on, and it is where you do raw number
  crunching on returns and prices.
* **pandas** gives you labeled tables and time series. You use it to load market
  data, line it up by date, and shape it before any analysis.
* **matplotlib** turns numbers into charts. The common entry point is its pyplot
  interface, which you usually import as plt.
* **scipy** covers scientific computing such as statistics, optimization, and
  integration. Option pricing and curve fitting reach for it often.
* **sklearn** is the import name for scikit learn, which provides machine learning
  models for regression, classification, and clustering.
* **statsmodels** provides statistical models and econometrics with detailed
  result summaries, which makes it the natural home for regression diagnostics.

## The standard library imports

These four come with Python itself. You never install them and they are always
present.

* **datetime** handles dates, times, and durations, which matters for settlement
  and for indexing time series.
* **math** holds basic math functions and constants such as sqrt and pi.
* **random** generates random numbers, useful for quick simulations and shuffles.
* **json** reads and writes JSON text, which is the format most web data arrives in.

## A short example

```python
import numpy as np
import pandas as pd
from datetime import date

prices = np.array([100.0, 101.5, 99.8])
frame = pd.DataFrame({"price": prices})
print(frame.describe())
print(date.today())
```

## Functions in this module

* `library_overview()` returns a dict of the ten import names mapped to a plain
  description of each.
* `is_available(name)` reports whether an import can be loaded on this machine.
* `run_demo(name)` runs a tiny demonstration for one library and returns the
  result as text, and it never crashes when a library is missing.

## Where to go next

If you want to understand the import statements themselves, the import forms, the
search path, and how to load a module safely, read the companion lesson in the
folder named Python Basics, Imports and Modules.

---

## Running it

```bash
cd "Python Basics - Essential Libraries"
python "essential_libraries.py"
```

## More in Python Fundamentals

- [Python Basics - Comprehensions](Python Basics - Comprehensions.md): Comprehensions are Python's most elegant way to transform data—replacing loops with readable, performant one-liners.
- [Python Basics - Control Flow](Python Basics - Control Flow.md): Control flow structures (if/elif/else, for, while, comprehensions, break, continue) are the foundation of all algorithms.
- [Python Basics - Dates and Times](Python Basics - Dates and Times.md): Markets run on a calendar, not a clock. Interest accrues over days, options expire on a date, and any backtest that miscounts trading days will quietly misreport every annualised number it produces.
- [Python Basics - Functions](Python Basics - Functions.md): This utility teaches Python functions - the building blocks of modular, reusable code.
- [Python Basics - Imports and Modules](Python Basics - Imports and Modules.md): Almost every Python program begins with a few import lines.
- [Python Basics - NumPy](Python Basics - NumPy.md): Covers the NumPy primitives that appear in virtually every quant codebase — from vectorised return calculations to portfolio variance via the quadratic form.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
