# Python Basics, Imports and Modules

<p class="lq-meta">Beginner <span>/</span> Python Fundamentals <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Python%20Basics%20-%20Imports%20and%20Modules">source</a></p>


Almost every Python program begins with a few import lines. An import is how you
bring in code that lives somewhere else so you do not have to write it yourself.
A module is simply a Python file. A package is a folder of modules. Once you see
how Python finds and loads them, most beginner confusion around a failing import
disappears.

This lesson shows the main import forms, explains where Python searches for code,
describes what a package and its __init__ file are for, and gives you a safe way
to load a module at runtime when you are not sure it is installed.

## The main import forms

There are three forms you will use all the time.

* The plain form `import math` loads the whole module and you reach names through
  it as `math.pi`.
* The from form `from math import sqrt` pulls one name straight into your file so
  you can write `sqrt` on its own.
* The alias form `import numpy as np` loads a module under a shorter name, which
  is why you see `np` and `pd` everywhere in quant code.

```python
import math
from math import sqrt
import json as js

print(math.pi)
print(sqrt(144))
print(js.dumps({"ok": True}))
```

## Modules and packages

A module is one Python file. When you write `import helpers` Python runs that
file once and hands you back a module object. A package is a folder that groups
related modules together. Older packages contain a file named __init__ that
marks the folder as a package and runs when the package is first imported. You
import inside a package using dotted names such as `import scipy.stats`.

## Where Python looks

When you import something, Python walks an ordered list of folders and uses the
first match it finds. That list is `sys.path`. The current working folder
usually sits near the front, which is why a local file can accidentally hide an
installed package that shares its name. If an import fails with a not found
error, checking `sys.path` is the first thing to do.

## Standard library, third party, and local

Imports come from three places.

* The standard library ships with Python itself. Modules such as math, json,
  datetime, and random are always available.
* Third party packages are ones you install separately with a tool such as pip.
  numpy and pandas are examples. They live in a folder named site packages.
* Local modules are your own files sitting next to the script you are running.

## Importing safely at runtime

Sometimes a feature depends on a package that may or may not be present. Rather
than letting a missing package crash everything, you can try to import it and
fall back when it is absent. The helper `safe_import` in this module returns the
module when it loads and returns None when it does not, so your program can keep
going.

```python
from import_basics import safe_import

pandas = safe_import("pandas")
if pandas is None:
    print("pandas is missing, using a simpler path")
else:
    print("pandas is ready")
```

## Functions in this module

* `safe_import(name)` returns the module or None if it cannot be loaded.
* `is_installed(name)` reports whether a module can be found without running it.
* `module_origin(name)` reports the file a module was loaded from.
* `classify(name)` sorts a module into standard library, third party, or not found.
* `search_paths()` returns the folders Python searches, in order.

## Where to go next

Now that you know how imports work, the companion lesson in the folder named
Python Basics, Essential Libraries walks through the ten libraries a quant
reaches for most often and what each one does.

---

## Running it

```bash
cd "Python Basics - Imports and Modules"
python "import_basics.py"
```

## More in Python Fundamentals

- [Python Basics - Comprehensions](Python Basics - Comprehensions.md): Comprehensions are Python's most elegant way to transform data—replacing loops with readable, performant one-liners.
- [Python Basics - Control Flow](Python Basics - Control Flow.md): Control flow structures (if/elif/else, for, while, comprehensions, break, continue) are the foundation of all algorithms.
- [Python Basics - Dates and Times](Python Basics - Dates and Times.md): Markets run on a calendar, not a clock. Interest accrues over days, options expire on a date, and any backtest that miscounts trading days will quietly misreport every annualised number it produces.
- [Python Basics - Essential Libraries](Python Basics - Essential Libraries.md): A working quant leans on a small set of libraries for almost everything.
- [Python Basics - Functions](Python Basics - Functions.md): This utility teaches Python functions - the building blocks of modular, reusable code.
- [Python Basics - NumPy](Python Basics - NumPy.md): Covers the NumPy primitives that appear in virtually every quant codebase — from vectorised return calculations to portfolio variance via the quadratic form.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
