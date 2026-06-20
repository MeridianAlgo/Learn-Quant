<p class="lq-badges"><span class="lq-badge lq-beginner">Beginner</span><span class="lq-badge lq-cat">Python Fundamentals</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Python Basics - Imports and Modules"
    python "import_basics.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Python%20Basics%20-%20Imports%20and%20Modules)

---
# Python Basics, Imports and Modules

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

## Continue in Python Fundamentals

<div class="grid cards" markdown>

-   :material-language-python: __[Python Basics - Comprehensions](Python Basics - Comprehensions.md)__

    Comprehensions are Python's most elegant way to transform data—replacing loops with readable, performant one-liners. This module teaches **list, dict, set comprehensions**, **generator expressions**, and **functional tools** (`map`, `filter`, `reduce`, `accumulate`) used constantly in quantitative finance for data cleaning, signal generation, and portfolio calculations.

-   :material-language-python: __[Python Basics - Control Flow](Python Basics - Control Flow.md)__

    Control flow structures (`if/elif/else`, `for`, `while`, comprehensions, `break`, `continue`) are the foundation of all algorithms. This module teaches how to make decisions, iterate through data, and build the logic patterns used in trading systems, backtests, and risk management tools.

-   :material-language-python: __[Python Basics - Dates and Times](Python Basics - Dates and Times.md)__

    Markets run on a calendar, not a clock. Interest accrues over **days**, options

-   :material-language-python: __[Python Basics - Essential Libraries](Python Basics - Essential Libraries.md)__

    A working quant leans on a small set of libraries for almost everything. A few of

-   :material-language-python: __[Python Basics - Functions](Python Basics - Functions.md)__

    This utility teaches Python functions - the building blocks of modular, reusable code. Learn to write efficient trading algorithms and financial tools using proper function design.

-   :material-language-python: __[Python Basics - NumPy](Python Basics - NumPy.md)__

    Covers the NumPy primitives that appear in virtually every quant codebase — from vectorised return calculations to portfolio variance via the quadratic form. All examples use realistic financial data so the connection between the NumPy API and actual quant work is immediate.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
