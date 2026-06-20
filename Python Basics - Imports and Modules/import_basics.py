"""
Imports and Modules
-------------------
Almost every useful Python program starts with a few import lines. An import is
how you pull code that lives somewhere else into the file you are working on, so
you do not have to rewrite it. A module is just a Python file, and a package is a
folder of modules. Understanding how Python finds and loads them removes most of
the confusion beginners hit when an import fails.

This lesson demonstrates the main import forms, shows where Python looks for code
through sys.path, explains the role of the __init__.py file in a package, and
gives you small helpers for loading a module safely at runtime when you are not
sure it is installed.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from types import ModuleType
from typing import Optional


def safe_import(name: str) -> Optional[ModuleType]:
    """Return the imported module, or None if it is not installed.

    This is the pattern you use when a feature is optional. Instead of letting a
    missing package crash the whole program, you check for it and fall back.
    """
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def is_installed(name: str) -> bool:
    """Report whether a top level module can be found without importing it.

    find_spec only locates the module, it does not run its code, so this is
    cheaper and safer than a real import when all you want is a yes or no.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def module_origin(name: str) -> str:
    """Return a short description of where a module was loaded from.

    Built in modules report that they are built in. Everything else reports the
    file path of the code that backs it.
    """
    mod = safe_import(name)
    if mod is None:
        return "not found"
    path = getattr(mod, "__file__", None)
    return path if path else "built in module"


def classify(name: str) -> str:
    """Sort a module into standard library, third party, or not found.

    The standard library ships with Python itself. A third party package is one
    you install separately with a tool such as pip.
    """
    if name in sys.builtin_module_names:
        return "standard library"
    if not is_installed(name):
        return "not found"
    origin = module_origin(name)
    lowered = origin.lower().replace("\\", "/")
    if "site-packages" in lowered or "dist-packages" in lowered:
        return "third party"
    return "standard library"


def search_paths() -> list[str]:
    """Return the list of directories Python searches for imports, in order.

    When you write import foo, Python walks this list and uses the first match.
    The current folder usually comes first, which is why a local file can shadow
    an installed package of the same name.
    """
    return list(sys.path)


if __name__ == "__main__":
    print("Imports and Modules")
    print("=" * 40)

    # The plain import form. You reach names through the module object.
    import math

    print(f"math.pi via plain import is {math.pi:.5f}")

    # The from import form pulls a single name straight into your namespace.
    from math import sqrt

    print(f"sqrt(144) via from import is {sqrt(144)}")

    # The as form gives a module or a name a shorter alias.
    import json as js

    print(f"json dumped with an alias gives {js.dumps({'ok': True})}")

    print("\nClassifying a few modules")
    for name in ["math", "json", "numpy", "no_such_module_xyz"]:
        print(f"  {name:20s} {classify(name)}")

    print("\nWhere each module loads from")
    for name in ["math", "os", "numpy"]:
        print(f"  {name:8s} {module_origin(name)}")

    print(f"\nPython searches {len(search_paths())} locations for imports")
    print("The first few are")
    for entry in search_paths()[:4]:
        shown = entry if entry else "(the current folder)"
        print(f"  {shown}")

    print("\nOptional import demo")
    pandas = safe_import("pandas")
    if pandas is None:
        print("  pandas is not installed, so the program keeps running anyway")
    else:
        print(f"  pandas is installed at version {pandas.__version__}")
