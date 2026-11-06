# AsyncIO for High-Frequency Data

<p class="lq-meta">Intermediate <span>/</span> Advanced Python <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Advanced%20Python%20-%20AsyncIO">source</a></p>


## Overview
In quantitative finance, speed is edge. Python's `asyncio` library allows for **concurrency**, letting your program handle multiple tasks (like fetching data from 10 different exchanges) at once, rather than waiting for one to finish before starting the next.

## Why Async?
- **Sync (Blocking)**: Request 1 (wait 1s) -> Request 2 (wait 1s) = 2s total.
- **Async (Non-blocking)**: Request 1 (start) -> Request 2 (start) -> Wait for both = ~1s total.

## Usage
Run the script to compare the execution speed/flow.

```bash
python async_fetching.py
```

---

## Running it

```bash
cd "Advanced Python - AsyncIO"
python "async_fetching.py"
```

## More in Advanced Python

- [Advanced Python - Context Managers](Advanced Python - Context Managers.md): Context Managers are a powerful Python feature for resource management.
- [Advanced Python - Decorators and Generators](Advanced Python - Decorators and Generators.md): Decorators and Generators are powerful Python features that separate professional code from beginner scripts.
- [Advanced Python - Error Handling](Advanced Python - Error Handling.md): Robust error handling is what separates a script that crashes overnight from a professional trading system that runs for years.
- [Advanced Python - Multiprocessing](Advanced Python - Multiprocessing.md): Python Global Interpreter Lock prevents multiple threads from executing Python bytecode at the same time.
- [Advanced Python - OOP](Advanced Python - OOP.md): Object-Oriented Programming (OOP) is essential for building scalable, maintainable trading systems and financial applications.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
