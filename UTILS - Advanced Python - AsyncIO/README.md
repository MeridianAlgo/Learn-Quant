# AsyncIO for High-Frequency Data

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
