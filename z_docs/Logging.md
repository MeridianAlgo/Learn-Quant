<p class="lq-badges"><span class="lq-badge lq-beginner">Beginner</span><span class="lq-badge lq-cat">Utilities & Tools</span><span class="lq-badge lq-lang">Python · JavaScript</span></p>

!!! tip "Run this module"
    ```bash
    cd "Logging"
    python "logger.py"
    node "logger.js"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Logging)

---
# Logging Utilities

A pair of minimal, dependency-light logging utilities implemented in both Python and JavaScript. Each supports adding, reading, editing, and deleting log entries through an interactive command-line menu. All entries are persisted to a plain-text `log.txt` file in the working directory.

No external APIs or network access are used — logging is entirely local, making this module suitable for offline study of file I/O and CRUD patterns.

## Files

| File | Description |
|---|---|
| `logger.py` | Python implementation (standard library only) |
| `logger.js` | Node.js implementation (uses `readline-sync` for the CLI) |
| `log.txt` | Shared log store, created on first write |

## Requirements

- **Python**: 3.x — no third-party packages required.
- **Node.js**: any LTS release, plus the `readline-sync` package:
  ```sh
  npm install readline-sync
  ```

## Usage

**Python**

```sh
python logger.py
```

**Node.js**

```sh
node logger.js
```

Either entry point presents the same interactive menu:

- Add a log entry
- Read all log entries
- Edit a log entry
- Delete a log entry
- Exit

## Notes

- Both implementations default to the same `log.txt` file, so they can be used interchangeably within one directory and will operate on a shared log.
- Each function is documented inline to illustrate file handling and basic CRUD operations.

## License

MIT


---

## Continue in Utilities & Tools

<div class="grid cards" markdown>

-   :material-tools: __[Core Utilities](Core Utilities.md)__

    This folder contains core mathematical and date/time utilities that form the foundation for quantitative finance calculations.

-   :material-tools: __[Currency Converter](Currency Converter.md)__

    **This utility does NOT use any external APIs.** All exchange rates are entered manually for learning and experimentation.

-   :material-tools: __[Data Processing](Data Processing.md)__

    This folder contains utilities for data processing, validation, and manipulation in financial applications.

-   :material-tools: __[Economic Calendar](Economic Calendar.md)__

    **This utility does NOT use any external APIs.** All data is managed locally for learning and experimentation.

-   :material-tools: __[Historical Data](Historical Data.md)__

    A Node.js script that fetches historical bars (OHLCV data) for stocks or crypto from the Alpaca Market Data API. It prompts interactively for the symbol type, symbol, timeframe, and date range, then prints the results as JSON.

-   :material-tools: __[Market Data](Market Data.md)__

    This folder contains utilities for processing, analyzing, and fetching market data for financial applications.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
