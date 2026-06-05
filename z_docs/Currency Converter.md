<p class="lq-badges"><span class="lq-badge lq-beginner">Beginner</span><span class="lq-badge lq-cat">Utilities & Tools</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Currency Converter"
    python "currency_converter.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Currency%20Converter)

---
# Currency Converter Utility (NO API)

**This utility does NOT use any external APIs.** All exchange rates are entered manually for learning and experimentation.

This tool lets you convert between currencies using user-supplied exchange rates. You can add, edit, and view rates, and perform conversions between any two currencies.

## Features
- Add, edit, and remove exchange rates (currency pairs and rates)
- Convert any amount between two currencies
- View all stored rates
- CLI interface (Python script)
- **Beginner-friendly:** All code is commented for learning

## Requirements
- Python 3.7+
- No external libraries required (uses only Python standard library)

## Setup
1. Copy `currency_converter.py` to your desired folder.
2. Open a terminal in that folder.

## Usage Workflow (Step-by-Step)
1. Run the script:
   ```sh
   python currency_converter.py
   ```
2. Follow the menu prompts:
   - Add, edit, or remove exchange rates
   - Convert between currencies
   - View all rates
   - Exit when done.

**No real market data is used. This is for learning only!**

## Example Session
```
Welcome to the Currency Converter!
1. Add rate
2. Edit rate
3. Remove rate
4. Convert currency
5. View all rates
6. Exit
Enter your choice: 1
Enter base currency (e.g., USD): USD
Enter quote currency (e.g., EUR): EUR
Enter exchange rate (1 USD = ? EUR): 0.92
Rate added!
```

## Learning Notes
- **No API:** All rates are managed in Python, so you can see and modify the logic yourself.
- **How does it work?** The code is structured with classes and functions, with comments explaining each step.
- **How can you extend it?** Try adding support for historical rates, or plotting exchange rate trends!

## License
MIT


---

## Continue in Utilities & Tools

<div class="grid cards" markdown>

-   :material-tools: __[Core Utilities](Core Utilities.md)__

    This folder contains core mathematical and date/time utilities that form the foundation for quantitative finance calculations.

-   :material-tools: __[Data Processing](Data Processing.md)__

    This folder contains utilities for data processing, validation, and manipulation in financial applications.

-   :material-tools: __[Economic Calendar](Economic Calendar.md)__

    **This utility does NOT use any external APIs.** All data is managed locally for learning and experimentation.

-   :material-tools: __[Historical Data](Historical Data.md)__

    A Node.js script that fetches historical bars (OHLCV data) for stocks or crypto from the Alpaca Market Data API. It prompts interactively for the symbol type, symbol, timeframe, and date range, then prints the results as JSON.

-   :material-tools: __[Logging](Logging.md)__

    A pair of minimal, dependency-light logging utilities implemented in both Python and JavaScript. Each supports adding, reading, editing, and deleting log entries through an interactive command-line menu. All entries are persisted to a plain-text `log.txt` file in the working directory.

-   :material-tools: __[Market Data](Market Data.md)__

    This folder contains utilities for processing, analyzing, and fetching market data for financial applications.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
