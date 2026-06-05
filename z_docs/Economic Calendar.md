<p class="lq-badges"><span class="lq-badge lq-beginner">Beginner</span><span class="lq-badge lq-cat">Utilities & Tools</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Economic Calendar"
    python "economic_calendar.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Economic%20Calendar)

---
# Economic Calendar Simulator Utility (NO API)

**This utility does NOT use any external APIs.** All data is managed locally for learning and experimentation.

This tool lets you simulate and explore economic events (like FOMC meetings, CPI releases, etc.) and see how they might impact markets. You can add, edit, and view events, and simulate their effects on a sample portfolio.

## Features
- Add, edit, and remove economic events (date, type, description, expected impact)
- View upcoming events in a calendar format
- Simulate the effect of events on a sample portfolio (manual input)
- CLI interface (Python script)
- **Beginner-friendly:** All code is commented for learning

## Requirements
- Python 3.7+
- No external libraries required (uses only Python standard library)

## Setup
1. Copy `economic_calendar.py` to your desired folder.
2. Open a terminal in that folder.

## Usage Workflow (Step-by-Step)
1. Run the script:
   ```sh
   python economic_calendar.py
   ```
2. Follow the menu prompts:
   - Add, edit, or remove economic events
   - View the calendar of upcoming events
   - Simulate event impact on a sample portfolio
   - Exit when done.

**No real market data is used. This is for learning only!**

## Example Session
```
Welcome to the Economic Calendar Simulator!
1. Add event
2. Edit event
3. Remove event
4. View calendar
5. Simulate event impact
6. Exit
Enter your choice: 1
Enter event date (YYYY-MM-DD): 2024-07-15
Enter event type: FOMC Meeting
Enter description: Federal Reserve interest rate decision
Enter expected impact: High
Event added!
```

## Learning Notes
- **No API:** All data is managed in Python, so you can see and modify the logic yourself.
- **How does it work?** The code is structured with classes and functions, with comments explaining each step.
- **How can you extend it?** Try adding recurring events, or linking to real news headlines!

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

-   :material-tools: __[Historical Data](Historical Data.md)__

    A Node.js script that fetches historical bars (OHLCV data) for stocks or crypto from the Alpaca Market Data API. It prompts interactively for the symbol type, symbol, timeframe, and date range, then prints the results as JSON.

-   :material-tools: __[Logging](Logging.md)__

    A pair of minimal, dependency-light logging utilities implemented in both Python and JavaScript. Each supports adding, reading, editing, and deleting log entries through an interactive command-line menu. All entries are persisted to a plain-text `log.txt` file in the working directory.

-   :material-tools: __[Market Data](Market Data.md)__

    This folder contains utilities for processing, analyzing, and fetching market data for financial applications.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
