<p class="lq-badges"><span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-cat">Options, Derivatives & Finance</span><span class="lq-badge lq-lang">Python · JavaScript</span></p>

!!! tip "Run this module"
    ```bash
    cd "Technical Indicators"
    python "technical_indicators.py"
    node "technicalIndicators.js"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Technical%20Indicators)

---
# Technical Indicators Calculator Utility (NO API)

**This utility does NOT use any external APIs.** All calculations are done locally for learning and experimentation.

This tool lets you calculate and visualize common technical indicators (SMA, EMA, RSI, MACD) for any price series. You can input prices manually or load from a CSV file.

## Features
- Calculate Simple Moving Average (SMA)
- Calculate Exponential Moving Average (EMA)
- Calculate Relative Strength Index (RSI)
- Calculate MACD (Moving Average Convergence Divergence)
- Input prices manually or load from a CSV file
- Display indicator values and simple text-based plots
- CLI interface (Python script)
- **Beginner-friendly:** All code is commented for learning

## Requirements
- Python 3.7+
- No external libraries required (uses only Python standard library)

## Setup
1. Copy `technical_indicators.py` to your desired folder.
2. Open a terminal in that folder.

## Usage Workflow (Step-by-Step)
1. Run the script:
   ```sh
   python technical_indicators.py
   ```
2. Follow the menu prompts:
   - Enter prices manually or load from CSV
   - Calculate SMA, EMA, RSI, MACD for chosen window/period
   - View indicator values and simple text plots
   - Exit when done.

**No real market data is used. This is for learning only!**

## Example Session
```
Welcome to the Technical Indicators Calculator!
1. Enter prices manually
2. Load prices from CSV
3. Calculate SMA
4. Calculate EMA
5. Calculate RSI
6. Calculate MACD
7. Exit
Enter your choice: 1
Enter prices separated by commas: 100,101,102,103,104
```

## Learning Notes
- **No API:** All calculations are done in Python, so you can see and modify the math yourself.
- **How does it work?** Each indicator is implemented in the code, with comments explaining each step.
- **How can you extend it?** Try adding new indicators, or plotting the results with matplotlib!

## License
MIT


---

## Continue in Options, Derivatives & Finance

<div class="grid cards" markdown>

-   :material-chart-bell-curve: __[Advanced Options Pricing](Advanced Options Pricing.md)__

    This module covers advanced mathematical techniques for pricing financial derivatives. The focus is on models beyond the standard assumptions. Rather than assuming constant volatility, we explore dynamic and local volatility models. These models are crucial for correctly valuing exotic options and managing the risks of complex derivatives portfolios.

-   :material-chart-bell-curve: __[Black-Scholes Option Pricing](Black-Scholes Option Pricing.md)__

    This module lets you price basic stock options (calls and puts) using the Black-Scholes formula, a foundation of modern financial analysis.

-   :material-chart-bell-curve: __[Bond Price and Yield](Bond Price and Yield.md)__

    This utility lets you calculate the fair price of a bond or estimate its yield to maturity (YTM), two of the most basic (and important!) ideas in investing.

-   :material-chart-bell-curve: __[CAPM](CAPM.md)__

    CAPM is the idea that won a Nobel Prize and still anchors how the industry

-   :material-chart-bell-curve: __[Discounted Cash Flow (DCF)](Discounted Cash Flow (DCF).md)__

    This tool calculates the present value of a series of future cash flows—the basic principle behind valuing businesses, real estate, projects, and stocks!

-   :material-chart-bell-curve: __[Dividend Tracker](Dividend Tracker.md)__

    **This utility does NOT use any external APIs.** All data is managed locally for learning and experimentation.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
