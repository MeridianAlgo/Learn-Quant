<p class="lq-badges"><span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-cat">Options, Derivatives & Finance</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Dividend Tracker"
    python "dividend_tracker.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Dividend%20Tracker)

---
# Dividend Tracker Utility (NO API)

**This utility does NOT use any external APIs.** All data is managed locally for learning and experimentation.

This tool lets you track upcoming dividends for a list of stocks, calculate expected income, and view a dividend calendar. All data is entered manually for learning purposes.

## Features
- Add, edit, and remove dividend entries (ticker, ex-date, pay date, amount, shares)
- View upcoming dividends in a calendar format
- Calculate total expected dividend income
- CLI interface (Python script)
- **Beginner-friendly:** All code is commented for learning

## Requirements
- Python 3.7+
- No external libraries required (uses only Python standard library)

## Setup
1. Copy `dividend_tracker.py` to your desired folder.
2. Open a terminal in that folder.

## Usage Workflow (Step-by-Step)
1. Run the script:
   ```sh
   python dividend_tracker.py
   ```
2. Follow the menu prompts:
   - Add, edit, or remove dividend entries
   - View the dividend calendar
   - Calculate total expected income
   - Exit when done.

**No real market data is used. This is for learning only!**

## Example Session
```
Welcome to the Dividend Tracker!
1. Add dividend
2. Edit dividend
3. Remove dividend
4. View calendar
5. Calculate total income
6. Exit
Enter your choice: 1
Enter ticker: AAPL
Enter ex-date (YYYY-MM-DD): 2024-08-01
Enter pay date (YYYY-MM-DD): 2024-08-15
Enter dividend amount per share: 0.24
Enter number of shares: 50
Dividend added!
```

## Learning Notes
- **No API:** All data is managed in Python, so you can see and modify the logic yourself.
- **How does it work?** The code is structured with classes and functions, with comments explaining each step.
- **How can you extend it?** Try adding support for dividend reinvestment, or plotting your income over time!

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

-   :material-chart-bell-curve: __[Finance - Beta Calculator](Finance - Beta Calculator.md)__

    **Beta** measures how much a stock or portfolio moves compared to the overall market.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
