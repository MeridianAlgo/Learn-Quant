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
