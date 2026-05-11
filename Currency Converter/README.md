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
