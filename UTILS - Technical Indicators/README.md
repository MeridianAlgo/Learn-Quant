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
