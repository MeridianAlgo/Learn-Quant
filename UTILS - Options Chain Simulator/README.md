# Options Chain Simulator Utility (NO API)

**This utility does NOT use any external APIs.** All calculations are done locally for learning and experimentation.

This tool lets you simulate and explore options chains for a given stock. You can input a stock price, generate call/put options at various strikes and expiries, and see payoff diagrams and basic Greeks (Delta, Gamma, etc.).

## Features
- Simulate call and put options for a user-defined stock price
- Choose strike prices, expiry dates, and volatility
- Calculate option prices using Black-Scholes (for European options)
- Display payoff diagrams for options and simple strategies
- Calculate and display basic Greeks (Delta, Gamma, Theta, Vega, Rho)
- CLI interface (Python script)
- **Beginner-friendly:** All code is commented for learning

## Requirements
- Python 3.7+
- No external libraries required (uses only Python standard library and math)

## Setup
1. Copy `options_chain_simulator.py` to your desired folder.
2. Open a terminal in that folder.

## Usage Workflow (Step-by-Step)
1. Run the script:
   ```sh
   python options_chain_simulator.py
   ```
2. Follow the menu prompts:
   - Set your stock price, volatility, risk-free rate, and expiry.
   - Generate an options chain for different strikes.
   - View payoff diagrams and Greeks for any option.
   - Exit when done.

**No real market data is used. This is for learning only!**

## Example Session
```
Welcome to the Options Chain Simulator!
Enter stock price: 100
Enter volatility (e.g. 0.2 for 20%): 0.2
Enter risk-free rate (e.g. 0.05 for 5%): 0.05
Enter expiry in years (e.g. 0.5): 0.5
Available strikes: 80, 90, 100, 110, 120
Select option type (call/put): call
Option price: 7.96
Delta: 0.6368
...etc.
```

## Learning Notes
- **No API:** All calculations are done in Python, so you can see and modify the math yourself.
- **How does it work?** The Black-Scholes formula is implemented in the code, with comments explaining each step.
- **How can you extend it?** Try adding American options, or plotting the payoff diagram with matplotlib!

## License
MIT
