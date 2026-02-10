# Kelly Criterion for Position Sizing

This module explains the Kelly Criterion, a formula used to determine the optimal size of a series of bets to maximize the logarithm of wealth. In trading, it helps in managing risk and optimizing position sizes based on the probability of winning and the win/loss ratio.

## Formula
The basic Kelly formula is:
f* = (bp - q) / b

Where:
- f* is the fraction of the current bankroll to wager.
- b is the net odds received on the wager (b to 1).
- p is the probability of winning.
- q is the probability of losing (q = 1 - p).

## Contents
- `kelly_criterion.py`: Implementation of the Kelly Criterion for single and multiple assets.

## Usage
Run the script to see examples of optimal f calculation.
```bash
python kelly_criterion.py
```
