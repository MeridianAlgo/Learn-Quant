# Binomial Tree Option Pricing

## Overview
This module demonstrates how to implement a binomial tree option pricing model in Python. The binomial tree model provides a discrete-time approach to valuing options, offering more flexibility than the continuous-time Black-Scholes model, especially for American-style options which can be exercised before maturity.

## What You'll Learn
- The Cox-Ross-Rubinstein (CRR) approach to option pricing.
- How to construct forward and backward induction steps.
- The difference in valuing European vs. American options.
- Visualizing the convergence of binomial tree pricing as time steps increase.

## Running the Code
Run the demonstration script:
```bash
python binomial_tree.py
```
This script computes both European and American call/put prices and generates a convergence plot `binomial_convergence.png`.
