# FX (Foreign Exchange) Tools

Core analytics for foreign exchange markets: no-arbitrage pricing, option valuation, and cross-rate calculations.

## Functions

| Function | Description |
|---|---|
| `forward_rate(spot, r_d, r_f, T)` | CIP-implied forward exchange rate |
| `forward_points(spot, r_d, r_f, T)` | Forward-spot differential in pips |
| `cip_deviation(spot, forward, r_d, r_f, T)` | Covered Interest Parity basis (bps) |
| `cross_rate(s_ab, s_ac)` | Derive B/C rate from two pairs |
| `triangular_arbitrage_profit(s_ab, s_bc, s_ca)` | Detect/quantify triangular arb |
| `garman_kohlhagen(S, K, r_d, r_f, sigma, T, type)` | European FX option pricing |

## Key Concepts

### Covered Interest Rate Parity (CIP)
`F = S × exp((r_d - r_f) × T)`
In theory, no arbitrage → forward rate is fully determined by spot + rate differential. In practice, CIP deviations (the "FX basis") are a significant source of hedge fund alpha.

### Forward Points
Market convention: quote forward as "pips" above/below spot. Forward points = `(F - S) / pip_size`. Positive when domestic rate > foreign rate.

### Garman-Kohlhagen
Black-Scholes extension for FX options. Foreign rate acts as a continuous dividend yield. Delta is expressed in domestic currency terms.

## Example

```python
from fx_tools import forward_rate, garman_kohlhagen, cip_deviation

# USD/EUR spot = 1.10, US rate 5%, EU rate 2%
F = forward_rate(1.10, r_domestic=0.05, r_foreign=0.02, T=1.0)  # ~1.1332

# FX call option
call = garman_kohlhagen(S=1.10, K=1.10, r_d=0.05, r_f=0.02, sigma=0.10, T=0.25)
```

## Pip Conventions

| Pair | Pip size |
|---|---|
| EUR/USD, GBP/USD | 0.0001 |
| USD/JPY | 0.01 |
| Most others | 0.0001 |
