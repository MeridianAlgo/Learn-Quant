# Market Microstructure, Liquidity Measures

<p class="lq-meta">Advanced <span>/</span> Market Microstructure <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Market%20Microstructure%20-%20Liquidity%20Measures">source</a></p>


Every strategy pays to trade, and the size of that bill decides whether a
backtest survives contact with a real market. The trouble is that the best
source for trading costs, tick by tick quotes, is expensive, enormous, and
missing for most of history. This module covers the estimators people use
instead, which back out liquidity from daily prices and volume.

## The pieces

* `roll_spread(prices)` reads the bid ask spread from negative
  autocorrelation in price changes.
* `corwin_schultz_spread(high, low)` reads it from daily high and low prices.
* `amihud_illiquidity(returns, dollar_volume)` is absolute return per dollar
  traded.
* `kyle_lambda(price_changes, signed_volume)` is the price impact of order
  flow, from a regression.
* `effective_spread(...)` and `realized_spread(...)` measure the cost of real
  trades when you do have quotes.

## Roll, from the bounce

If the true value follows a random walk and trades land on the bid or the ask
at random, the observed price ticks up and down by half a spread for no
reason. That bounce makes consecutive price changes negatively correlated,
and the algebra works out to

```
cov(dp_t, dp_t-1) = -s^2 / 4      so      s = 2 * sqrt(-cov)
```

The demo simulates 20,000 trades with a five cent spread and recovers it
closely. On daily data the estimate fails often, because momentum or news can
make the autocovariance positive. When that happens the function returns NaN
rather than invent a number.

## Corwin and Schultz, from the range

A day's high is usually a buy at the ask and its low a sell at the bid, so the
high low range is part volatility and part spread. Volatility scales with the
length of the window. The spread does not. Comparing the sum of two single
day squared ranges with the squared range over both days together separates
the two components. It works on any data set that has highs and lows, which
is almost all of them.

## Amihud, price move per dollar

```
ILLIQ = mean of |r_t| / dollar_volume_t
```

No model, just the average size of the daily move divided by how much money
changed hands. It is crude but it ranks stocks by liquidity remarkably well,
and it is the standard liquidity control in academic asset pricing.

## Kyle's lambda, price move per unit of order flow

In Kyle's 1985 model a market maker cannot tell informed orders from noise,
so they move the price in proportion to net buying to protect themselves.
The slope of price change on signed volume is that proportion. A high lambda
means a small order moves the price a lot.

## Effective and realised spreads

With quotes in hand you can measure cost directly. The effective spread
compares each trade to the midquote at the time. The realised spread compares
it to the midquote a few minutes later. The difference between them is price
impact, the part of the cost that came from trading against someone better
informed.

## Example

```python
from liquidity import amihud_illiquidity, effective_spread, roll_spread

prices = [100.02, 99.98, 100.03, 99.99, 100.01, 99.97, 100.04]
print(roll_spread(prices))
print(amihud_illiquidity([0.01, -0.02, 0.015], [1e6, 2e6, 1.5e6]))
print(effective_spread([100.03], [100.0], [1]))
```

## Things to watch

* These estimators are noisy at the level of a single stock and month. They
  are best for ranking or averaging across many names.
* Corwin Schultz is biased by overnight gaps. The paper adjusts the next
  day's range for the close to open move. This version keeps the simple form.
* Amihud depends on price level and currency. Compare like with like.

## Where to go next

* For the order book these costs come from see
  [`Market Microstructure`](Market Microstructure.md).
* For measuring the cost of a whole order see
  [`Finance - Transaction Cost Analysis`](Finance - Transaction Cost Analysis.md).
* For trading an order to keep impact down see
  [`Strategies - Optimal Execution`](Strategies - Optimal Execution.md).
* For being on the other side of the spread see
  [`Strategies - Market Making`](Strategies - Market Making.md).

---

## Running it

```bash
cd "Market Microstructure - Liquidity Measures"
python "liquidity.py"
```

## More in Market Microstructure

- [High Frequency Trading](High Frequency Trading.md): High Frequency Trading (HFT) encompasses algorithmic strategies that execute a large number of orders at extremely high speeds — typically microseconds to milliseconds.
- [Market Microstructure](Market Microstructure.md): Market microstructure studies how trading mechanisms — the rules, protocols, and participants in a market — affect price formation, liquidity, and transaction costs.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
