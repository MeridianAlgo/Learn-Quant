// technicalIndicators.js
// Pure-JS implementations of common technical indicators for quantitative trading.
// No external dependencies — all math computed from scratch.
// Usage: node technicalIndicators.js

'use strict';

/**
 * Simple Moving Average.
 * @param {number[]} prices - Closing price array
 * @param {number} period - Lookback window
 * @returns {(number|null)[]} SMA array (null where insufficient data)
 */
function sma(prices, period) {
  return prices.map((_, i) => {
    if (i < period - 1) return null;
    const window = prices.slice(i - period + 1, i + 1);
    return window.reduce((s, v) => s + v, 0) / period;
  });
}

/**
 * Exponential Moving Average (seeded with first SMA).
 * @param {number[]} prices
 * @param {number} period
 * @returns {(number|null)[]}
 */
function ema(prices, period) {
  if (prices.length < period) return prices.map(() => null);
  const k = 2 / (period + 1);
  const result = new Array(prices.length).fill(null);
  result[period - 1] = prices.slice(0, period).reduce((s, v) => s + v, 0) / period;
  for (let i = period; i < prices.length; i++) {
    result[i] = prices[i] * k + result[i - 1] * (1 - k);
  }
  return result;
}

/**
 * Relative Strength Index (Wilder smoothing).
 * @param {number[]} prices
 * @param {number} [period=14]
 * @returns {(number|null)[]}
 */
function rsi(prices, period = 14) {
  const result = new Array(prices.length).fill(null);
  if (prices.length <= period) return result;

  const gains = [];
  const losses = [];
  for (let i = 1; i < prices.length; i++) {
    const diff = prices[i] - prices[i - 1];
    gains.push(diff > 0 ? diff : 0);
    losses.push(diff < 0 ? -diff : 0);
  }

  let avgGain = gains.slice(0, period).reduce((s, v) => s + v, 0) / period;
  let avgLoss = losses.slice(0, period).reduce((s, v) => s + v, 0) / period;
  result[period] = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);

  for (let i = period + 1; i < prices.length; i++) {
    avgGain = (avgGain * (period - 1) + gains[i - 1]) / period;
    avgLoss = (avgLoss * (period - 1) + losses[i - 1]) / period;
    result[i] = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);
  }
  return result;
}

/**
 * MACD: line, signal, and histogram.
 * @param {number[]} prices
 * @param {number} [fastPeriod=12]
 * @param {number} [slowPeriod=26]
 * @param {number} [signalPeriod=9]
 * @returns {{ macdLine: (number|null)[], signal: (number|null)[], histogram: (number|null)[] }}
 */
function macd(prices, fastPeriod = 12, slowPeriod = 26, signalPeriod = 9) {
  const fastEma = ema(prices, fastPeriod);
  const slowEma = ema(prices, slowPeriod);
  const macdLine = prices.map((_, i) =>
    fastEma[i] !== null && slowEma[i] !== null ? fastEma[i] - slowEma[i] : null
  );

  // Build a compact array of non-null MACD values to feed into EMA
  const firstValid = macdLine.findIndex(v => v !== null);
  const compactMacd = macdLine.filter(v => v !== null);
  const compactSignal = ema(compactMacd, signalPeriod);

  const signal = new Array(prices.length).fill(null);
  compactSignal.forEach((v, i) => { signal[firstValid + i] = v; });

  const histogram = prices.map((_, i) =>
    macdLine[i] !== null && signal[i] !== null ? macdLine[i] - signal[i] : null
  );

  return { macdLine, signal, histogram };
}

/**
 * Bollinger Bands (SMA ± k * rolling stddev).
 * @param {number[]} prices
 * @param {number} [period=20]
 * @param {number} [k=2]
 * @returns {{ middle: (number|null)[], upper: (number|null)[], lower: (number|null)[] }}
 */
function bollingerBands(prices, period = 20, k = 2) {
  const middle = sma(prices, period);
  const upper = prices.map((_, i) => {
    if (middle[i] === null) return null;
    const window = prices.slice(i - period + 1, i + 1);
    const mean = middle[i];
    const variance = window.reduce((s, v) => s + (v - mean) ** 2, 0) / period;
    return mean + k * Math.sqrt(variance);
  });
  const lower = prices.map((_, i) => {
    if (middle[i] === null) return null;
    const window = prices.slice(i - period + 1, i + 1);
    const mean = middle[i];
    const variance = window.reduce((s, v) => s + (v - mean) ** 2, 0) / period;
    return mean - k * Math.sqrt(variance);
  });
  return { middle, upper, lower };
}

/**
 * Average True Range (Wilder smoothing).
 * @param {number[]} high
 * @param {number[]} low
 * @param {number[]} close
 * @param {number} [period=14]
 * @returns {(number|null)[]}
 */
function atr(high, low, close, period = 14) {
  const n = high.length;
  const tr = high.map((h, i) => {
    if (i === 0) return h - low[i];
    return Math.max(h - low[i], Math.abs(h - close[i - 1]), Math.abs(low[i] - close[i - 1]));
  });

  const result = new Array(n).fill(null);
  if (n < period) return result;
  result[period - 1] = tr.slice(0, period).reduce((s, v) => s + v, 0) / period;
  for (let i = period; i < n; i++) {
    result[i] = (result[i - 1] * (period - 1) + tr[i]) / period;
  }
  return result;
}

// ── Demo ──────────────────────────────────────────────────────────────
if (require.main === module) {
  const prices = [
    44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.15,
    43.61, 44.33, 44.83, 45.10, 45.15, 45.98, 45.60, 46.28,
    46.28, 46.00, 46.03, 46.41, 46.22, 45.64, 46.21, 46.25,
    45.71, 46.45, 45.78, 45.35, 44.03, 44.18,
  ];

  const fmt = v => (v !== null ? v.toFixed(4) : 'N/A');
  const last5 = arr => arr.slice(-5);
  const tag = i => `Day ${prices.length - 5 + i}`;

  console.log('\n' + '='.repeat(52));
  console.log('TECHNICAL INDICATORS DEMO');
  console.log('='.repeat(52));

  console.log('\nSMA(10) — last 5:');
  last5(sma(prices, 10)).forEach((v, i) => console.log(`  ${tag(i)}: ${fmt(v)}`));

  console.log('\nEMA(10) — last 5:');
  last5(ema(prices, 10)).forEach((v, i) => console.log(`  ${tag(i)}: ${fmt(v)}`));

  console.log('\nRSI(14) — last 5:');
  last5(rsi(prices, 14)).forEach((v, i) =>
    console.log(`  ${tag(i)}: ${v !== null ? v.toFixed(2) : 'N/A'}`)
  );

  const { macdLine, signal, histogram } = macd(prices, 12, 26, 9);
  console.log('\nMACD(12,26,9) — last 5:');
  last5(macdLine).forEach((v, i) => {
    const idx = prices.length - 5 + i;
    console.log(
      `  ${tag(i)}: MACD=${fmt(v)}  Signal=${fmt(signal[idx])}  Hist=${fmt(histogram[idx])}`
    );
  });

  const bb = bollingerBands(prices, 20, 2);
  console.log('\nBollinger Bands(20,2) — last 5:');
  last5(bb.upper).forEach((u, i) => {
    const idx = prices.length - 5 + i;
    console.log(
      `  ${tag(i)}: Upper=${fmt(u)}  Mid=${fmt(bb.middle[idx])}  Lower=${fmt(bb.lower[idx])}`
    );
  });

  const high = prices.map(p => p * 1.01);
  const low = prices.map(p => p * 0.99);
  console.log('\nATR(14) — last 5:');
  last5(atr(high, low, prices, 14)).forEach((v, i) =>
    console.log(`  ${tag(i)}: ${fmt(v)}`)
  );

  console.log('\n✓ Technical indicators demo complete.');
}

module.exports = { sma, ema, rsi, macd, bollingerBands, atr };
