// test_technical_indicators_js.js
// Unit tests for technicalIndicators.js — no external test framework required.
// Usage: node tests/test_technical_indicators_js.js

'use strict';

const path = require('path');
const { sma, ema, rsi, macd, bollingerBands, atr } = require(
  path.join(__dirname, '..', 'Technical Indicators', 'technicalIndicators.js')
);

let passed = 0;
let failed = 0;

function assert(condition, message) {
  if (condition) {
    console.log(`  PASS: ${message}`);
    passed++;
  } else {
    console.error(`  FAIL: ${message}`);
    failed++;
  }
}

function assertClose(a, b, tol, message) {
  assert(Math.abs(a - b) < tol, `${message} (got ${a}, expected ~${b})`);
}

// ── SMA ──────────────────────────────────────────────────────────────
console.log('\nSMA tests:');
const prices5 = [1, 2, 3, 4, 5];
const sma3 = sma(prices5, 3);
assert(sma3[0] === null && sma3[1] === null, 'SMA nulls for insufficient data');
assertClose(sma3[2], 2.0, 1e-9, 'SMA(3) at index 2');
assertClose(sma3[4], 4.0, 1e-9, 'SMA(3) at index 4');

// ── EMA ──────────────────────────────────────────────────────────────
console.log('\nEMA tests:');
const ema3 = ema(prices5, 3);
assert(ema3[0] === null && ema3[1] === null, 'EMA nulls for insufficient data');
assert(ema3[2] !== null, 'EMA first valid value exists');
// Seed is SMA(1,2,3)=2; next: 4*0.5 + 2*0.5 = 3
assertClose(ema3[3], 3.0, 1e-9, 'EMA(3) at index 3');

// ── RSI ──────────────────────────────────────────────────────────────
console.log('\nRSI tests:');
// All-up prices → RSI should be 100
const allUp = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
const rsi14 = rsi(allUp, 14);
assertClose(rsi14[14], 100, 1e-6, 'RSI=100 when all prices up');

// All-down prices → RSI should be 0
const allDown = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
const rsiDown = rsi(allDown, 14);
assertClose(rsiDown[14], 0, 1e-6, 'RSI=0 when all prices down');

// RSI values in range [0, 100]
const mixed = [44, 44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10,
               45.15, 43.61, 44.33, 44.83, 45.10, 45.15, 45.98, 46.00];
const rsiMixed = rsi(mixed, 14);
const rsiValues = rsiMixed.filter(v => v !== null);
assert(rsiValues.every(v => v >= 0 && v <= 100), 'All RSI values in [0, 100]');

// ── MACD ─────────────────────────────────────────────────────────────
console.log('\nMACD tests:');
const longPrices = Array.from({ length: 40 }, (_, i) => 100 + i * 0.5 + Math.sin(i) * 2);
const { macdLine, signal, histogram } = macd(longPrices, 12, 26, 9);
const validMacd = macdLine.filter(v => v !== null);
assert(validMacd.length > 0, 'MACD has some valid values');
const validHist = histogram.filter(v => v !== null);
assert(validHist.length > 0, 'Histogram has some valid values');

// ── Bollinger Bands ───────────────────────────────────────────────────
console.log('\nBollinger Bands tests:');
const flat = new Array(25).fill(100);
const bb = bollingerBands(flat, 20, 2);
// Flat series → upper = lower = middle = 100
assertClose(bb.middle[24], 100, 1e-9, 'Bollinger middle = 100 for flat series');
assertClose(bb.upper[24], 100, 1e-9, 'Bollinger upper = 100 for flat series (zero vol)');
assertClose(bb.lower[24], 100, 1e-9, 'Bollinger lower = 100 for flat series (zero vol)');

// Upper must be >= lower for general series
const general = [44, 44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10,
                 45.15, 43.61, 44.33, 44.83, 45.10, 45.15, 45.98,
                 46.00, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64];
const bbG = bollingerBands(general, 20, 2);
const validUpper = bbG.upper.filter(v => v !== null);
const validLower = bbG.lower.filter(v => v !== null);
assert(
  validUpper.every((u, i) => u >= validLower[i]),
  'Bollinger upper >= lower for all valid values'
);

// ── ATR ──────────────────────────────────────────────────────────────
console.log('\nATR tests:');
const n = 20;
const highArr = Array.from({ length: n }, (_, i) => 100 + i + 1);
const lowArr  = Array.from({ length: n }, (_, i) => 100 + i - 1);
const closeArr = Array.from({ length: n }, (_, i) => 100 + i);
const atr14 = atr(highArr, lowArr, closeArr, 14);
assert(atr14[13] !== null, 'ATR first valid value is not null');
assert(atr14.slice(0, 13).every(v => v === null), 'ATR nulls before period');
assert(atr14[13] > 0, 'ATR is positive');

// ── Summary ───────────────────────────────────────────────────────────
console.log(`\nResults: ${passed} passed, ${failed} failed.`);
if (failed > 0) process.exit(1);
