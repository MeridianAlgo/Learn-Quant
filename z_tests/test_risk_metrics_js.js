// test_risk_metrics_js.js
// Unit tests for riskMetrics.js, no external test framework required.
// Usage: node z_tests/test_risk_metrics_js.js

'use strict';

const path = require('path');
const {
  mean,
  stdDev,
  toReturns,
  equityCurve,
  annualizedVolatility,
  annualizedReturn,
  maxDrawdown,
  sharpeRatio,
  sortinoRatio,
  calmarRatio,
  historicalVaR,
  conditionalVaR,
  ulcerIndex,
  riskReport,
} = require(path.join(__dirname, '..', 'Risk Metrics - JavaScript', 'riskMetrics.js'));

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

function assertThrows(fn, message) {
  try {
    fn();
    assert(false, `${message} (no error thrown)`);
  } catch (e) {
    assert(true, message);
  }
}

// Basics
console.log('\nBasic statistics tests:');
assertClose(mean([1, 2, 3, 4]), 2.5, 1e-12, 'mean of 1 through 4');
assertClose(stdDev([2, 4, 4, 4, 5, 5, 7, 9]), 2.13809, 1e-4, 'sample standard deviation');
assertClose(stdDev([5, 5, 5, 5]), 0, 1e-12, 'no spread means no deviation');
assertThrows(() => mean([]), 'mean rejects an empty array');
assertThrows(() => stdDev([1]), 'stdDev needs at least two values');

// Returns and equity curve
console.log('\nReturn conversion tests:');
const prices = [100, 110, 99];
const rets = toReturns(prices);
assert(rets.length === prices.length - 1, 'returns array is one shorter than prices');
assertClose(rets[0], 0.10, 1e-12, 'first return is 10 percent');
assertClose(rets[1], -0.10, 1e-12, 'second return is minus 10 percent');
assertThrows(() => toReturns([100]), 'toReturns needs two prices');
assertThrows(() => toReturns([0, 100]), 'toReturns rejects a zero price');

const curve = equityCurve([0.10, -0.10]);
assert(curve.length === 3, 'equity curve is one longer than returns');
assertClose(curve[0], 1, 1e-12, 'equity curve starts at one');
assertClose(curve[2], 0.99, 1e-12, 'up ten then down ten leaves you below par');

// Round trip
const roundTrip = equityCurve(toReturns([100, 110, 99, 120]));
assertClose(roundTrip[roundTrip.length - 1], 1.20, 1e-12, 'prices to returns to curve round trips');

// Annualization
console.log('\nAnnualization tests:');
const flat = new Array(252).fill(0.001);
assertClose(annualizedVolatility(flat), 0, 1e-12, 'a constant return has no volatility');
assertClose(annualizedReturn(flat), 1.001 ** 252 - 1, 1e-9, 'annualized return compounds');
const daily = [0.01, -0.005, 0.007, -0.002, 0.004, 0.001];
assertClose(
  annualizedVolatility(daily),
  stdDev(daily) * Math.sqrt(252),
  1e-12,
  'volatility scales by the square root of time'
);
assertClose(
  annualizedVolatility(daily, 12),
  stdDev(daily) * Math.sqrt(12),
  1e-12,
  'monthly data uses twelve periods'
);

// Drawdown
console.log('\nDrawdown tests:');
assertClose(maxDrawdown([0.1, 0.1, 0.1]).maxDrawdown, 0, 1e-12, 'a rising series never draws down');

const dd = maxDrawdown([0.20, -0.50, 0.10]);
assertClose(dd.maxDrawdown, -0.50, 1e-12, 'a fifty percent fall is a fifty percent drawdown');
assert(dd.peakIndex === 1, 'drawdown peak is at the top of the run up');
assert(dd.troughIndex === 2, 'drawdown trough is at the bottom');
assert(dd.recoveryIndex === -1, 'never recovered reports minus one');

// The dip first then rally case, which high minus low would get wrong.
const dipFirst = maxDrawdown([-0.20, 0.50, 0.50]);
assertClose(dipFirst.maxDrawdown, -0.20, 1e-9, 'drawdown tracks a running peak, not the range');

const recovered = maxDrawdown([0.25, -0.20, 0.25, 0.10]);
assert(recovered.recoveryIndex !== -1, 'a series back above its old high reports a recovery');
assert(recovered.recoveryIndex > recovered.troughIndex, 'recovery comes after the trough');

// Ratios
console.log('\nRisk adjusted ratio tests:');
const mixed = [0.01, -0.008, 0.012, -0.004, 0.006, -0.011, 0.009, 0.002];
assert(sharpeRatio(mixed) > 0, 'a profitable series has a positive Sharpe');
assert(sharpeRatio(mixed, 0.50) < sharpeRatio(mixed, 0), 'a higher risk free rate lowers Sharpe');
assertThrows(() => sharpeRatio(flat), 'a series with no variation has no Sharpe');

assert(sortinoRatio(mixed) > sharpeRatio(mixed), 'Sortino exceeds Sharpe when upside is ignored');
assert(sortinoRatio([0.01, 0.02, 0.03]) === Infinity, 'no downside means an infinite Sortino');

const losses = [-0.01, -0.02, -0.005, -0.015];
assert(sortinoRatio(losses) < 0, 'an all losing series has a negative Sortino');

assert(calmarRatio(mixed) !== 0, 'Calmar is defined for a series with a drawdown');
assert(calmarRatio([0.001, 0.002, 0.001]) === Infinity, 'no drawdown means an infinite Calmar');

// Tail risk
console.log('\nTail risk tests:');
const tail = [-0.10, -0.05, -0.03, -0.01, 0, 0.01, 0.02, 0.03, 0.04, 0.05];
const var90 = historicalVaR(tail, 0.90);
const cvar90 = conditionalVaR(tail, 0.90);
// The quantile convention here is sorted[floor((1 - confidence) * n)], so with
// ten observations the 90 percent VaR is the second worst and not the worst.
assertClose(var90, 0.05, 1e-12, 'VaR at 90 percent picks the tenth percentile of ten');
assert(cvar90 >= var90, 'CVaR is never smaller than VaR');
assertClose(conditionalVaR(tail, 0.80), 0.075, 1e-12, 'CVaR averages the tail beyond the cutoff');
assert(historicalVaR(tail, 0.95) >= historicalVaR(tail, 0.80), 'higher confidence means a larger VaR');
assertThrows(() => historicalVaR(tail, 1.0), 'VaR rejects a confidence of one');
assertThrows(() => conditionalVaR([], 0.95), 'CVaR needs data');

// A series with no losses still reports a number rather than throwing.
assert(historicalVaR([0.01, 0.02, 0.03], 0.95) < 0, 'an all winning series reports a negative VaR');

// Ulcer index
console.log('\nUlcer index tests:');
assertClose(ulcerIndex([0.01, 0.01, 0.01]), 0, 1e-12, 'a series always at its high has no ulcer');
const shallow = ulcerIndex([-0.02, 0.001, 0.001, 0.001, 0.001, 0.001]);
const sharp = ulcerIndex([-0.02, 0.021, 0, 0, 0, 0]);
assert(shallow > sharp, 'a long time underwater scores worse than a quick recovery');

// Report
console.log('\nRisk report tests:');
const report = riskReport(mixed);
const expectedKeys = [
  'annualReturn', 'annualVolatility', 'sharpe', 'sortino', 'calmar',
  'maxDrawdown', 'drawdownLength', 'recovered', 'var95', 'cvar95', 'ulcer',
];
assert(expectedKeys.every(k => k in report), 'report carries every metric');
assert(Object.values(report).every(v => typeof v === 'boolean' || Number.isFinite(v)), 'report has no NaN');
assertClose(report.sharpe, sharpeRatio(mixed), 1e-12, 'report Sharpe matches the direct call');
assertClose(report.maxDrawdown, maxDrawdown(mixed).maxDrawdown, 1e-12, 'report drawdown matches');
assert(report.drawdownLength >= 0, 'drawdown length is never negative');

// Summary
console.log(`\nResults: ${passed} passed, ${failed} failed.`);
if (failed > 0) process.exit(1);
