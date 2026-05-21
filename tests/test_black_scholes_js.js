// test_black_scholes_js.js
// Unit tests for blackScholes.js — no external test framework required.
// Usage: node tests/test_black_scholes_js.js

'use strict';

const path = require('path');
const { price, greeks, impliedVol, normCdf } = require(
  path.join(__dirname, '..', 'Options Pricing - JavaScript', 'blackScholes.js')
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
  assert(Math.abs(a - b) < tol, `${message} (got ${a.toFixed(6)}, expected ~${b})`);
}

// ── normCdf ───────────────────────────────────────────────────────────
console.log('\nnormCdf tests:');
assertClose(normCdf(0), 0.5, 1e-6, 'normCdf(0) = 0.5');
assertClose(normCdf(1.96), 0.975, 5e-4, 'normCdf(1.96) ≈ 0.975');
assertClose(normCdf(-1.96), 0.025, 5e-4, 'normCdf(-1.96) ≈ 0.025');
assertClose(normCdf(10), 1.0, 1e-6, 'normCdf(10) ≈ 1');
assertClose(normCdf(-10), 0.0, 1e-6, 'normCdf(-10) ≈ 0');

// ── price — known values ──────────────────────────────────────────────
console.log('\nBS price tests:');
// ATM call: S=K=100, T=1, r=0, sigma=0.2 → ~7.9656 (analytical)
const atmCall = price(100, 100, 1, 0, 0.20, 'call');
assertClose(atmCall, 7.9656, 0.01, 'ATM call price (r=0)');

// Deep ITM call should be close to intrinsic S - K*exp(-rT)
const itmCall = price(150, 100, 1, 0.05, 0.20, 'call');
assert(itmCall > 45, 'Deep ITM call > intrinsic lower bound');

// Deep OTM call should be close to 0
const otmCall = price(50, 100, 1, 0.05, 0.20, 'call');
assert(otmCall < 0.01, 'Deep OTM call is near zero');

// At expiry: intrinsic only
assertClose(price(110, 100, 0, 0.05, 0.20, 'call'), 10, 1e-9, 'Call at expiry: intrinsic');
assertClose(price(90, 100, 0, 0.05, 0.20, 'put'), 10, 1e-9, 'Put at expiry: intrinsic');
assertClose(price(90, 100, 0, 0.05, 0.20, 'call'), 0, 1e-9, 'OTM call at expiry: 0');

// ── Put-call parity ───────────────────────────────────────────────────
console.log('\nPut-call parity tests:');
for (const [S, K, T, r, sigma] of [
  [100, 105, 0.5, 0.05, 0.20],
  [80, 100, 1.0, 0.03, 0.30],
  [120, 100, 0.25, 0.06, 0.15],
]) {
  const callP = price(S, K, T, r, sigma, 'call');
  const putP = price(S, K, T, r, sigma, 'put');
  const parity = callP - putP - (S - K * Math.exp(-r * T));
  assertClose(parity, 0, 1e-6, `Put-call parity (S=${S}, K=${K})`);
}

// ── greeks ────────────────────────────────────────────────────────────
console.log('\nGreeks tests:');
const g = greeks(100, 100, 1, 0.05, 0.20, 'call');
assert(g.delta > 0.5 && g.delta < 1.0, 'ATM call delta in (0.5, 1)');
assert(g.gamma > 0, 'Gamma is positive');
assert(g.theta < 0, 'Call theta is negative (time decay)');
assert(g.vega > 0, 'Vega is positive');
assert(g.rho > 0, 'Call rho is positive');

const gPut = greeks(100, 100, 1, 0.05, 0.20, 'put');
assert(gPut.delta < 0 && gPut.delta > -0.5, 'ATM put delta in (-0.5, 0)');
assertClose(g.gamma, gPut.gamma, 1e-9, 'Call and put gamma are equal');

// ── Implied volatility ────────────────────────────────────────────────
console.log('\nImplied volatility tests:');
for (const sigma of [0.10, 0.20, 0.35, 0.50]) {
  const S = 100, K = 105, T = 0.5, r = 0.05;
  const mktPrice = price(S, K, T, r, sigma, 'call');
  const iv = impliedVol(mktPrice, S, K, T, r, 'call');
  assertClose(iv, sigma, 1e-4, `IV recovery for sigma=${sigma}`);
}

// ── Summary ───────────────────────────────────────────────────────────
console.log(`\nResults: ${passed} passed, ${failed} failed.`);
if (failed > 0) process.exit(1);
