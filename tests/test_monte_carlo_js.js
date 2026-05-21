// test_monte_carlo_js.js
// Unit tests for monteCarlo.js — no external test framework required.
// Usage: node tests/test_monte_carlo_js.js

'use strict';

const path = require('path');
const { simulatePath, simulatePortfolio, mcOptionPrice, varCvar, cholesky } = require(
  path.join(__dirname, '..', 'Monte Carlo Simulation - JavaScript', 'monteCarlo.js')
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

// ── cholesky ──────────────────────────────────────────────────────────
console.log('\nCholesky tests:');
// Identity matrix → Cholesky = Identity
const I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
const LI = cholesky(I);
assertClose(LI[0][0], 1, 1e-9, 'Cholesky of I: L[0][0]=1');
assertClose(LI[1][0], 0, 1e-9, 'Cholesky of I: L[1][0]=0');
assertClose(LI[2][2], 1, 1e-9, 'Cholesky of I: L[2][2]=1');

// 2x2 known example: [[4,2],[2,3]] → L=[[2,0],[1,√2]]
const A = [[4, 2], [2, 3]];
const LA = cholesky(A);
assertClose(LA[0][0], 2, 1e-9, 'Cholesky 2x2: L[0][0]=2');
assertClose(LA[1][0], 1, 1e-9, 'Cholesky 2x2: L[1][0]=1');
assertClose(LA[1][1], Math.sqrt(2), 1e-9, 'Cholesky 2x2: L[1][1]=√2');

// ── simulatePath ──────────────────────────────────────────────────────
console.log('\nSimulatePath tests:');
const path1 = simulatePath(100, 0.10, 0.20, 1, 252);
assert(path1.length === 253, 'Path length = steps + 1');
assertClose(path1[0], 100, 1e-9, 'Path starts at S0');
assert(path1.every(v => v > 0), 'All path prices positive (GBM)');

// Zero vol → deterministic drift
const deterministicPath = simulatePath(100, 0.10, 0.0001, 1, 10);
const expectedEnd = 100 * Math.exp(0.10 * 1);
assert(Math.abs(deterministicPath[10] / expectedEnd - 1) < 0.01, 'Near-zero vol path follows drift');

// ── varCvar ───────────────────────────────────────────────────────────
console.log('\nVaR / CVaR tests:');
// Trivial case: all returns = -0.10 → VaR = CVaR = 0.10
const uniformLoss = new Array(1000).fill(-0.10);
const { var: v1, cvar: c1 } = varCvar(uniformLoss, 0.95);
assertClose(v1, 0.10, 1e-9, 'VaR for uniform -10% loss = 0.10');
assertClose(c1, 0.10, 1e-9, 'CVaR for uniform -10% loss = 0.10');

// Positive returns only → VaR negative (profit scenario)
const allPos = new Array(1000).fill(0.05);
const { var: v2 } = varCvar(allPos, 0.95);
assert(v2 <= 0, 'VaR is non-positive when all returns are positive');

// CVaR >= VaR in general
const mixed = Array.from({ length: 1000 }, (_, i) => (i < 500 ? -0.02 : 0.03));
const { var: vMix, cvar: cMix } = varCvar(mixed, 0.95);
assert(cMix >= vMix, 'CVaR >= VaR');

// ── mcOptionPrice (statistical test) ─────────────────────────────────
console.log('\nMC option price tests:');
// Black-Scholes analytical for S=100, K=100, T=1, r=0, sigma=0.2: ~7.966
const { price: mcP, stdErr } = mcOptionPrice(100, 100, 1.0, 0.0, 0.20, 100_000);
const bs = 7.9656; // known analytical value
assert(Math.abs(mcP - bs) < 0.30, `MC ATM call within $0.30 of BS (~${bs})`);
assert(stdErr > 0, 'Standard error is positive');

// Put price via put-call parity (r=0 → call = put for ATM)
const { price: putMC } = mcOptionPrice(100, 100, 1.0, 0.0, 0.20, 100_000);
assert(Math.abs(putMC - bs) < 0.30, 'MC put ≈ call (ATM, r=0, put-call parity)');

// ── simulatePortfolio (statistical smoke test) ────────────────────────
console.log('\nSimulatePortfolio tests:');
const { finalReturns, samplePaths } = simulatePortfolio(
  [100, 200],
  [0.08, 0.10],
  [0.15, 0.20],
  [[1, 0.5], [0.5, 1]],
  [0.6, 0.4],
  1, 52, 2_000
);
assert(finalReturns.length === 2_000, 'Correct number of simulated returns');
assert(samplePaths.length === 5, 'Five sample equity paths recorded');
assert(samplePaths[0].length === 53, 'Sample path length = steps + 1');

// Mean return should be in a reasonable range (roughly E[drift] ~ 8-10% × 1yr)
const mean = finalReturns.reduce((s, v) => s + v, 0) / finalReturns.length;
assert(mean > -0.20 && mean < 0.40, `Mean portfolio return in reasonable range (${(mean * 100).toFixed(1)}%)`);

// ── Summary ───────────────────────────────────────────────────────────
console.log(`\nResults: ${passed} passed, ${failed} failed.`);
if (failed > 0) process.exit(1);
