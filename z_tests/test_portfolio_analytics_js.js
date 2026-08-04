// test_portfolio_analytics_js.js
// Unit tests for portfolioAnalytics.js, no external test framework required.
// Usage: node z_tests/test_portfolio_analytics_js.js

'use strict';

const path = require('path');
const {
  covarianceMatrix,
  correlationMatrix,
  portfolioReturn,
  portfolioVariance,
  portfolioVolatility,
  solve,
  minVarianceWeights,
  tangencyWeights,
  riskContributions,
  diversificationRatio,
} = require(path.join(__dirname, '..', 'Portfolio Analytics - JavaScript', 'portfolioAnalytics.js'));

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

// A fixed three asset covariance matrix used throughout.
const vols = [0.20, 0.12, 0.30];
const corr = [
  [1.00, 0.25, -0.15],
  [0.25, 1.00, 0.05],
  [-0.15, 0.05, 1.00],
];
const cov = corr.map((row, i) => row.map((c, j) => c * vols[i] * vols[j]));

// Covariance
console.log('\nCovariance tests:');
const sample = [[0.01, 0.02], [-0.01, -0.02], [0.02, 0.04], [0.00, 0.00]];
const sampleCov = covarianceMatrix(sample);
assert(sampleCov.length === 2 && sampleCov[0].length === 2, 'covariance matrix is n by n');
assertClose(sampleCov[0][1], sampleCov[1][0], 1e-15, 'covariance matrix is symmetric');
assert(sampleCov[0][0] > 0 && sampleCov[1][1] > 0, 'variances on the diagonal are positive');
// The second column is exactly twice the first, so the covariance must be too.
assertClose(sampleCov[0][1], 2 * sampleCov[0][0], 1e-15, 'a doubled series doubles the covariance');
assertThrows(() => covarianceMatrix([[0.01, 0.02]]), 'covariance needs at least two periods');
assertThrows(() => covarianceMatrix([[0.01, 0.02], [0.01]]), 'covariance rejects ragged rows');

console.log('\nCorrelation tests:');
const rho = correlationMatrix(cov);
assertClose(rho[0][0], 1, 1e-12, 'an asset correlates perfectly with itself');
assertClose(rho[0][1], 0.25, 1e-12, 'correlation recovered from covariance');
assertClose(rho[0][2], -0.15, 1e-12, 'negative correlation survives the round trip');
assert(rho.every(row => row.every(v => v >= -1 - 1e-12 && v <= 1 + 1e-12)), 'correlations stay in range');

// Portfolio return and risk
console.log('\nPortfolio return and risk tests:');
const equal = [1 / 3, 1 / 3, 1 / 3];
assertClose(portfolioReturn([0.5, 0.5], [0.10, 0.20]), 0.15, 1e-12, 'return is the weighted average');
assertThrows(() => portfolioReturn([0.5], [0.1, 0.2]), 'return needs one weight per asset');

assertClose(portfolioVolatility([1, 0, 0], cov), 0.20, 1e-12, 'a single asset book has that asset vol');
assert(portfolioVariance(equal, cov) > 0, 'variance of a real portfolio is positive');
assertThrows(() => portfolioVariance([0.5, 0.5], cov), 'variance rejects a size mismatch');

// The whole point of diversification.
const weightedAvgVol = equal.reduce((s, w, i) => s + w * vols[i], 0);
assert(
  portfolioVolatility(equal, cov) < weightedAvgVol,
  'the combination is less volatile than the average of its parts'
);

// Perfectly correlated assets give up nothing.
const perfect = [[0.04, 0.04], [0.04, 0.04]];
assertClose(
  portfolioVolatility([0.5, 0.5], perfect),
  0.20,
  1e-12,
  'perfect correlation means no diversification at all'
);

// Solver
console.log('\nLinear solver tests:');
const x = solve([[2, 1], [1, 3]], [5, 10]);
assertClose(x[0], 1, 1e-12, 'solver gets the first unknown');
assertClose(x[1], 3, 1e-12, 'solver gets the second unknown');

const identity = solve([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [4, 5, 6]);
assert(identity.every((v, i) => Math.abs(v - [4, 5, 6][i]) < 1e-12), 'identity returns the input');

// Pivoting is what makes this case work at all.
const pivoted = solve([[0, 1], [1, 0]], [2, 3]);
assertClose(pivoted[0], 3, 1e-12, 'partial pivoting handles a zero on the diagonal');

assertThrows(() => solve([[1, 2], [2, 4]], [1, 2]), 'solver rejects a singular matrix');
assertThrows(() => solve([[1, 2], [3, 4]], [1]), 'solver rejects a mismatched right hand side');

// The solver must not damage the caller's matrix.
const before = JSON.stringify(cov);
solve(cov, [1, 1, 1]);
assert(JSON.stringify(cov) === before, 'solving leaves the input matrix untouched');

// Minimum variance
console.log('\nMinimum variance tests:');
const minVar = minVarianceWeights(cov);
assertClose(minVar.reduce((s, w) => s + w, 0), 1, 1e-12, 'minimum variance weights sum to one');

const minVarVol = portfolioVolatility(minVar, cov);
assert(minVarVol < portfolioVolatility(equal, cov), 'minimum variance beats equal weighting');
assert(minVarVol < Math.min(...vols), 'minimum variance beats holding the safest asset alone');

// Nudge the solution in every direction and check nothing does better.
let allWorse = true;
for (let i = 0; i < 3; i++) {
  for (const step of [0.02, -0.02]) {
    const nudged = minVar.map((w, j) => (j === i ? w + step : w - step / 2));
    if (portfolioVolatility(nudged, cov) < minVarVol - 1e-12) allWorse = false;
  }
}
assert(allWorse, 'no small perturbation lowers the minimum variance volatility');

// The heaviest weight should land on the lowest volatility asset here.
assert(minVar[1] === Math.max(...minVar), 'the quiet asset takes the largest minimum variance weight');

// Tangency
console.log('\nTangency tests:');
const expected = [0.09, 0.04, 0.13];
const tangency = tangencyWeights(cov, expected, 0.02);
assertClose(tangency.reduce((s, w) => s + w, 0), 1, 1e-12, 'tangency weights sum to one');

const sharpeOf = w => (portfolioReturn(w, expected) - 0.02) / portfolioVolatility(w, cov);
assert(sharpeOf(tangency) > sharpeOf(equal), 'tangency beats equal weighting on Sharpe');
assert(sharpeOf(tangency) > sharpeOf(minVar), 'tangency beats minimum variance on Sharpe');

let noBetterSharpe = true;
for (let i = 0; i < 3; i++) {
  for (const step of [0.02, -0.02]) {
    const nudged = tangency.map((w, j) => (j === i ? w + step : w - step / 2));
    if (sharpeOf(nudged) > sharpeOf(tangency) + 1e-12) noBetterSharpe = false;
  }
}
assert(noBetterSharpe, 'no small perturbation improves the tangency Sharpe');

// Risk contributions
console.log('\nRisk contribution tests:');
const { absolute, percent } = riskContributions(equal, cov);
assertClose(percent.reduce((s, p) => s + p, 0), 1, 1e-12, 'risk contributions add to the whole');
assertClose(
  absolute.reduce((s, a) => s + a, 0),
  portfolioVolatility(equal, cov),
  1e-12,
  'absolute contributions add to portfolio volatility'
);
assert(percent[2] > percent[1], 'the volatile asset carries more risk than the quiet one');
assert(Math.abs(percent[0] - 1 / 3) > 1e-6, 'equal weights are not equal risk');
assertThrows(() => riskContributions([0, 0, 0], cov), 'a riskless book has no contributions');

// A single asset portfolio carries all of its risk in that asset.
const solo = riskContributions([1, 0, 0], cov);
assertClose(solo.percent[0], 1, 1e-12, 'one asset carries one hundred percent of the risk');

// Diversification ratio
console.log('\nDiversification tests:');
assert(diversificationRatio(equal, cov) > 1, 'a mixed book scores above one');
assertClose(diversificationRatio([1, 0, 0], cov), 1, 1e-12, 'a single asset gets no diversification');
assertClose(
  diversificationRatio([0.5, 0.5], perfect),
  1,
  1e-12,
  'perfectly correlated assets score one'
);
assert(
  diversificationRatio(minVar, cov) > diversificationRatio([1, 0, 0], cov),
  'the optimized book diversifies better than a single holding'
);

// Summary
console.log(`\nResults: ${passed} passed, ${failed} failed.`);
if (failed > 0) process.exit(1);
