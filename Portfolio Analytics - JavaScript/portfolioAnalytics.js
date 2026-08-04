// portfolioAnalytics.js
// Covariance, portfolio risk, risk contributions and minimum variance weights.
// No external dependencies, pure Node.js. The linear algebra is written out
// because at portfolio sizes it is a few dozen lines and importing a matrix
// library to solve a five by five system is not a trade worth making.
// Usage: node portfolioAnalytics.js

'use strict';

/**
 * Sample covariance matrix from a matrix of returns.
 * @param {number[][]} returns Rows are periods, columns are assets
 * @returns {number[][]} n by n covariance matrix
 */
function covarianceMatrix(returns) {
  if (!Array.isArray(returns) || returns.length < 2) throw new Error('need at least two periods');
  const periods = returns.length;
  const n = returns[0].length;
  if (returns.some(row => row.length !== n)) throw new Error('every period must cover the same assets');

  const means = Array.from({ length: n }, (_, j) =>
    returns.reduce((s, row) => s + row[j], 0) / periods
  );

  const cov = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      let sum = 0;
      for (let t = 0; t < periods; t++) sum += (returns[t][i] - means[i]) * (returns[t][j] - means[j]);
      cov[i][j] = cov[j][i] = sum / (periods - 1);
    }
  }
  return cov;
}

/**
 * Correlation matrix derived from a covariance matrix.
 * @param {number[][]} cov
 * @returns {number[][]}
 */
function correlationMatrix(cov) {
  const n = cov.length;
  const sd = cov.map((row, i) => Math.sqrt(row[i]));
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (sd[i] === 0 || sd[j] === 0 ? 0 : cov[i][j] / (sd[i] * sd[j])))
  );
}

/**
 * Expected portfolio return, the weighted average of the asset returns.
 * @param {number[]} weights
 * @param {number[]} expectedReturns
 * @returns {number}
 */
function portfolioReturn(weights, expectedReturns) {
  if (weights.length !== expectedReturns.length) throw new Error('one weight per asset, please');
  return weights.reduce((s, w, i) => s + w * expectedReturns[i], 0);
}

/**
 * Portfolio variance, w transpose times covariance times w.
 * @param {number[]} weights
 * @param {number[][]} cov
 * @returns {number}
 */
function portfolioVariance(weights, cov) {
  const n = weights.length;
  if (cov.length !== n) throw new Error('the covariance matrix must match the number of weights');
  let total = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) total += weights[i] * weights[j] * cov[i][j];
  }
  return total;
}

/**
 * Portfolio volatility, the square root of the variance.
 * Note that this is smaller than the weighted average of the individual
 * volatilities whenever the assets are less than perfectly correlated, and
 * that gap is the whole of what diversification buys.
 * @param {number[]} weights
 * @param {number[][]} cov
 * @returns {number}
 */
function portfolioVolatility(weights, cov) {
  return Math.sqrt(Math.max(portfolioVariance(weights, cov), 0));
}

/**
 * Solve A x = b by Gauss Jordan elimination with partial pivoting.
 * @param {number[][]} A
 * @param {number[]} b
 * @returns {number[]}
 */
function solve(A, b) {
  const n = A.length;
  if (b.length !== n) throw new Error('the right hand side must match the matrix');
  // Work on a copy so the caller's covariance matrix survives the call.
  const M = A.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
    if (Math.abs(M[pivot][col]) < 1e-14) throw new Error('the matrix is singular, two assets are probably identical');
    [M[col], M[pivot]] = [M[pivot], M[col]];

    const p = M[col][col];
    for (let c = col; c <= n; c++) M[col][c] /= p;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = M[r][col];
      if (factor === 0) continue;
      for (let c = col; c <= n; c++) M[r][c] -= factor * M[col][c];
    }
  }
  return M.map(row => row[n]);
}

/**
 * Minimum variance weights, the closed form solution with no short selling ban.
 * The answer can and often does come back negative for some assets, which
 * means the unconstrained optimum wants a short position. Constrain it if
 * that is not something the mandate allows.
 * @param {number[][]} cov
 * @returns {number[]} Weights summing to one
 */
function minVarianceWeights(cov) {
  const n = cov.length;
  const raw = solve(cov, new Array(n).fill(1));
  const total = raw.reduce((s, v) => s + v, 0);
  if (Math.abs(total) < 1e-14) throw new Error('no minimum variance portfolio exists for this matrix');
  return raw.map(v => v / total);
}

/**
 * Tangency weights, the portfolio with the best Sharpe ratio.
 * @param {number[][]} cov
 * @param {number[]} expectedReturns Annual, as decimals
 * @param {number} [riskFreeRate=0]
 * @returns {number[]} Weights summing to one
 */
function tangencyWeights(cov, expectedReturns, riskFreeRate = 0) {
  const excess = expectedReturns.map(r => r - riskFreeRate);
  const raw = solve(cov, excess);
  const total = raw.reduce((s, v) => s + v, 0);
  if (Math.abs(total) < 1e-14) throw new Error('no tangency portfolio exists for these inputs');
  return raw.map(v => v / total);
}

/**
 * How much of the portfolio risk each holding is actually responsible for.
 * Weights and risk contributions are very different things. A small holding
 * in something volatile and correlated with everything else can easily carry
 * more risk than a large holding in something that zigs when the rest zags.
 * @param {number[]} weights
 * @param {number[][]} cov
 * @returns {{ absolute: number[], percent: number[] }}
 */
function riskContributions(weights, cov) {
  const vol = portfolioVolatility(weights, cov);
  if (vol === 0) throw new Error('a portfolio with no risk has no risk contributions');
  const n = weights.length;
  const marginal = Array.from({ length: n }, (_, i) =>
    weights.reduce((s, w, j) => s + w * cov[i][j], 0) / vol
  );
  const absolute = marginal.map((m, i) => m * weights[i]);
  return { absolute, percent: absolute.map(a => a / vol) };
}

/**
 * Diversification ratio, weighted average volatility over portfolio volatility.
 * One means the diversification did nothing. Higher is better and the number
 * falls towards one exactly when correlations rise, which is why it collapses
 * in a crisis just as you need it most.
 * @param {number[]} weights
 * @param {number[][]} cov
 * @returns {number}
 */
function diversificationRatio(weights, cov) {
  const weighted = weights.reduce((s, w, i) => s + Math.abs(w) * Math.sqrt(cov[i][i]), 0);
  const vol = portfolioVolatility(weights, cov);
  if (vol === 0) throw new Error('a portfolio with no risk has no diversification ratio');
  return weighted / vol;
}

// Demo
if (require.main === module) {
  console.log('\n' + '='.repeat(56));
  console.log('PORTFOLIO ANALYTICS DEMO');
  console.log('='.repeat(56));

  // Three assets, annual covariance built from vols of 20%, 12% and 30%
  // with a low correlation between the first two and a negative one to the third.
  const vols = [0.20, 0.12, 0.30];
  const corr = [
    [1.00, 0.25, -0.15],
    [0.25, 1.00, 0.05],
    [-0.15, 0.05, 1.00],
  ];
  const cov = corr.map((row, i) => row.map((c, j) => c * vols[i] * vols[j]));
  const expected = [0.09, 0.04, 0.13];
  const names = ['Equity', 'Bonds ', 'Alts  '];

  const equal = [1 / 3, 1 / 3, 1 / 3];
  console.log('\nEqual weighted');
  console.log(`   expected return ${(portfolioReturn(equal, expected) * 100).toFixed(2)}%`);
  console.log(`   volatility      ${(portfolioVolatility(equal, cov) * 100).toFixed(2)}%`);
  console.log(`   weighted avg vol ${(equal.reduce((s, w, i) => s + w * vols[i], 0) * 100).toFixed(2)}%`);
  console.log(`   diversification  ${diversificationRatio(equal, cov).toFixed(3)}`);

  const minVar = minVarianceWeights(cov);
  console.log('\nMinimum variance');
  minVar.forEach((w, i) => console.log(`   ${names[i]} ${(w * 100).toFixed(2)}%`));
  console.log(`   expected return ${(portfolioReturn(minVar, expected) * 100).toFixed(2)}%`);
  console.log(`   volatility      ${(portfolioVolatility(minVar, cov) * 100).toFixed(2)}%`);

  const tangency = tangencyWeights(cov, expected, 0.02);
  console.log('\nTangency, best Sharpe at a 2% risk free rate');
  tangency.forEach((w, i) => console.log(`   ${names[i]} ${(w * 100).toFixed(2)}%`));
  const tr = portfolioReturn(tangency, expected);
  const tv = portfolioVolatility(tangency, cov);
  console.log(`   expected return ${(tr * 100).toFixed(2)}%`);
  console.log(`   volatility      ${(tv * 100).toFixed(2)}%`);
  console.log(`   Sharpe          ${((tr - 0.02) / tv).toFixed(3)}`);

  console.log('\nWhere the risk really sits in the equal weighted book');
  const { percent } = riskContributions(equal, cov);
  percent.forEach((p, i) =>
    console.log(`   ${names[i]} weight 33.33%   risk ${(p * 100).toFixed(2)}%`)
  );
  console.log('\nEqual weights are not equal risk. They almost never are.');
}

module.exports = {
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
};
