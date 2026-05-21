// monteCarlo.js
// Monte Carlo simulation for portfolio returns and European option pricing via GBM.
// No external dependencies — pure JavaScript.
// Usage: node monteCarlo.js

'use strict';

/**
 * Box-Muller transform: generate a standard-normal variate.
 * @returns {number}
 */
function randNormal() {
  let u;
  do { u = Math.random(); } while (u === 0);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
}

/**
 * Simulate one GBM price path.
 * @param {number} S0 - Initial price
 * @param {number} mu - Annual drift (decimal)
 * @param {number} sigma - Annual volatility (decimal)
 * @param {number} T - Horizon in years
 * @param {number} steps - Number of time steps
 * @returns {number[]} Price path of length steps + 1
 */
function simulatePath(S0, mu, sigma, T, steps) {
  const dt = T / steps;
  const path = [S0];
  for (let i = 0; i < steps; i++) {
    const prev = path[path.length - 1];
    path.push(
      prev * Math.exp((mu - 0.5 * sigma ** 2) * dt + sigma * Math.sqrt(dt) * randNormal())
    );
  }
  return path;
}

/**
 * Cholesky decomposition of a symmetric positive-definite matrix.
 * @param {number[][]} A - n×n matrix
 * @returns {number[][]} Lower triangular matrix L such that A = L * L^T
 */
function cholesky(A) {
  const n = A.length;
  const L = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = A[i][j];
      for (let k = 0; k < j; k++) sum -= L[i][k] * L[j][k];
      L[i][j] = j === i ? Math.sqrt(Math.max(sum, 0)) : sum / L[j][j];
    }
  }
  return L;
}

/**
 * Multi-asset correlated GBM portfolio simulation.
 * @param {number[]} S0 - Initial prices per asset
 * @param {number[]} mu - Annual drifts per asset
 * @param {number[]} sigma - Annual vols per asset
 * @param {number[][]} corr - Correlation matrix (n×n)
 * @param {number[]} weights - Portfolio weights (should sum to 1)
 * @param {number} T - Horizon in years
 * @param {number} steps - Time steps per path
 * @param {number} nSims - Number of Monte Carlo paths
 * @returns {{ finalReturns: number[], samplePaths: number[][] }}
 */
function simulatePortfolio(S0, mu, sigma, corr, weights, T, steps, nSims) {
  const n = S0.length;
  const dt = T / steps;
  const L = cholesky(corr);
  const finalReturns = [];
  const samplePaths = [];
  const initialPortfolio = weights.reduce((s, w, i) => s + w * S0[i], 0);

  for (let sim = 0; sim < nSims; sim++) {
    const prices = [...S0];
    const track = sim < 5;
    const equityCurve = track ? [initialPortfolio] : null;

    for (let t = 0; t < steps; t++) {
      const z = Array.from({ length: n }, () => randNormal());
      // Apply Cholesky to impose correlation
      const correlated = Array.from({ length: n }, (_, i) =>
        L[i].reduce((s, lij, j) => s + lij * z[j], 0)
      );
      for (let i = 0; i < n; i++) {
        prices[i] *= Math.exp(
          (mu[i] - 0.5 * sigma[i] ** 2) * dt + sigma[i] * Math.sqrt(dt) * correlated[i]
        );
      }
      if (track) equityCurve.push(weights.reduce((s, w, i) => s + w * prices[i], 0));
    }

    finalReturns.push(
      weights.reduce((s, w, i) => s + w * prices[i], 0) / initialPortfolio - 1
    );
    if (track) samplePaths.push(equityCurve);
  }

  return { finalReturns, samplePaths };
}

/**
 * Monte Carlo price for a European call option (risk-neutral GBM).
 * Uses antithetic variates for variance reduction.
 * @param {number} S - Spot price
 * @param {number} K - Strike price
 * @param {number} T - Time to expiry (years)
 * @param {number} r - Risk-free rate (decimal)
 * @param {number} sigma - Volatility (decimal)
 * @param {number} [nSims=50000]
 * @returns {{ price: number, stdErr: number }}
 */
function mcOptionPrice(S, K, T, r, sigma, nSims = 50_000) {
  const drift = (r - 0.5 * sigma ** 2) * T;
  const vol = sigma * Math.sqrt(T);
  const discount = Math.exp(-r * T);
  const payoffs = [];

  for (let i = 0; i < nSims; i++) {
    const z = randNormal();
    const payoff = (Math.max(S * Math.exp(drift + vol * z) - K, 0) +
                    Math.max(S * Math.exp(drift - vol * z) - K, 0)) / 2;
    payoffs.push(payoff);
  }

  const mean = payoffs.reduce((s, v) => s + v, 0) / nSims;
  const variance = payoffs.reduce((s, v) => s + (v - mean) ** 2, 0) / (nSims - 1);
  return { price: discount * mean, stdErr: discount * Math.sqrt(variance / nSims) };
}

/**
 * Value-at-Risk and Conditional VaR from simulated returns.
 * @param {number[]} returns
 * @param {number} [confidence=0.95]
 * @returns {{ var: number, cvar: number }}
 */
function varCvar(returns, confidence = 0.95) {
  const sorted = [...returns].sort((a, b) => a - b);
  const idx = Math.floor((1 - confidence) * sorted.length);
  const varValue = -sorted[idx];
  const tailMean = sorted.slice(0, idx + 1).reduce((s, v) => s + v, 0) / (idx + 1);
  return { var: varValue, cvar: -tailMean };
}

// ── Demo ──────────────────────────────────────────────────────────────
if (require.main === module) {
  console.log('\n' + '='.repeat(56));
  console.log('MONTE CARLO SIMULATION DEMO');
  console.log('='.repeat(56));

  // 1. Single asset path
  console.log('\n1. SINGLE ASSET PATH (GBM, 252 steps, 1 year)');
  const path = simulatePath(150, 0.10, 0.25, 1, 252);
  const pathReturn = (path[path.length - 1] / path[0] - 1) * 100;
  console.log(`   Start: $${path[0].toFixed(2)}   End: $${path[path.length - 1].toFixed(2)}`);
  console.log(`   Return: ${pathReturn.toFixed(2)}%`);

  // 2. Portfolio simulation
  console.log('\n2. THREE-ASSET PORTFOLIO (10,000 paths, 1 year)');
  const { finalReturns } = simulatePortfolio(
    [100, 50, 200],
    [0.10, 0.08, 0.12],
    [0.20, 0.15, 0.25],
    [[1, 0.40, 0.20], [0.40, 1, 0.30], [0.20, 0.30, 1]],
    [0.50, 0.30, 0.20],
    1, 252, 10_000
  );
  const n = finalReturns.length;
  const mean = finalReturns.reduce((s, v) => s + v, 0) / n;
  const std = Math.sqrt(finalReturns.reduce((s, v) => s + (v - mean) ** 2, 0) / n);
  const { var: var95, cvar: cvar95 } = varCvar(finalReturns, 0.95);

  console.log(`   Mean return:   ${(mean * 100).toFixed(2)}%`);
  console.log(`   Volatility:    ${(std * 100).toFixed(2)}%`);
  console.log(`   Sharpe (rf=0): ${(mean / std).toFixed(3)}`);
  console.log(`   VaR  (95%):    ${(var95 * 100).toFixed(2)}%`);
  console.log(`   CVaR (95%):    ${(cvar95 * 100).toFixed(2)}%`);

  // 3. Option pricing
  console.log('\n3. EUROPEAN CALL PRICE VIA MC (S=100, K=105, T=0.5yr, r=5%, σ=20%)');
  const { price: mcPrice, stdErr } = mcOptionPrice(100, 105, 0.5, 0.05, 0.20, 50_000);
  console.log(`   MC price:  $${mcPrice.toFixed(4)} ± $${stdErr.toFixed(4)} (1 std err)`);
  console.log('   Analytical Black-Scholes: ~$5.08 — expect close match.');

  console.log('\n✓ Monte Carlo demo complete.');
}

module.exports = { simulatePath, simulatePortfolio, mcOptionPrice, varCvar, randNormal, cholesky };
