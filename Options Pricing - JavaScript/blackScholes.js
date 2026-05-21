// blackScholes.js
// Black-Scholes European options pricing, Greeks, and implied volatility.
// No external dependencies — pure JavaScript implementation.
// Usage: node blackScholes.js

'use strict';

/**
 * Cumulative standard normal CDF (Abramowitz & Stegun approximation, error < 7.5e-8).
 * @param {number} x
 * @returns {number}
 */
function normCdf(x) {
  const a = [0.31938153, -0.356563782, 1.781477937, -1.821255978, 1.330274429];
  const L = Math.abs(x);
  const K = 1 / (1 + 0.2316419 * L);
  let poly = 0;
  for (let j = a.length - 1; j >= 0; j--) poly = poly * K + a[j];
  poly *= K;
  const pdf = Math.exp(-0.5 * L * L) / Math.sqrt(2 * Math.PI);
  const cdf = 1 - pdf * poly;
  return x >= 0 ? cdf : 1 - cdf;
}

/**
 * Standard normal PDF.
 * @param {number} x
 * @returns {number}
 */
function normPdf(x) {
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

/**
 * Internal d1 and d2 for Black-Scholes.
 * @param {number} S - Spot price
 * @param {number} K - Strike price
 * @param {number} T - Time to expiry (years)
 * @param {number} r - Risk-free rate (decimal)
 * @param {number} sigma - Annualised volatility (decimal)
 * @returns {{ d1: number, d2: number }}
 */
function _d1d2(S, K, T, r, sigma) {
  const d1 = (Math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * Math.sqrt(T));
  return { d1, d2: d1 - sigma * Math.sqrt(T) };
}

/**
 * Black-Scholes price for a European call or put.
 * @param {number} S - Spot price
 * @param {number} K - Strike price
 * @param {number} T - Time to expiry in years
 * @param {number} r - Risk-free rate (decimal, e.g. 0.05)
 * @param {number} sigma - Annualised volatility (decimal)
 * @param {'call'|'put'} [type='call']
 * @returns {number} Option premium
 */
function price(S, K, T, r, sigma, type = 'call') {
  if (T <= 0) return Math.max(type === 'call' ? S - K : K - S, 0);
  const { d1, d2 } = _d1d2(S, K, T, r, sigma);
  const discount = K * Math.exp(-r * T);
  return type === 'call'
    ? S * normCdf(d1) - discount * normCdf(d2)
    : discount * normCdf(-d2) - S * normCdf(-d1);
}

/**
 * All five Black-Scholes Greeks.
 * @param {number} S
 * @param {number} K
 * @param {number} T
 * @param {number} r
 * @param {number} sigma
 * @param {'call'|'put'} [type='call']
 * @returns {{ delta: number, gamma: number, theta: number, vega: number, rho: number }}
 */
function greeks(S, K, T, r, sigma, type = 'call') {
  if (T <= 0) {
    const itm = type === 'call' ? S > K : S < K;
    return { delta: itm ? (type === 'call' ? 1 : -1) : 0, gamma: 0, theta: 0, vega: 0, rho: 0 };
  }
  const { d1, d2 } = _d1d2(S, K, T, r, sigma);
  const sqrtT = Math.sqrt(T);
  const nd1 = normPdf(d1);
  const discount = K * Math.exp(-r * T);

  const delta = type === 'call' ? normCdf(d1) : normCdf(d1) - 1;
  const gamma = nd1 / (S * sigma * sqrtT);
  // Theta divided by 365 to express as daily decay
  const theta = type === 'call'
    ? (-(S * nd1 * sigma) / (2 * sqrtT) - r * discount * normCdf(d2)) / 365
    : (-(S * nd1 * sigma) / (2 * sqrtT) + r * discount * normCdf(-d2)) / 365;
  const vega = (S * nd1 * sqrtT) / 100;   // per 1% volatility move
  const rho = type === 'call'
    ? (discount * T * normCdf(d2)) / 100
    : (-discount * T * normCdf(-d2)) / 100;

  return { delta, gamma, theta, vega, rho };
}

/**
 * Implied volatility via bisection search.
 * @param {number} marketPrice - Observed option premium
 * @param {number} S
 * @param {number} K
 * @param {number} T
 * @param {number} r
 * @param {'call'|'put'} [type='call']
 * @param {number} [tol=1e-6]
 * @param {number} [maxIter=200]
 * @returns {number|null} Implied volatility or null if not found
 */
function impliedVol(marketPrice, S, K, T, r, type = 'call', tol = 1e-6, maxIter = 200) {
  let lo = 1e-6;
  let hi = 10.0;
  for (let i = 0; i < maxIter; i++) {
    const mid = (lo + hi) / 2;
    const diff = price(S, K, T, r, mid, type) - marketPrice;
    if (Math.abs(diff) < tol) return mid;
    if (diff > 0) hi = mid; else lo = mid;
  }
  return null;
}

// ── Demo ──────────────────────────────────────────────────────────────
if (require.main === module) {
  const S = 100, K = 105, T = 0.5, r = 0.05, sigma = 0.20;

  console.log('\n' + '='.repeat(56));
  console.log('BLACK-SCHOLES OPTIONS PRICER — DEMO');
  console.log('='.repeat(56));
  console.log(
    `Spot: $${S}  Strike: $${K}  T: ${T}yr  ` +
    `r: ${(r * 100).toFixed(1)}%  σ: ${(sigma * 100).toFixed(1)}%`
  );

  for (const type of ['call', 'put']) {
    const p = price(S, K, T, r, sigma, type);
    const g = greeks(S, K, T, r, sigma, type);
    const iv = impliedVol(p, S, K, T, r, type);

    console.log(`\n${type.toUpperCase()}`);
    console.log(`  Price:  $${p.toFixed(4)}`);
    console.log(`  Delta:  ${g.delta.toFixed(4)}`);
    console.log(`  Gamma:  ${g.gamma.toFixed(4)}`);
    console.log(`  Theta:  ${g.theta.toFixed(4)}  (daily)`);
    console.log(`  Vega:   ${g.vega.toFixed(4)}  (per 1% vol move)`);
    console.log(`  Rho:    ${g.rho.toFixed(4)}  (per 1% rate move)`);
    console.log(`  IV check: ${iv !== null ? (iv * 100).toFixed(4) + '%' : 'N/A'} (should equal ${(sigma * 100).toFixed(1)}%)`);
  }

  // Put-call parity verification
  const callP = price(S, K, T, r, sigma, 'call');
  const putP = price(S, K, T, r, sigma, 'put');
  const parity = callP - putP - (S - K * Math.exp(-r * T));
  console.log(`\nPut-call parity residual: ${parity.toExponential(3)} (should be ~0)`);
  console.log('\n✓ Black-Scholes demo complete.');
}

module.exports = { price, greeks, impliedVol, normCdf, normPdf };
