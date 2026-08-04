// riskMetrics.js
// Drawdown, volatility, risk adjusted ratios and tail risk in plain JavaScript.
// No external dependencies, pure Node.js.
// Usage: node riskMetrics.js

'use strict';

const TRADING_DAYS = 252;

/**
 * Mean of an array.
 * @param {number[]} values
 * @returns {number}
 */
function mean(values) {
  if (values.length === 0) throw new Error('need at least one value');
  return values.reduce((s, v) => s + v, 0) / values.length;
}

/**
 * Sample standard deviation, the n minus one version.
 * @param {number[]} values
 * @returns {number}
 */
function stdDev(values) {
  if (values.length < 2) throw new Error('need at least two values');
  const m = mean(values);
  return Math.sqrt(values.reduce((s, v) => s + (v - m) ** 2, 0) / (values.length - 1));
}

/**
 * Turn a series of prices into simple period returns.
 * @param {number[]} prices
 * @returns {number[]} One shorter than the input
 */
function toReturns(prices) {
  if (prices.length < 2) throw new Error('need at least two prices');
  const out = [];
  for (let i = 1; i < prices.length; i++) {
    if (prices[i - 1] === 0) throw new Error('cannot compute a return from a zero price');
    out.push(prices[i] / prices[i - 1] - 1);
  }
  return out;
}

/**
 * Compound a series of returns into an equity curve starting at 1.
 * @param {number[]} returns
 * @returns {number[]} One longer than the input
 */
function equityCurve(returns) {
  const curve = [1];
  for (const r of returns) curve.push(curve[curve.length - 1] * (1 + r));
  return curve;
}

/**
 * Annualized volatility from period returns.
 * @param {number[]} returns
 * @param {number} [periods=252] Periods per year
 * @returns {number}
 */
function annualizedVolatility(returns, periods = TRADING_DAYS) {
  return stdDev(returns) * Math.sqrt(periods);
}

/**
 * Compound annual growth rate implied by a return series.
 * @param {number[]} returns
 * @param {number} [periods=252]
 * @returns {number}
 */
function annualizedReturn(returns, periods = TRADING_DAYS) {
  const curve = equityCurve(returns);
  const total = curve[curve.length - 1];
  if (total <= 0) throw new Error('the equity curve went to zero, no growth rate exists');
  return total ** (periods / returns.length) - 1;
}

/**
 * Worst peak to trough fall, along with where it happened.
 * The peak is tracked as the series is walked, so this is the real drawdown
 * and not the distance between the highest and lowest points, which would
 * be wrong whenever the low comes first.
 * @param {number[]} returns
 * @returns {{ maxDrawdown: number, peakIndex: number, troughIndex: number, recoveryIndex: number }}
 */
function maxDrawdown(returns) {
  const curve = equityCurve(returns);
  let peak = curve[0];
  let peakIndex = 0;
  let worst = 0;
  let bestPeak = 0;
  let trough = 0;

  for (let i = 1; i < curve.length; i++) {
    if (curve[i] > peak) {
      peak = curve[i];
      peakIndex = i;
    }
    const decline = curve[i] / peak - 1;
    if (decline < worst) {
      worst = decline;
      bestPeak = peakIndex;
      trough = i;
    }
  }

  // Recovery is the first point after the trough back at the old high, or -1.
  let recoveryIndex = -1;
  for (let i = trough + 1; i < curve.length; i++) {
    if (curve[i] >= curve[bestPeak]) { recoveryIndex = i; break; }
  }

  return { maxDrawdown: worst, peakIndex: bestPeak, troughIndex: trough, recoveryIndex };
}

/**
 * Sharpe ratio, annualized, excess return per unit of total volatility.
 * @param {number[]} returns
 * @param {number} [riskFreeRate=0] Annual rate as a decimal
 * @param {number} [periods=252]
 * @returns {number}
 */
function sharpeRatio(returns, riskFreeRate = 0, periods = TRADING_DAYS) {
  const perPeriodRf = riskFreeRate / periods;
  const excess = returns.map(r => r - perPeriodRf);
  const vol = stdDev(excess);
  // Summing a long run of identical returns leaves a deviation of about 1e-19
  // rather than a clean zero, so compare against a tolerance and not to zero.
  if (vol < 1e-15) throw new Error('a series with no variation has no Sharpe ratio');
  return (mean(excess) / vol) * Math.sqrt(periods);
}

/**
 * Sortino ratio, which penalises only the downside.
 * Upside volatility is not risk to anyone holding the position, so the
 * denominator here counts only returns below the target.
 * @param {number[]} returns
 * @param {number} [target=0] Annual minimum acceptable return
 * @param {number} [periods=252]
 * @returns {number}
 */
function sortinoRatio(returns, target = 0, periods = TRADING_DAYS) {
  const perPeriodTarget = target / periods;
  const excess = returns.map(r => r - perPeriodTarget);
  const shortfalls = excess.map(r => Math.min(r, 0) ** 2);
  const downside = Math.sqrt(mean(shortfalls));
  if (downside === 0) return Infinity;
  return (mean(excess) / downside) * Math.sqrt(periods);
}

/**
 * Calmar ratio, annual return divided by the worst drawdown.
 * @param {number[]} returns
 * @param {number} [periods=252]
 * @returns {number}
 */
function calmarRatio(returns, periods = TRADING_DAYS) {
  const dd = Math.abs(maxDrawdown(returns).maxDrawdown);
  if (dd === 0) return Infinity;
  return annualizedReturn(returns, periods) / dd;
}

/**
 * Historical Value at Risk, reported as a positive loss.
 * This is the empirical quantile, so it makes no assumption about the shape
 * of the distribution and it cannot see a loss worse than any in the sample.
 * @param {number[]} returns
 * @param {number} [confidence=0.95]
 * @returns {number}
 */
function historicalVaR(returns, confidence = 0.95) {
  if (confidence <= 0 || confidence >= 1) throw new Error('confidence must sit between zero and one');
  if (returns.length === 0) throw new Error('need at least one return');
  const sorted = [...returns].sort((a, b) => a - b);
  // One minus 0.8 is 0.19999999999999996, so a bare floor lands one row early
  // on exactly the round confidence levels people actually use.
  const idx = Math.min(sorted.length - 1, Math.floor((1 - confidence) * sorted.length + 1e-9));
  return -sorted[idx];
}

/**
 * Conditional Value at Risk, the average loss once VaR is breached.
 * VaR says how bad a normal bad day is. This says how bad the bad days are
 * on average once you are already in one, which is the number that matters
 * for sizing a position you might have to hold through it.
 * @param {number[]} returns
 * @param {number} [confidence=0.95]
 * @returns {number}
 */
function conditionalVaR(returns, confidence = 0.95) {
  if (confidence <= 0 || confidence >= 1) throw new Error('confidence must sit between zero and one');
  if (returns.length === 0) throw new Error('need at least one return');
  const sorted = [...returns].sort((a, b) => a - b);
  const cutoff = Math.max(1, Math.floor((1 - confidence) * sorted.length + 1e-9));
  return -mean(sorted.slice(0, cutoff));
}

/**
 * Ulcer index, the root mean square drawdown across the whole path.
 * A single deep hole and a long shallow one can share a maximum drawdown
 * while feeling nothing alike. This measures the time spent underwater too.
 * @param {number[]} returns
 * @returns {number}
 */
function ulcerIndex(returns) {
  const curve = equityCurve(returns);
  let peak = curve[0];
  let sum = 0;
  for (const v of curve) {
    if (v > peak) peak = v;
    sum += (v / peak - 1) ** 2;
  }
  return Math.sqrt(sum / curve.length);
}

/**
 * A single object holding every metric above, for a quick strategy report.
 * @param {number[]} returns
 * @param {number} [riskFreeRate=0]
 * @param {number} [periods=252]
 * @returns {Object}
 */
function riskReport(returns, riskFreeRate = 0, periods = TRADING_DAYS) {
  const dd = maxDrawdown(returns);
  return {
    annualReturn: annualizedReturn(returns, periods),
    annualVolatility: annualizedVolatility(returns, periods),
    sharpe: sharpeRatio(returns, riskFreeRate, periods),
    sortino: sortinoRatio(returns, riskFreeRate, periods),
    calmar: calmarRatio(returns, periods),
    maxDrawdown: dd.maxDrawdown,
    drawdownLength: dd.troughIndex - dd.peakIndex,
    recovered: dd.recoveryIndex !== -1,
    var95: historicalVaR(returns, 0.95),
    cvar95: conditionalVaR(returns, 0.95),
    ulcer: ulcerIndex(returns),
  };
}

// Demo
if (require.main === module) {
  console.log('\n' + '='.repeat(56));
  console.log('RISK METRICS DEMO');
  console.log('='.repeat(56));

  // A reproducible pseudo random walk, no seeding library needed.
  let seed = 42;
  const rand = () => {
    seed = (seed * 1103515245 + 12345) % 2147483648;
    return seed / 2147483648;
  };
  const normal = () => Math.sqrt(-2 * Math.log(rand() || 1e-12)) * Math.cos(2 * Math.PI * rand());

  const steady = Array.from({ length: 756 }, () => 0.0004 + 0.008 * normal());
  const jumpy = Array.from({ length: 756 }, (_, i) => 0.0009 + 0.020 * normal() - (i > 400 && i < 440 ? 0.01 : 0));

  for (const [name, series] of [['STEADY', steady], ['JUMPY', jumpy]]) {
    const r = riskReport(series);
    console.log(`\n${name} strategy, three years of daily returns`);
    console.log(`   annual return    ${(r.annualReturn * 100).toFixed(2)}%`);
    console.log(`   annual vol       ${(r.annualVolatility * 100).toFixed(2)}%`);
    console.log(`   Sharpe           ${r.sharpe.toFixed(3)}`);
    console.log(`   Sortino          ${r.sortino.toFixed(3)}`);
    console.log(`   Calmar           ${r.calmar.toFixed(3)}`);
    console.log(`   max drawdown     ${(r.maxDrawdown * 100).toFixed(2)}% over ${r.drawdownLength} days`);
    console.log(`   recovered        ${r.recovered ? 'yes' : 'not yet'}`);
    console.log(`   VaR  95%         ${(r.var95 * 100).toFixed(2)}%`);
    console.log(`   CVaR 95%         ${(r.cvar95 * 100).toFixed(2)}%`);
    console.log(`   ulcer index      ${(r.ulcer * 100).toFixed(2)}%`);
  }

  console.log('\nThe higher return is not the better strategy once the drawdown is priced in.');

  console.log('\nPrices to returns, then straight into the report');
  const prices = [100, 102, 101, 105, 103, 108, 107, 110];
  const rr = toReturns(prices);
  console.log(`   ${rr.length} returns, worst drawdown ${(maxDrawdown(rr).maxDrawdown * 100).toFixed(2)}%`);
}

module.exports = {
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
};
