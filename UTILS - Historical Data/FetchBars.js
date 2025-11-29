// fetchAlpacaHistorical.js
// Fetch historical bars from Alpaca Market Data API v2 (stocks) or v1beta3 (crypto)
// Prompts user for symbol type, symbol, and timeframe
// Requires: node-fetch, dotenv
// Usage: node fetchAlpacaHistorical.js

require('dotenv').config();
const readline = require('readline');
const fetch = (...args) => import('node-fetch').then(({default: fetch}) => fetch(...args));

// Load Alpaca API credentials from environment variables
const API_KEY = process.env.ALPACA_API_KEY;
const API_SECRET = process.env.ALPACA_API_SECRET;

if (!API_KEY || !API_SECRET) {
  console.error('Missing Alpaca API credentials. Please set ALPACA_API_KEY and ALPACA_API_SECRET in your .env file.');
  process.exit(1);
}

// Create readline interface for user input
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

// Helper to prompt user for input
function ask(question) {
  return new Promise(resolve => rl.question(question, answer => resolve(answer.trim())));
}

// Fetch historical bars for stocks or crypto
async function fetchHistoricalBars({ type, symbol, start, end, timeframe }) {
  let url;
  let headers = {
    'APCA-API-KEY-ID': API_KEY,
    'APCA-API-SECRET-KEY': API_SECRET,
  };

  if (type === 'stock') {
    // Stock endpoint (v2)
    url = `https://data.alpaca.markets/v2/stocks/${symbol}/bars?start=${start}&end=${end}&timeframe=${timeframe}`;
  } else if (type === 'crypto') {
    // Crypto endpoint (v1beta3, US market)
    url = `https://data.alpaca.markets/v1beta3/crypto/us/bars?symbols=${symbol}&start=${start}&end=${end}&timeframe=${timeframe}`;
  } else {
    console.error('Invalid type. Must be "stock" or "crypto".');
    return;
  }

  try {
    const response = await fetch(url, { headers });
    if (!response.ok) {
      if (response.status === 403) {
        throw new Error('HTTP 403 Forbidden: Check your API credentials and permissions.');
      }
      throw new Error(`HTTP error! Status: ${response.status} - ${await response.text()}`);
    }
    const data = await response.json();
    console.log(JSON.stringify(data, null, 2));
  } catch (error) {
    console.error('Error fetching historical bars:', error.message);
  }
}

// Main function to prompt user and fetch data
(async () => {
  // Ask for symbol type
  let type = (await ask('Enter symbol type (stock/crypto): ')).toLowerCase();
  while (type !== 'stock' && type !== 'crypto') {
    type = (await ask('Invalid type. Please enter "stock" or "crypto": ')).toLowerCase();
  }

  // Ask for symbol
  let symbol = await ask('Enter symbol (e.g. AAPL for stock, BTC/USD for crypto): ');
  while (!symbol) {
    symbol = await ask('Symbol cannot be empty. Enter symbol: ');
  }

  // Ask for timeframe
  let timeframe = await ask('Enter timeframe (e.g. 1Day, 1Hour, 5Min): ');
  while (!timeframe) {
    timeframe = await ask('Timeframe cannot be empty. Enter timeframe: ');
  }

  // Ask for start and end dates
  let start = await ask('Enter start date (YYYY-MM-DD): ');
  while (!/^\d{4}-\d{2}-\d{2}$/.test(start)) {
    start = await ask('Invalid date format. Enter start date (YYYY-MM-DD): ');
  }
  let end = await ask('Enter end date (YYYY-MM-DD): ');
  while (!/^\d{4}-\d{2}-\d{2}$/.test(end)) {
    end = await ask('Invalid date format. Enter end date (YYYY-MM-DD): ');
  }

  // Fetch and display historical bars
  await fetchHistoricalBars({ type, symbol, start, end, timeframe });
  rl.close();
});
