# Market Data Utilities

<p class="lq-meta">Beginner <span>/</span> Utilities & Tools <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Market%20Data">source</a></p>


This folder contains utilities for processing, analyzing, and fetching market data for financial applications.

## Available Utilities

### Market Data (`market_data_utils.py`)
- Return calculations (simple and logarithmic)
- Outlier detection and missing data handling
- Market sentiment analysis
- Data validation and quality control
- Market timing indicators
- Data smoothing techniques

### API (`api_utils.py`)
- HTTP request handling with retry logic
- API key generation and secure storage
- Error handling and timeout management
- Response parsing and validation

## Usage

```python
# Market data operations
from market_data_utils import calculate_returns, detect_outliers, calculate_market_sentiment
from api_utils import make_api_request, retry_api_request, generate_api_key

# Data analysis
returns = calculate_returns(prices, 'log')
outliers = detect_outliers(data, 'iqr')
sentiment = calculate_market_sentiment(news_data, keywords)

# API operations
response = make_api_request("https://api.example.com/stocks/AAPL")
api_key = generate_api_key(32)
```

## Installation

Requires numpy, scipy, and requests:

```bash
pip install numpy scipy requests
```

## Testing

Run each utility directly to see demonstrations:

```bash
python market_data_utils.py
python api_utils.py
```

## Common Use Cases

- **Data Processing**: Clean and validate market data
- **Technical Analysis**: Calculate indicators and signals
- **API Integration**: Fetch data from external sources
- **Quality Control**: Ensure data accuracy and completeness
- **Sentiment Analysis**: Analyze market sentiment from news
- **Algorithm Development**: Prepare data for trading strategies

---

## Running it

```bash
cd "Market Data"
python "api_utils.py"
```

## More in Utilities & Tools

- [Core Utilities](Core Utilities.md): This folder contains core mathematical and date/time utilities that form the foundation for quantitative finance calculations.
- [Currency Converter](Currency Converter.md): This tool lets you convert between currencies using user-supplied exchange rates.
- [Data Processing](Data Processing.md): This folder contains utilities for data processing, validation, and manipulation in financial applications.
- [Economic Calendar](Economic Calendar.md): This tool lets you simulate and explore economic events (like FOMC meetings, CPI releases, etc.) and see how they might impact markets.
- [Historical Data](Historical Data.md): A Node.js script that fetches historical bars (OHLCV data) for stocks or crypto from the Alpaca Market Data API.
- [Logging](Logging.md): A pair of minimal, dependency-light logging utilities implemented in both Python and JavaScript.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
