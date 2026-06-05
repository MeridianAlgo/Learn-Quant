<p class="lq-badges"><span class="lq-badge lq-beginner">Beginner</span><span class="lq-badge lq-cat">Utilities & Tools</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Market Data"
    python "api_utils.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Market%20Data)

---
# Market Data Utilities

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

## Continue in Utilities & Tools

<div class="grid cards" markdown>

-   :material-tools: __[Core Utilities](Core Utilities.md)__

    This folder contains core mathematical and date/time utilities that form the foundation for quantitative finance calculations.

-   :material-tools: __[Currency Converter](Currency Converter.md)__

    **This utility does NOT use any external APIs.** All exchange rates are entered manually for learning and experimentation.

-   :material-tools: __[Data Processing](Data Processing.md)__

    This folder contains utilities for data processing, validation, and manipulation in financial applications.

-   :material-tools: __[Economic Calendar](Economic Calendar.md)__

    **This utility does NOT use any external APIs.** All data is managed locally for learning and experimentation.

-   :material-tools: __[Historical Data](Historical Data.md)__

    A Node.js script that fetches historical bars (OHLCV data) for stocks or crypto from the Alpaca Market Data API. It prompts interactively for the symbol type, symbol, timeframe, and date range, then prints the results as JSON.

-   :material-tools: __[Logging](Logging.md)__

    A pair of minimal, dependency-light logging utilities implemented in both Python and JavaScript. Each supports adding, reading, editing, and deleting log entries through an interactive command-line menu. All entries are persisted to a plain-text `log.txt` file in the working directory.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
