# Market Data Utilities

This module provides comprehensive market data utilities for financial applications, including price data processing, technical analysis, market sentiment analysis, data validation, and market timing indicators.

## Functions

### `calculate_returns(prices: List[float], method: str = 'simple') -> List[float]`
Calculates returns from price series.

**Parameters:**
- `prices`: List of prices
- `method`: Return calculation method ('simple' or 'log')

**Returns:**
- List of returns

**Example:**
```python
>>> prices = [100, 105, 102, 108]
>>> returns = calculate_returns(prices)
>>> print(returns)
[0.05, -0.0286, 0.0588]
```

### `detect_outliers(data: List[float], method: str = 'iqr', threshold: float = 1.5) -> List[int]`
Detects outliers in data series.

**Parameters:**
- `data`: List of numerical data
- `method`: Detection method ('iqr', 'zscore', 'modified_zscore')
- `threshold`: Threshold for outlier detection

**Returns:**
- List of outlier indices

**Example:**
```python
>>> data = [100, 102, 105, 98, 150, 103]  # 150 is outlier
>>> outliers = detect_outliers(data)
>>> print(outliers)
[4]
```

### `fill_missing_data(data: List[float], method: str = 'linear') -> List[float]`
Fills missing data points in time series.

**Parameters:**
- `data`: List with None values for missing data
- `method`: Fill method ('linear', 'forward', 'backward', 'mean')

**Returns:**
- List with filled values

**Example:**
```python
>>> data = [100, None, 105, None, 110]
>>> filled = fill_missing_data(data, 'linear')
>>> print(filled)
[100, 102.5, 105, 107.5, 110]
```

### `calculate_market_sentiment(news_data: List[Dict[str, str]], keywords: Dict[str, List[str]]) -> Dict[str, float]`
Calculates market sentiment from news data.

**Parameters:**
- `news_data`: List of news articles with 'title' and 'content'
- `keywords`: Sentiment keywords {'positive': [...], 'negative': [...]}

**Returns:**
- Sentiment scores

**Example:**
```python
>>> news = [{"title": "Stocks rally on strong earnings", "content": "..."}]
>>> keywords = {"positive": ["rally", "strong"], "negative": ["crash", "weak"]}
>>> sentiment = calculate_market_sentiment(news, keywords)
>>> print(f"Sentiment: {sentiment['overall']:.2f}")
0.75
```

### `validate_market_data(data: Dict[str, Any], schema: Dict[str, Any]) -> bool`
Validates market data against schema.

**Parameters:**
- `data`: Market data dictionary
- `schema`: Validation schema

**Returns:**
- True if data is valid

**Example:**
```python
>>> data = {"symbol": "AAPL", "price": 150.25, "volume": 1000000}
>>> schema = {"symbol": str, "price": (int, float), "volume": int}
>>> is_valid = validate_market_data(data, schema)
>>> print(is_valid)
True
```

### `calculate_market_timing_indicators(prices: List[float], volumes: List[float]) -> Dict[str, float]`
Calculates market timing indicators.

**Parameters:**
- `prices`: List of prices
- `volumes`: List of volumes

**Returns:**
- Dictionary of timing indicators

**Example:**
```python
>>> prices = [100, 105, 102, 108, 110]
>>> volumes = [1000, 1200, 800, 1500, 1300]
>>> indicators = calculate_market_timing_indicators(prices, volumes)
>>> print(f"Volume Price Trend: {indicators['volume_price_trend']:.2f}")
```

### `smooth_data(data: List[float], method: str = 'moving_average', window: int = 5) -> List[float]`
Smooths noisy data using various methods.

**Parameters:**
- `data`: List of data points
- `method`: Smoothing method ('moving_average', 'exponential', 'savgol')
- `window`: Window size for smoothing

**Returns:**
- Smoothed data

**Example:**
```python
>>> noisy_data = [100, 102, 98, 105, 103, 107, 101]
>>> smooth = smooth_data(noisy_data, 'moving_average', 3)
>>> print(smooth)
[None, 100.0, 101.67, 102.0, 105.33, 103.67, 103.67]
```

### `calculate_market_microstructure(tick_data: List[Dict[str, Any]]) -> Dict[str, float]`
Calculates market microstructure indicators.

**Parameters:**
- `tick_data`: List of tick data with price, volume, timestamp

**Returns:**
- Microstructure metrics

**Example:**
```python
>>> ticks = [{"price": 150.25, "volume": 100, "timestamp": "2024-01-01T09:30:00"}]
>>> micro = calculate_market_microstructure(ticks)
>>> print(f"Average tick size: {micro['avg_tick_size']:.2f}")
```

## Usage

```python
from market_data_utils import (
    calculate_returns, detect_outliers, fill_missing_data,
    calculate_market_sentiment, validate_market_data,
    calculate_market_timing_indicators, smooth_data,
    calculate_market_microstructure
)

# Price data analysis
prices = [100, 105, 102, 108, 110, 95, 98, 112]
returns = calculate_returns(prices, 'log')
print(f"Log returns: {returns}")

# Outlier detection
outliers = detect_outliers(prices, 'iqr', 2.0)
print(f"Outlier indices: {outliers}")

# Missing data handling
data_with_gaps = [100, None, 105, None, None, 110, 108]
filled_data = fill_missing_data(data_with_gaps, 'linear')
print(f"Filled data: {filled_data}")

# Sentiment analysis
news_data = [
    {"title": "Markets rally on positive earnings", "content": "Strong growth reported"},
    {"title": "Concerns over inflation rise", "content": "Investors worried about prices"}
]
keywords = {
    "positive": ["rally", "positive", "strong", "growth"],
    "negative": ["concerns", "worried", "inflation", "rise"]
}
sentiment = calculate_market_sentiment(news_data, keywords)
print(f"Market sentiment: {sentiment}")

# Data validation
market_data = {"symbol": "AAPL", "price": 150.25, "volume": 1000000, "timestamp": "2024-01-01"}
schema = {"symbol": str, "price": (int, float), "volume": int, "timestamp": str}
is_valid = validate_market_data(market_data, schema)
print(f"Data valid: {is_valid}")

# Market timing
volumes = [1000, 1200, 800, 1500, 1300, 900, 1100]
timing_indicators = calculate_market_timing_indicators(prices, volumes)
print(f"Timing indicators: {timing_indicators}")

# Data smoothing
smooth_prices = smooth_data(prices, 'exponential', 5)
print(f"Smoothed prices: {smooth_prices}")
```

## Installation

Requires numpy and scipy for advanced calculations:

```bash
pip install numpy scipy
```

## Testing

Run the module directly to see demonstrations:

```bash
python market_data_utils.py
```

## Common Use Cases

- **Data Preprocessing**: Clean and validate market data
- **Technical Analysis**: Calculate indicators and timing signals
- **Sentiment Analysis**: Analyze news and social media sentiment
- **Quality Control**: Detect outliers and handle missing data
- **Market Research**: Analyze market microstructure patterns
- **Algorithm Development**: Prepare data for trading algorithms
- **Risk Management**: Validate data quality and detect anomalies

## Notes

- Return calculations support both simple and logarithmic returns
- Outlier detection uses multiple methods (IQR, Z-score, Modified Z-score)
- Missing data interpolation supports various methods
- Sentiment analysis uses keyword-based approach
- Market timing indicators include volume-price relationships
- Data smoothing helps reduce noise in technical analysis
- Microstructure analysis requires high-frequency tick data

## Data Quality Best Practices

- Always validate incoming market data before processing
- Handle outliers appropriately (remove or adjust based on context)
- Use appropriate interpolation methods for missing data
- Monitor data quality metrics continuously
- Implement automated alerts for data quality issues
- Maintain data lineage and audit trails
- Use multiple data sources for critical applications
