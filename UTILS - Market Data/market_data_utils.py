"""
Market Data Utilities for Financial Applications

This module provides comprehensive market data utilities for financial applications,
including price data processing, technical analysis, market sentiment analysis,
data validation, and market timing indicators.

Author: Generated for Learn-Quant Project
Version: 1.0.0
"""

import math
import statistics
from typing import Any, Dict, List, Union

try:
    import numpy as np
    from scipy import signal
except ImportError:
    print(
        "Warning: numpy/scipy not found. Some functions may not work optimally. Install with: pip install numpy scipy"
    )
    np = None
    signal = None


def calculate_returns(prices: List[float], method: str = "simple") -> List[float]:
    """
    Calculate returns from price series.

    Args:
        prices: List of prices
        method: Return calculation method ('simple' or 'log')

    Returns:
        List of returns

    Example:
        >>> prices = [100, 105, 102, 108]
        >>> returns = calculate_returns(prices)
        >>> print(returns)
        [0.05, -0.0286, 0.0588]
    """
    if not prices or len(prices) < 2:
        return []

    returns = []
    for i in range(1, len(prices)):
        if prices[i - 1] == 0:
            returns.append(0.0)
            continue

        if method == "simple":
            ret = (prices[i] - prices[i - 1]) / prices[i - 1]
        elif method == "log":
            ret = math.log(prices[i] / prices[i - 1])
        else:
            raise ValueError(f"Unknown method: {method}")

        returns.append(ret)

    return returns


def detect_outliers(data: List[float], method: str = "iqr", threshold: float = 1.5) -> List[int]:
    """
    Detect outliers in data series.

    Args:
        data: List of numerical data
        method: Detection method ('iqr', 'zscore', 'modified_zscore')
        threshold: Threshold for outlier detection

    Returns:
        List of outlier indices

    Example:
        >>> data = [100, 102, 105, 98, 150, 103]  # 150 is outlier
        >>> outliers = detect_outliers(data)
        >>> print(outliers)
        [4]
    """
    if not data:
        return []

    outliers = []

    if method == "iqr":
        q1 = statistics.median(data[: len(data) // 2])
        q3 = statistics.median(data[len(data) // 2 + (len(data) % 2) :])
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        for i, value in enumerate(data):
            if value < lower_bound or value > upper_bound:
                outliers.append(i)

    elif method == "zscore":
        mean_val = statistics.mean(data)
        std_val = statistics.stdev(data) if len(data) > 1 else 0

        if std_val == 0:
            return outliers

        for i, value in enumerate(data):
            zscore = abs((value - mean_val) / std_val)
            if zscore > threshold:
                outliers.append(i)

    elif method == "modified_zscore":
        median_val = statistics.median(data)
        mad = statistics.median([abs(x - median_val) for x in data])

        if mad == 0:
            return outliers

        for i, value in enumerate(data):
            modified_zscore = 0.6745 * (value - median_val) / mad
            if abs(modified_zscore) > threshold:
                outliers.append(i)

    else:
        raise ValueError(f"Unknown method: {method}")

    return outliers


def fill_missing_data(data: List[Union[float, None]], method: str = "linear") -> List[float]:
    """
    Fill missing data points in time series.

    Args:
        data: List with None values for missing data
        method: Fill method ('linear', 'forward', 'backward', 'mean')

    Returns:
        List with filled values

    Example:
        >>> data = [100, None, 105, None, 110]
        >>> filled = fill_missing_data(data, 'linear')
        >>> print(filled)
        [100, 102.5, 105, 107.5, 110]
    """
    if not data:
        return []

    filled_data = []
    n = len(data)

    if method == "linear":
        # Linear interpolation
        for i, value in enumerate(data):
            if value is not None:
                filled_data.append(value)
            else:
                # Find previous and next non-None values
                prev_idx = i - 1
                while prev_idx >= 0 and data[prev_idx] is None:
                    prev_idx -= 1

                next_idx = i + 1
                while next_idx < n and data[next_idx] is None:
                    next_idx += 1

                if prev_idx >= 0 and next_idx < n:
                    # Linear interpolation
                    prev_val = data[prev_idx]
                    next_val = data[next_idx]
                    ratio = (i - prev_idx) / (next_idx - prev_idx)
                    filled_val = prev_val + ratio * (next_val - prev_val)
                    filled_data.append(filled_val)
                elif prev_idx >= 0:
                    # Forward fill
                    filled_data.append(data[prev_idx])
                elif next_idx < n:
                    # Backward fill
                    filled_data.append(data[next_idx])
                else:
                    # No data available, use 0
                    filled_data.append(0.0)

    elif method == "forward":
        # Forward fill
        last_valid = None
        for value in data:
            if value is not None:
                filled_data.append(value)
                last_valid = value
            else:
                filled_data.append(last_valid if last_valid is not None else 0.0)

    elif method == "backward":
        # Backward fill
        # First pass: find next valid values
        next_valid = None
        for i in range(n - 1, -1, -1):
            if data[i] is not None:
                next_valid = data[i]
                break

        for i, value in enumerate(data):
            if value is not None:
                filled_data.append(value)
                next_valid = value
            else:
                filled_data.append(next_valid if next_valid is not None else 0.0)

    elif method == "mean":
        # Mean fill
        valid_values = [v for v in data if v is not None]
        mean_val = statistics.mean(valid_values) if valid_values else 0.0

        for value in data:
            filled_data.append(value if value is not None else mean_val)

    else:
        raise ValueError(f"Unknown method: {method}")

    return filled_data


def calculate_market_sentiment(news_data: List[Dict[str, str]], keywords: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Calculate market sentiment from news data.

    Args:
        news_data: List of news articles with 'title' and 'content'
        keywords: Sentiment keywords {'positive': [...], 'negative': [...]}

    Returns:
        Sentiment scores

    Example:
        >>> news = [{"title": "Stocks rally on strong earnings", "content": "..."}]
        >>> keywords = {"positive": ["rally", "strong"], "negative": ["crash", "weak"]}
        >>> sentiment = calculate_market_sentiment(news, keywords)
        >>> print(f"Sentiment: {sentiment['overall']:.2f}")
        0.75
    """
    if not news_data or not keywords:
        return {"overall": 0.0, "positive": 0.0, "negative": 0.0}

    positive_keywords = [k.lower() for k in keywords.get("positive", [])]
    negative_keywords = [k.lower() for k in keywords.get("negative", [])]

    total_positive = 0
    total_negative = 0
    total_words = 0

    for article in news_data:
        title = article.get("title", "").lower()
        content = article.get("content", "").lower()
        text = f"{title} {content}"

        words = text.split()
        total_words += len(words)

        for word in words:
            if word in positive_keywords:
                total_positive += 1
            elif word in negative_keywords:
                total_negative += 1

    if total_words == 0:
        return {"overall": 0.0, "positive": 0.0, "negative": 0.0}

    positive_score = total_positive / total_words
    negative_score = total_negative / total_words

    # Overall sentiment: -1 (very negative) to +1 (very positive)
    if positive_score + negative_score == 0:
        overall_sentiment = 0.0
    else:
        overall_sentiment = (positive_score - negative_score) / (positive_score + negative_score)

    return {
        "overall": overall_sentiment,
        "positive": positive_score,
        "negative": negative_score,
        "total_articles": len(news_data),
        "total_words": total_words,
    }


def validate_market_data(data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """
    Validate market data against schema.

    Args:
        data: Market data dictionary
        schema: Validation schema

    Returns:
        True if data is valid

    Example:
        >>> data = {"symbol": "AAPL", "price": 150.25, "volume": 1000000}
        >>> schema = {"symbol": str, "price": (int, float), "volume": int}
        >>> is_valid = validate_market_data(data, schema)
        >>> print(is_valid)
        True
    """
    if not data or not schema:
        return False

    for field, expected_type in schema.items():
        if field not in data:
            return False

        value = data[field]

        # Handle tuple of types (e.g., (int, float))
        if isinstance(expected_type, tuple):
            if not isinstance(value, expected_type):
                return False
        else:
            if not isinstance(value, expected_type):
                return False

        # Additional validations
        if field == "price" and isinstance(value, (int, float)):
            if value <= 0:
                return False
        elif field == "volume" and isinstance(value, int):
            if value < 0:
                return False
        elif field == "symbol" and isinstance(value, str):
            if not value or len(value) > 10:
                return False

    return True


def calculate_market_timing_indicators(prices: List[float], volumes: List[float]) -> Dict[str, float]:
    """
    Calculate market timing indicators.

    Args:
        prices: List of prices
        volumes: List of volumes

    Returns:
        Dictionary of timing indicators

    Example:
        >>> prices = [100, 105, 102, 108, 110]
        >>> volumes = [1000, 1200, 800, 1500, 1300]
        >>> indicators = calculate_market_timing_indicators(prices, volumes)
        >>> print(f"Volume Price Trend: {indicators['volume_price_trend']:.2f}")
    """
    if len(prices) != len(volumes) or len(prices) < 2:
        return {}

    returns = calculate_returns(prices)

    # Volume-Price Trend (VPT)
    vpt = 0.0
    for i, ret in enumerate(returns):
        vpt += volumes[i + 1] * ret

    # On-Balance Volume (OBV)
    obv = [volumes[0]]
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            obv.append(obv[-1] + volumes[i])
        elif prices[i] < prices[i - 1]:
            obv.append(obv[-1] - volumes[i])
        else:
            obv.append(obv[-1])

    # Money Flow Index (MFI) - simplified version
    typical_prices = [(prices[i] + prices[i] + volumes[i]) / 3 for i in range(len(prices))]  # Simplified

    # Accumulation/Distribution Line
    ad_line = 0.0
    for i in range(1, len(prices)):
        high = prices[i] * 1.01  # Simplified - no high/low data
        low = prices[i] * 0.99
        clv = ((prices[i] - low) - (high - prices[i])) / (high - low) * volumes[i]
        ad_line += clv

    # Price-Volume Trend
    price_change = (prices[-1] - prices[0]) / prices[0]
    volume_change = (volumes[-1] - volumes[0]) / volumes[0] if volumes[0] != 0 else 0

    return {
        "volume_price_trend": vpt,
        "obv": obv[-1],
        "accumulation_distribution": ad_line,
        "price_change_pct": price_change,
        "volume_change_pct": volume_change,
        "avg_volume": statistics.mean(volumes),
        "volume_volatility": statistics.stdev(volumes) if len(volumes) > 1 else 0.0,
    }


def smooth_data(data: List[float], method: str = "moving_average", window: int = 5) -> List[float]:
    """
    Smooth noisy data using various methods.

    Args:
        data: List of data points
        method: Smoothing method ('moving_average', 'exponential', 'savgol')
        window: Window size for smoothing

    Returns:
        Smoothed data

    Example:
        >>> noisy_data = [100, 102, 98, 105, 103, 107, 101]
        >>> smooth = smooth_data(noisy_data, 'moving_average', 3)
        >>> print(smooth)
        [None, 100.0, 101.67, 102.0, 105.33, 103.67, 103.67]
    """
    if not data or window < 1:
        return data.copy()

    if method == "moving_average":
        smoothed = []
        for i in range(len(data)):
            if i < window - 1:
                smoothed.append(None)
            else:
                window_data = data[i - window + 1 : i + 1]
                smoothed.append(statistics.mean(window_data))
        return smoothed

    elif method == "exponential":
        alpha = 2.0 / (window + 1)
        smoothed = [data[0]]
        for i in range(1, len(data)):
            smoothed.append(alpha * data[i] + (1 - alpha) * smoothed[-1])
        return smoothed

    elif method == "savgol":
        if np is not None and signal is not None:
            try:
                smoothed_array = signal.savgol_filter(data, window, 3)
                return smoothed_array.tolist()
            except:
                pass

        # Fallback to moving average
        return smooth_data(data, "moving_average", window)

    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_market_microstructure(
    tick_data: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Calculate market microstructure indicators.

    Args:
        tick_data: List of tick data with price, volume, timestamp

    Returns:
        Microstructure metrics

    Example:
        >>> ticks = [{"price": 150.25, "volume": 100, "timestamp": "2024-01-01T09:30:00"}]
        >>> micro = calculate_market_microstructure(ticks)
        >>> print(f"Average tick size: {micro['avg_tick_size']:.2f}")
    """
    if not tick_data:
        return {}

    prices = [tick["price"] for tick in tick_data if "price" in tick]
    volumes = [tick["volume"] for tick in tick_data if "volume" in tick]

    if not prices:
        return {}

    # Basic metrics
    avg_price = statistics.mean(prices)
    avg_volume = statistics.mean(volumes) if volumes else 0

    # Price volatility
    if len(prices) > 1:
        price_volatility = statistics.stdev(prices)
    else:
        price_volatility = 0

    # Tick frequency (ticks per minute)
    if len(tick_data) > 1:
        timestamps = [tick.get("timestamp") for tick in tick_data if "timestamp" in tick]
        if timestamps:
            # Simplified - assuming timestamps are in order
            duration_minutes = len(tick_data)  # Simplified assumption
            tick_frequency = len(tick_data) / duration_minutes if duration_minutes > 0 else 0
        else:
            tick_frequency = 0
    else:
        tick_frequency = 0

    # Price changes
    price_changes = []
    for i in range(1, len(prices)):
        price_changes.append(abs(prices[i] - prices[i - 1]))

    avg_price_change = statistics.mean(price_changes) if price_changes else 0

    # Volume-weighted average price (VWAP)
    if volumes and len(volumes) == len(prices):
        total_value = sum(p * v for p, v in zip(prices, volumes))
        total_volume = sum(volumes)
        vwap = total_value / total_volume if total_volume > 0 else avg_price
    else:
        vwap = avg_price

    return {
        "avg_price": avg_price,
        "avg_volume": avg_volume,
        "price_volatility": price_volatility,
        "tick_frequency": tick_frequency,
        "avg_price_change": avg_price_change,
        "vwap": vwap,
        "total_ticks": len(tick_data),
        "price_range": max(prices) - min(prices) if len(prices) > 1 else 0,
    }


def demo_market_data_utils():
    """Demonstrate market data utilities."""
    print("=" * 60)
    print("MARKET DATA UTILITIES DEMONSTRATION")
    print("=" * 60)

    # Sample price data
    prices = [100, 105, 102, 108, 110, 95, 98, 112, 115, 109]
    volumes = [1000, 1200, 800, 1500, 1300, 900, 1100, 1400, 1200, 1000]

    print("\n1. Return Calculation:")
    simple_returns = calculate_returns(prices, "simple")
    log_returns = calculate_returns(prices, "log")
    print(f"  Simple returns: {[round(r, 4) for r in simple_returns[:3]]}...")
    print(f"  Log returns: {[round(r, 4) for r in log_returns[:3]]}...")

    print("\n2. Outlier Detection:")
    data_with_outlier = [100, 102, 105, 98, 150, 103, 107]  # 150 is outlier
    outliers_iqr = detect_outliers(data_with_outlier, "iqr")
    outliers_zscore = detect_outliers(data_with_outlier, "zscore", 2.0)
    print(f"  Data: {data_with_outlier}")
    print(f"  IQR outliers: {outliers_iqr}")
    print(f"  Z-score outliers: {outliers_zscore}")

    print("\n3. Missing Data Handling:")
    data_with_gaps = [100, None, 105, None, None, 110, 108]
    filled_linear = fill_missing_data(data_with_gaps, "linear")
    filled_forward = fill_missing_data(data_with_gaps, "forward")
    print(f"  Original: {data_with_gaps}")
    print(f"  Linear fill: {[round(x, 2) if x is not None else None for x in filled_linear]}")
    print(f"  Forward fill: {filled_forward}")

    print("\n4. Market Sentiment Analysis:")
    news_data = [
        {
            "title": "Markets rally on positive earnings",
            "content": "Strong growth reported across sectors",
        },
        {
            "title": "Concerns over inflation rise",
            "content": "Investors worried about price pressures",
        },
        {
            "title": "Tech stocks surge higher",
            "content": "Technology sector shows strong momentum",
        },
    ]
    keywords = {
        "positive": ["rally", "positive", "strong", "growth", "surge", "momentum"],
        "negative": ["concerns", "worried", "inflation", "pressures", "fears"],
    }
    sentiment = calculate_market_sentiment(news_data, keywords)
    print(f"  Overall sentiment: {sentiment['overall']:.3f}")
    print(f"  Positive score: {sentiment['positive']:.3f}")
    print(f"  Negative score: {sentiment['negative']:.3f}")

    print("\n5. Data Validation:")
    market_data = {
        "symbol": "AAPL",
        "price": 150.25,
        "volume": 1000000,
        "timestamp": "2024-01-01",
    }
    schema = {"symbol": str, "price": (int, float), "volume": int, "timestamp": str}
    is_valid = validate_market_data(market_data, schema)
    print(f"  Data valid: {is_valid}")

    # Test invalid data
    invalid_data = {"symbol": "AAPL", "price": -150.25, "volume": -1000}
    is_invalid = validate_market_data(invalid_data, schema)
    print(f"  Invalid data valid: {is_invalid}")

    print("\n6. Market Timing Indicators:")
    timing = calculate_market_timing_indicators(prices, volumes)
    print(f"  Volume-Price Trend: {timing['volume_price_trend']:.2f}")
    print(f"  Price Change: {timing['price_change_pct']:.2%}")
    print(f"  Volume Change: {timing['volume_change_pct']:.2%}")
    print(f"  Average Volume: {timing['avg_volume']:.0f}")

    print("\n7. Data Smoothing:")
    noisy_data = [100, 102, 98, 105, 103, 107, 101, 106, 104, 108]
    smooth_ma = smooth_data(noisy_data, "moving_average", 3)
    smooth_exp = smooth_data(noisy_data, "exponential", 3)
    print(f"  Original: {noisy_data[:5]}...")
    print(f"  Moving Avg: {[round(x, 2) if x is not None else None for x in smooth_ma[:5]]}...")
    print(f"  Exponential: {[round(x, 2) for x in smooth_exp[:5]]}...")

    print("\n8. Market Microstructure:")
    tick_data = [
        {"price": 150.25, "volume": 100, "timestamp": "2024-01-01T09:30:00"},
        {"price": 150.26, "volume": 200, "timestamp": "2024-01-01T09:30:01"},
        {"price": 150.24, "volume": 150, "timestamp": "2024-01-01T09:30:02"},
        {"price": 150.27, "volume": 300, "timestamp": "2024-01-01T09:30:03"},
        {"price": 150.25, "volume": 180, "timestamp": "2024-01-01T09:30:04"},
    ]
    micro = calculate_market_microstructure(tick_data)
    print(f"  Average Price: ${micro['avg_price']:.2f}")
    print(f"  Average Volume: {micro['avg_volume']:.0f}")
    print(f"  Price Volatility: {micro['price_volatility']:.4f}")
    print(f"  VWAP: ${micro['vwap']:.2f}")
    print(f"  Total Ticks: {micro['total_ticks']}")


def main():
    """Main function to run demonstrations."""
    demo_market_data_utils()
    print("\n🎉 Market data utilities demonstration complete!")


if __name__ == "__main__":
    main()
