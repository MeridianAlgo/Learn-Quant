# API Utilities

This module provides comprehensive API utilities for financial applications, including HTTP request handling, retry logic, API key generation and hashing, and error handling.

## Functions

### `make_api_request(url: str, method: str = 'GET', headers: Dict[str, str] = None, params: Dict[str, Any] = None, data: Dict[str, Any] = None, timeout: int = 30) -> Dict[str, Any]`
Makes HTTP API request with error handling.

**Parameters:**
- `url`: API endpoint URL
- `method`: HTTP method (GET, POST, PUT, DELETE)
- `headers`: Request headers
- `params`: Query parameters
- `data`: Request body data
- `timeout`: Request timeout in seconds

**Returns:**
- Response data as dictionary

**Raises:**
- `requests.RequestException`: If request fails

**Example:**
```python
>>> response = make_api_request("https://api.example.com/stocks/AAPL")
>>> print(response["price"])
150.25
```

### `retry_api_request(url: str, max_retries: int = 3, delay: float = 1.0, **kwargs) -> Dict[str, Any]`
Makes API request with retry logic.

**Parameters:**
- `url`: API endpoint URL
- `max_retries`: Maximum number of retries
- `delay`: Delay between retries in seconds
- `**kwargs`: Additional arguments for make_api_request

**Returns:**
- Response data as dictionary

**Example:**
```python
>>> response = retry_api_request("https://api.example.com/data", max_retries=5)
>>> print(response["status"])
"success"
```

### `generate_api_key(length: int = 32) -> str`
Generates secure API key.

**Parameters:**
- `length`: Length of API key

**Returns:**
- Random API key string

**Example:**
```python
>>> api_key = generate_api_key()
>>> print(len(api_key))
32
>>> api_key = generate_api_key(64)
>>> print(len(api_key))
64
```

### `hash_api_key(api_key: str, salt: str = None) -> str`
Hashes API key for secure storage.

**Parameters:**
- `api_key`: API key to hash
- `salt`: Salt for hashing (generated if not provided)

**Returns:**
- Hashed API key

**Example:**
```python
>>> hashed = hash_api_key("secret_key_123")
>>> print(hashed.startswith("salt:"))
True
```

## Usage

```python
from api_utils import (
    make_api_request, retry_api_request, generate_api_key, hash_api_key
)

# Basic API request
response = make_api_request(
    "https://api.example.com/stocks/AAPL",
    headers={"Authorization": "Bearer your_token"}
)
print(f"Stock price: ${response['price']}")

# POST request with data
order_data = {
    "symbol": "AAPL",
    "quantity": 100,
    "order_type": "market"
}
response = make_api_request(
    "https://api.example.com/orders",
    method="POST",
    data=order_data
)

# Retry failed requests
response = retry_api_request(
    "https://api.example.com/unreliable-endpoint",
    max_retries=5,
    delay=2.0
)

# Generate API keys for users
user_api_key = generate_api_key(32)
print(f"Generated API key: {user_api_key}")

# Hash API keys for secure storage
hashed_key = hash_api_key(user_api_key)
print(f"Hashed for storage: {hashed_key}")
```

## Installation

Requires the `requests` library:

```bash
pip install requests
```

## Testing

Run the module directly to see demonstrations:

```bash
python api_utils.py
```

## Common Use Cases

- **Financial Data APIs**: Fetch stock prices, market data, and economic indicators
- **Trading Platforms**: Execute orders and retrieve account information
- **Payment Processing**: Handle payment gateway requests
- **Third-party Integrations**: Connect to external financial services
- **API Key Management**: Generate and secure API keys for authentication
- **Error Handling**: Implement robust retry logic for unreliable APIs
- **Data Validation**: Validate API responses and handle errors gracefully

## Notes

- All requests include proper error handling with descriptive error messages
- Retry logic uses exponential backoff to avoid overwhelming servers
- API key generation uses cryptographically secure random numbers
- Hashing includes salt for security (store the full returned string)
- Timeout is set to 30 seconds by default to prevent hanging requests
- JSON responses are automatically parsed; text responses returned as-is

## Security Considerations

- Never store raw API keys in your code
- Use environment variables or secure configuration for API keys
- Always hash API keys before storing them in databases
- Use HTTPS endpoints whenever possible
- Implement proper authentication and authorization
- Consider rate limiting to avoid API abuse
- Validate and sanitize all API responses
