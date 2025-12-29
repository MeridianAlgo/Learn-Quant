"""
API Utilities for Financial Applications

This module provides comprehensive API utilities for financial applications,
including HTTP request handling, retry logic, API key generation and hashing,
and error handling.

Author: Generated for Learn-Quant Project
Version: 1.0.0
"""

import json
import hashlib
import secrets
import time
from typing import Dict, Any, Optional

try:
    import requests
except ImportError:
    print("Warning: requests library not found. Install with: pip install requests")
    requests = None


def make_api_request(url: str, method: str = 'GET', headers: Dict[str, str] = None, 
                    params: Dict[str, Any] = None, data: Dict[str, Any] = None,
                    timeout: int = 30) -> Dict[str, Any]:
    """
    Make HTTP API request with error handling.
    
    Args:
        url: API endpoint URL
        method: HTTP method (GET, POST, PUT, DELETE)
        headers: Request headers
        params: Query parameters
        data: Request body data
        timeout: Request timeout in seconds
        
    Returns:
        Response data as dictionary
        
    Raises:
        requests.RequestException: If request fails
        
    Example:
        >>> response = make_api_request("https://api.example.com/stocks/AAPL")
        >>> print(response["price"])
        150.25
    """
    if requests is None:
        raise ImportError("requests library is required. Install with: pip install requests")
    
    try:
        response = requests.request(
            method=method.upper(),
            url=url,
            headers=headers or {},
            params=params,
            json=data,
            timeout=timeout
        )
        
        response.raise_for_status()
        
        # Try to parse as JSON, return text if not possible
        try:
            return response.json()
        except json.JSONDecodeError:
            return {'text': response.text}
    
    except requests.exceptions.Timeout:
        raise requests.RequestException(f"Request timeout after {timeout} seconds")
    except requests.exceptions.ConnectionError:
        raise requests.RequestException("Connection error")
    except requests.exceptions.HTTPError as e:
        raise requests.RequestException(f"HTTP error: {e}")
    except Exception as e:
        raise requests.RequestException(f"Unexpected error: {e}")


def retry_api_request(url: str, max_retries: int = 3, delay: float = 1.0, 
                     **kwargs) -> Dict[str, Any]:
    """
    Make API request with retry logic.
    
    Args:
        url: API endpoint URL
        max_retries: Maximum number of retries
        delay: Delay between retries in seconds
        **kwargs: Additional arguments for make_api_request
        
    Returns:
        Response data as dictionary
        
    Example:
        >>> response = retry_api_request("https://api.example.com/data", max_retries=5)
        >>> print(response["status"])
        "success"
    """
    for attempt in range(max_retries):
        try:
            return make_api_request(url, **kwargs)
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise e
            
            time.sleep(delay * (2 ** attempt))  # Exponential backoff
    
    raise requests.RequestException("Max retries exceeded")


def generate_api_key(length: int = 32) -> str:
    """
    Generate secure API key.
    
    Args:
        length: Length of API key
        
    Returns:
        Random API key string
        
    Example:
        >>> api_key = generate_api_key()
        >>> print(len(api_key))
        32
        >>> api_key = generate_api_key(64)
        >>> print(len(api_key))
        64
    """
    return secrets.token_urlsafe(length)


def hash_api_key(api_key: str, salt: str = None) -> str:
    """
    Hash API key for secure storage.
    
    Args:
        api_key: API key to hash
        salt: Salt for hashing (generated if not provided)
        
    Returns:
        Hashed API key
        
    Example:
        >>> hashed = hash_api_key("secret_key_123")
        >>> print(hashed.startswith("salt:"))
        True
    """
    if salt is None:
        salt = secrets.token_hex(16)
    
    hash_obj = hashlib.sha256((api_key + salt).encode())
    return f"{salt}:{hash_obj.hexdigest()}"


def verify_api_key(api_key: str, hashed_key: str) -> bool:
    """
    Verify API key against hashed version.
    
    Args:
        api_key: Plain API key to verify
        hashed_key: Hashed API key (salt:hash format)
        
    Returns:
        True if API key matches, False otherwise
        
    Example:
        >>> hashed = hash_api_key("secret_key")
        >>> verify_api_key("secret_key", hashed)
        True
        >>> verify_api_key("wrong_key", hashed)
        False
    """
    try:
        salt, stored_hash = hashed_key.split(':', 1)
        hash_obj = hashlib.sha256((api_key + salt).encode())
        return hash_obj.hexdigest() == stored_hash
    except ValueError:
        return False


def demo_api_utils():
    """Demonstrate API utilities (without making actual HTTP requests)."""
    print("=" * 60)
    print("API UTILITIES DEMONSTRATION")
    print("=" * 60)
    
    # API key generation
    print("\nAPI Key Generation:")
    for length in [16, 32, 64]:
        api_key = generate_api_key(length)
        print(f"  {length}-character key: {api_key}")
    
    # API key hashing and verification
    print("\nAPI Key Hashing and Verification:")
    original_key = "test_api_key_12345"
    hashed = hash_api_key(original_key)
    print(f"  Original key: {original_key}")
    print(f"  Hashed key: {hashed}")
    
    # Verification
    is_valid = verify_api_key(original_key, hashed)
    print(f"  Verification (correct key): {is_valid}")
    
    is_valid = verify_api_key("wrong_key", hashed)
    print(f"  Verification (wrong key): {is_valid}")
    
    # Mock API request examples (without actual HTTP calls)
    print("\nAPI Request Examples (Mock):")
    print("  GET request example:")
    print("    url: 'https://api.example.com/stocks/AAPL'")
    print("    method: 'GET'")
    print("    headers: {'Authorization': 'Bearer token123'}")
    
    print("\n  POST request example:")
    print("    url: 'https://api.example.com/orders'")
    print("    method: 'POST'")
    print("    data: {'symbol': 'AAPL', 'quantity': 100}")
    
    print("\n  Retry logic example:")
    print("    max_retries: 3")
    print("    delay: 1.0 (exponential backoff)")
    print("    Retry delays: 1.0s, 2.0s, 4.0s")
    
    # Error handling examples
    print("\nError Handling:")
    print("  Timeout errors: Clear timeout message")
    print("  Connection errors: Network issue notification")
    print("  HTTP errors: Status code and message")
    print("  JSON errors: Fallback to text response")
    
    # Security considerations
    print("\nSecurity Best Practices:")
    print("  ✓ Use HTTPS endpoints")
    print("  ✓ Hash API keys before storage")
    print("  ✓ Use environment variables for secrets")
    print("  ✓ Implement rate limiting")
    print("  ✓ Validate all responses")
    print("  ✓ Use proper authentication")


def main():
    """Main function to run demonstrations."""
    demo_api_utils()
    print("\n🎉 API utilities demonstration complete!")
    print("\nNote: Install 'requests' library to use HTTP functionality:")
    print("pip install requests")


if __name__ == "__main__":
    main()
