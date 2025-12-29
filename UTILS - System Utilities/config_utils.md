# Configuration Utilities

This module provides comprehensive configuration management utilities for financial applications, including JSON configuration loading/saving, dot notation access, configuration merging, and hierarchical configuration management.

## Functions

### `load_config(config_path: str) -> Dict[str, Any]`
Loads configuration from JSON file.

**Parameters:**
- `config_path`: Path to configuration file

**Returns:**
- Configuration dictionary

**Example:**
```python
>>> config = load_config("config.json")
>>> print(config["api_key"])
"your_api_key_here"
```

### `save_config(config: Dict[str, Any], config_path: str) -> None`
Saves configuration to JSON file.

**Parameters:**
- `config`: Configuration dictionary
- `config_path`: Path to save configuration

**Example:**
```python
>>> config = {"api_key": "key123", "timeout": 30}
>>> save_config(config, "config.json")
```

### `get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any`
Gets configuration value using dot notation.

**Parameters:**
- `config`: Configuration dictionary
- `key_path`: Dot-separated key path (e.g., 'database.host')
- `default`: Default value if key not found

**Returns:**
- Configuration value

**Example:**
```python
>>> config = {"database": {"host": "localhost", "port": 5432}}
>>> get_config_value(config, "database.host")
"localhost"
>>> get_config_value(config, "database.password", "default")
"default"
```

### `set_config_value(config: Dict[str, Any], key_path: str, value: Any) -> None`
Sets configuration value using dot notation.

**Parameters:**
- `config`: Configuration dictionary
- `key_path`: Dot-separated key path (e.g., 'database.host')
- `value`: Value to set

**Example:**
```python
>>> config = {}
>>> set_config_value(config, "database.host", "localhost")
>>> set_config_value(config, "database.port", 5432)
>>> print(config)
{"database": {"host": "localhost", "port": 5432}}
```

### `merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]`
Merges two configuration dictionaries.

**Parameters:**
- `base_config`: Base configuration
- `override_config`: Override configuration

**Returns:**
- Merged configuration

**Example:**
```python
>>> base = {"api": {"timeout": 30}, "debug": False}
>>> override = {"api": {"timeout": 60}, "debug": True}
>>> merged = merge_configs(base, override)
>>> print(merged)
{"api": {"timeout": 60}, "debug": True}
```

## Usage

```python
from config_utils import (
    load_config, save_config, get_config_value, set_config_value, merge_configs
)

# Load configuration
config = load_config("app_config.json")

# Get nested configuration values
api_key = get_config_value(config, "api.key")
timeout = get_config_value(config, "api.timeout", 30)
db_host = get_config_value(config, "database.host", "localhost")

# Set configuration values
set_config_value(config, "api.key", "new_api_key")
set_config_value(config, "features.new_feature", True)

# Merge configurations (e.g., default + user config)
default_config = {
    "api": {"timeout": 30, "retries": 3},
    "debug": False,
    "logging": {"level": "INFO"}
}

user_config = {
    "api": {"timeout": 60},
    "debug": True,
    "logging": {"file": "app.log"}
}

final_config = merge_configs(default_config, user_config)

# Save updated configuration
save_config(final_config, "final_config.json")
```

## Configuration File Structure

Typical configuration file structure:

```json
{
  "api": {
    "key": "your_api_key_here",
    "timeout": 30,
    "retries": 3,
    "base_url": "https://api.example.com"
  },
  "database": {
    "host": "localhost",
    "port": 5432,
    "name": "trading_db",
    "username": "trader",
    "password": "secure_password"
  },
  "trading": {
    "default_currency": "USD",
    "risk_tolerance": "medium",
    "max_position_size": 10000,
    "stop_loss_percent": 2.0
  },
  "logging": {
    "level": "INFO",
    "file": "trading.log",
    "max_size": "10MB",
    "backup_count": 5
  },
  "features": {
    "paper_trading": true,
    "real_time_data": false,
    "advanced_charts": true
  }
}
```

## Installation

No additional dependencies required. Uses only Python standard library.

## Testing

Run the module directly to see demonstrations:

```bash
python config_utils.py
```

## Common Use Cases

- **Application Configuration**: Manage settings for trading applications
- **Environment Management**: Handle different configurations for dev/staging/production
- **User Preferences**: Store and retrieve user-specific settings
- **API Configuration**: Manage API keys, endpoints, and timeouts
- **Database Settings**: Configure database connections and parameters
- **Feature Flags**: Enable/disable features dynamically
- **Security Settings**: Manage authentication and authorization parameters

## Best Practices

- Use environment variables for sensitive data (API keys, passwords)
- Separate configuration by environment (dev, staging, production)
- Use meaningful default values for optional settings
- Validate configuration values on application startup
- Document all configuration options and their purposes
- Use version control for configuration files (excluding sensitive data)
- Implement configuration hot-reloading for production applications

## Security Considerations

- Never commit API keys or passwords to version control
- Use environment variables or secure vaults for sensitive data
- Encrypt configuration files containing sensitive information
- Implement proper access controls for configuration files
- Use different configurations for different environments
- Regularly rotate API keys and secrets
- Audit configuration changes for security compliance
