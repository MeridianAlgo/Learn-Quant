"""
Configuration Utilities for Financial Applications

This module provides comprehensive configuration management utilities for financial applications,
including JSON configuration loading/saving, dot notation access, configuration merging,
and hierarchical configuration management.

Author: Generated for Learn-Quant Project
Version: 1.0.0
"""

import json
import os
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Example:
        >>> config = load_config("config.json")
        >>> print(config["api_key"])
        "your_api_key_here"
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}")


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
        
    Example:
        >>> config = {"api_key": "key123", "timeout": 30}
        >>> save_config(config, "config.json")
    """
    # Ensure directory exists
    directory = os.path.dirname(config_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., 'database.host')
        default: Default value if key not found
        
    Returns:
        Configuration value
        
    Example:
        >>> config = {"database": {"host": "localhost", "port": 5432}}
        >>> get_config_value(config, "database.host")
        "localhost"
        >>> get_config_value(config, "database.password", "default")
        "default"
    """
    keys = key_path.split('.')
    current = config
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    
    return current


def set_config_value(config: Dict[str, Any], key_path: str, value: Any) -> None:
    """
    Set configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., 'database.host')
        value: Value to set
        
    Example:
        >>> config = {}
        >>> set_config_value(config, "database.host", "localhost")
        >>> set_config_value(config, "database.port", 5432)
        >>> print(config)
        {"database": {"host": "localhost", "port": 5432}}
    """
    keys = key_path.split('.')
    current = config
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
        
    Example:
        >>> base = {"api": {"timeout": 30}, "debug": False}
        >>> override = {"api": {"timeout": 60}, "debug": True}
        >>> merged = merge_configs(base, override)
        >>> print(merged)
        {"api": {"timeout": 60}, "debug": True}
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def validate_config(config: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """
    Validate configuration against a schema.
    
    Args:
        config: Configuration dictionary to validate
        schema: Schema dictionary defining required structure
        
    Returns:
        True if configuration is valid, False otherwise
        
    Example:
        >>> schema = {"api": {"key": str, "timeout": int}}
        >>> config = {"api": {"key": "test", "timeout": 30}}
        >>> validate_config(config, schema)
        True
    """
    def _validate_recursive(config_part: Dict[str, Any], schema_part: Dict[str, Any]) -> bool:
        for key, expected_type in schema_part.items():
            if key not in config_part:
                return False
            
            if isinstance(expected_type, dict):
                if not isinstance(config_part[key], dict):
                    return False
                if not _validate_recursive(config_part[key], expected_type):
                    return False
            elif not isinstance(config_part[key], expected_type):
                return False
        
        return True
    
    return _validate_recursive(config, schema)


def get_environment_config() -> Dict[str, Any]:
    """
    Get configuration from environment variables.
    
    Returns:
        Dictionary with environment variables
        
    Example:
        >>> # Set environment variables: API_KEY=test123, DEBUG=true
        >>> env_config = get_environment_config()
        >>> print(env_config["API_KEY"])
        "test123"
    """
    env_config = {}
    
    # Common configuration environment variables
    config_mappings = {
        'API_KEY': 'api.key',
        'API_TIMEOUT': 'api.timeout',
        'API_RETRIES': 'api.retries',
        'DB_HOST': 'database.host',
        'DB_PORT': 'database.port',
        'DB_NAME': 'database.name',
        'DB_USER': 'database.username',
        'DB_PASSWORD': 'database.password',
        'DEBUG': 'debug',
        'LOG_LEVEL': 'logging.level',
        'LOG_FILE': 'logging.file'
    }
    
    for env_var, config_path in config_mappings.items():
        value = os.getenv(env_var)
        if value is not None:
            # Try to convert to appropriate type
            if value.lower() in ('true', 'false'):
                value = value.lower() == 'true'
            elif value.isdigit():
                value = int(value)
            elif value.replace('.', '').isdigit():
                value = float(value)
            
            set_config_value(env_config, config_path, value)
    
    return env_config


def demo_config_utils():
    """Demonstrate configuration utilities."""
    print("=" * 60)
    print("CONFIGURATION UTILITIES DEMONSTRATION")
    print("=" * 60)
    
    # Create sample configuration
    print("\n1. Creating Sample Configuration:")
    sample_config = {
        "api": {
            "key": "sample_api_key_123",
            "timeout": 30,
            "retries": 3,
            "base_url": "https://api.example.com"
        },
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "trading_db"
        },
        "trading": {
            "default_currency": "USD",
            "risk_tolerance": "medium",
            "max_position_size": 10000
        },
        "debug": False,
        "features": {
            "paper_trading": True,
            "real_time_data": False
        }
    }
    
    print("  Sample configuration created")
    
    # Save and load configuration
    print("\n2. Save and Load Configuration:")
    config_file = "demo_config.json"
    save_config(sample_config, config_file)
    print(f"  ✓ Configuration saved to {config_file}")
    
    loaded_config = load_config(config_file)
    print(f"  ✓ Configuration loaded successfully")
    
    # Get configuration values
    print("\n3. Get Configuration Values:")
    api_key = get_config_value(loaded_config, "api.key")
    timeout = get_config_value(loaded_config, "api.timeout", 60)
    missing = get_config_value(loaded_config, "missing.key", "default_value")
    
    print(f"  API Key: {api_key}")
    print(f"  Timeout: {timeout}")
    print(f"  Missing key (with default): {missing}")
    
    # Set configuration values
    print("\n4. Set Configuration Values:")
    set_config_value(loaded_config, "api.new_setting", "new_value")
    set_config_value(loaded_config, "new_section.new_key", "new_section_value")
    print(f"  ✓ Added new settings to configuration")
    
    # Merge configurations
    print("\n5. Merge Configurations:")
    override_config = {
        "api": {"timeout": 60, "retries": 5},
        "debug": True,
        "new_feature": True
    }
    
    merged_config = merge_configs(loaded_config, override_config)
    print(f"  Original timeout: {loaded_config['api']['timeout']}")
    print(f"  Merged timeout: {merged_config['api']['timeout']}")
    print(f"  Debug mode: {merged_config['debug']}")
    print(f"  New feature: {merged_config['new_feature']}")
    
    # Configuration validation
    print("\n6. Configuration Validation:")
    schema = {
        "api": {
            "key": str,
            "timeout": int
        },
        "debug": bool
    }
    
    is_valid = validate_config(merged_config, schema)
    print(f"  Configuration is valid: {is_valid}")
    
    # Environment configuration
    print("\n7. Environment Configuration:")
    print("  Set environment variables to test:")
    print("    export API_KEY=env_key_123")
    print("    export DEBUG=true")
    print("    export API_TIMEOUT=45")
    
    env_config = get_environment_config()
    if env_config:
        print("  Environment configuration found:")
        for key_path, value in env_config.items():
            print(f"    {key_path}: {value}")
    else:
        print("  No environment configuration variables found")
    
    # Cleanup
    if os.path.exists(config_file):
        os.remove(config_file)
        print(f"\n  ✓ Cleaned up demo file: {config_file}")


def main():
    """Main function to run demonstrations."""
    demo_config_utils()
    print("\n🎉 Configuration utilities demonstration complete!")


if __name__ == "__main__":
    main()
