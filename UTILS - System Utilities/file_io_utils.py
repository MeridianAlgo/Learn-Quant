"""
File I/O Utilities for Financial Applications

This module provides comprehensive file input/output utilities for financial applications,
including JSON and CSV operations, directory management, file backup functionality,
and file size formatting.

Author: Generated for Learn-Quant Project
Version: 1.0.0
"""

import os
import json
import csv
from typing import List, Dict, Any


def ensure_directory_exists(file_path: str) -> None:
    """
    Ensure that the directory for the given file path exists.

    Args:
        file_path: File path that requires directory existence

    Example:
        >>> ensure_directory_exists("data/trades/2024/01/trades.json")
        # Creates directory structure if it doesn't exist
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def read_json_file(file_path: str) -> Dict[str, Any]:
    """
    Read and parse JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Parsed JSON data as dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON

    Example:
        >>> data = read_json_file("config.json")
        >>> print(data["api_key"])
        "your_api_key_here"
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json_file(data: Dict[str, Any], file_path: str, indent: int = 2) -> None:
    """
    Write data to JSON file.

    Args:
        data: Data to write
        file_path: Output file path
        indent: JSON indentation level

    Example:
        >>> portfolio_data = {"AAPL": 100, "GOOGL": 50}
        >>> write_json_file(portfolio_data, "portfolio.json")
    """
    ensure_directory_exists(file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def read_csv_file(file_path: str, delimiter: str = ",") -> List[Dict[str, str]]:
    """
    Read CSV file and return as list of dictionaries.

    Args:
        file_path: Path to CSV file
        delimiter: CSV delimiter character

    Returns:
        List of rows as dictionaries

    Example:
        >>> trades = read_csv_file("trades.csv")
        >>> print(trades[0]["symbol"])
        "AAPL"
    """
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        return list(reader)


def write_csv_file(
    data: List[Dict[str, str]], file_path: str, delimiter: str = ","
) -> None:
    """
    Write data to CSV file.

    Args:
        data: List of dictionaries to write
        file_path: Output file path
        delimiter: CSV delimiter character

    Example:
        >>> trades = [{"symbol": "AAPL", "shares": 100, "price": 150.25}]
        >>> write_csv_file(trades, "trades.csv")
    """
    if not data:
        return

    ensure_directory_exists(file_path)
    with open(file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys(), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(data)


def backup_file(file_path: str, backup_suffix: str = ".bak") -> str:
    """
    Create a backup of the specified file.

    Args:
        file_path: Path to file to backup
        backup_suffix: Suffix for backup file

    Returns:
        Path to backup file

    Example:
        >>> backup_path = backup_file("important_data.json")
        >>> print(backup_path)
        "important_data.json.bak"
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    backup_path = file_path + backup_suffix
    counter = 1

    # If backup exists, add number suffix
    while os.path.exists(backup_path):
        backup_path = f"{file_path}{backup_suffix}.{counter}"
        counter += 1

    import shutil

    shutil.copy2(file_path, backup_path)
    return backup_path


def get_file_size(file_path: str) -> int:
    """
    Get file size in bytes.

    Args:
        file_path: Path to file

    Returns:
        File size in bytes

    Example:
        >>> size = get_file_size("large_file.csv")
        >>> print(size)
        1048576
    """
    return os.path.getsize(file_path)


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Human-readable size string

    Example:
        >>> format_file_size(1048576)
        "1.0 MB"
        >>> format_file_size(1536)
        "1.5 KB"
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def demo_file_io():
    """Demonstrate file I/O utilities."""
    print("=" * 60)
    print("FILE I/O UTILITIES DEMONSTRATION")
    print("=" * 60)

    # Create test data
    test_config = {
        "api_key": "test_key_123",
        "trading_account": "ACC123456",
        "preferences": {"risk_tolerance": "medium", "default_currency": "USD"},
    }

    test_trades = [
        {"symbol": "AAPL", "shares": "100", "price": "150.25", "date": "2024-01-15"},
        {"symbol": "GOOGL", "shares": "50", "price": "2800.50", "date": "2024-01-15"},
        {"symbol": "TSLA", "shares": "25", "price": "850.75", "date": "2024-01-16"},
    ]

    # JSON operations
    print("\nJSON Operations:")
    config_file = "demo_config.json"
    write_json_file(test_config, config_file)
    print(f"  ✓ Wrote JSON config to {config_file}")

    loaded_config = read_json_file(config_file)
    print(f"  ✓ Loaded JSON config: {loaded_config['api_key']}")

    # CSV operations
    print("\nCSV Operations:")
    trades_file = "demo_trades.csv"
    write_csv_file(test_trades, trades_file)
    print(f"  ✓ Wrote CSV trades to {trades_file}")

    loaded_trades = read_csv_file(trades_file)
    print(f"  ✓ Loaded CSV trades: {len(loaded_trades)} trades")

    # File operations
    print("\nFile Operations:")
    size = get_file_size(config_file)
    formatted_size = format_file_size(size)
    print(f"  {config_file} size: {formatted_size}")

    # Backup operations
    backup_path = backup_file(config_file)
    print(f"  ✓ Created backup: {backup_path}")

    # Directory operations
    nested_file = "data/2024/01/trades.json"
    ensure_directory_exists(nested_file)
    print(f"  ✓ Ensured directory exists for: {nested_file}")

    # Cleanup demo files
    import os

    demo_files = [config_file, trades_file, backup_path]
    for file in demo_files:
        if os.path.exists(file):
            os.remove(file)

    # Clean up directories
    import shutil

    for dir_path in ["data/2024/01", "data/2024", "data"]:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
            except:
                pass

    print("\n  ✓ Cleaned up demo files")


def main():
    """Main function to run demonstrations."""
    demo_file_io()
    print("\n🎉 File I/O utilities demonstration complete!")


if __name__ == "__main__":
    main()
