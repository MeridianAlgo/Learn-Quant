"""
Latency Optimization Tools

Tools for measuring and optimizing trading system latency,
including network latency, processing time, and execution delays.
"""

import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class LatencyMeasurement:
    """Represents a latency measurement."""

    timestamp: datetime
    operation: str
    latency_ms: float
    metadata: Dict[str, any]


class LatencyOptimizer:
    """
    Comprehensive latency optimization and monitoring tool.
    """

    def __init__(self):
        self.measurements: List[LatencyMeasurement] = []
        self.operation_stats: Dict[str, List[float]] = {}
        self.baseline_measurements: Dict[str, float] = {}

    def measure_latency(self, operation: str, func: Callable, *args, **kwargs) -> Tuple[any, float]:
        """
        Measure latency of a function execution.

        Args:
            operation: Name of the operation
            func: Function to measure
            *args, **kwargs: Function arguments

        Returns:
            Tuple of (result, latency_ms)
        """
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000

        # Store measurement
        measurement = LatencyMeasurement(
            timestamp=datetime.now(),
            operation=operation,
            latency_ms=latency_ms,
            metadata={},
        )
        self.measurements.append(measurement)

        # Update operation stats
        if operation not in self.operation_stats:
            self.operation_stats[operation] = []
        self.operation_stats[operation].append(latency_ms)

        return result, latency_ms

    def measure_network_latency(self, host: str, port: int, timeout: float = 1.0) -> float:
        """
        Measure network latency to a remote host.

        Args:
            host: Target host
            port: Target port
            timeout: Timeout in seconds

        Returns:
            Latency in milliseconds
        """
        import socket

        start_time = time.perf_counter()
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((host, port))
            sock.close()
            end_time = time.perf_counter()
            return (end_time - start_time) * 1000
        except Exception:
            return float("inf")

    def benchmark_operation(
        self, operation: str, func: Callable, iterations: int = 100, *args, **kwargs
    ) -> Dict[str, float]:
        """
        Benchmark an operation over multiple iterations.

        Args:
            operation: Operation name
            func: Function to benchmark
            iterations: Number of iterations
            *args, **kwargs: Function arguments

        Returns:
            Dictionary with benchmark statistics
        """
        latencies = []

        for _ in range(iterations):
            _, latency = self.measure_latency(operation, func, *args, **kwargs)
            latencies.append(latency)

        stats = {
            "mean_ms": statistics.mean(latencies),
            "median_ms": statistics.median(latencies),
            "std_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "p95_ms": np.percentile(latencies, 95),
            "p99_ms": np.percentile(latencies, 99),
        }

        return stats

    def set_baseline(self, operation: str, baseline_latency: float) -> None:
        """Set baseline latency for an operation."""
        self.baseline_measurements[operation] = baseline_latency

    def get_latency_regression(self, operation: str) -> Optional[float]:
        """
        Calculate latency regression relative to baseline.

        Args:
            operation: Operation name

        Returns:
            Regression percentage or None if no baseline
        """
        if operation not in self.baseline_measurements:
            return None

        if operation not in self.operation_stats:
            return None

        baseline = self.baseline_measurements[operation]
        current = statistics.mean(self.operation_stats[operation])

        return ((current - baseline) / baseline) * 100

    def get_slow_operations(self, threshold_ms: float = 10.0) -> List[str]:
        """
        Get operations with average latency above threshold.

        Args:
            threshold_ms: Latency threshold in milliseconds

        Returns:
            List of slow operations
        """
        slow_ops = []
        for operation, latencies in self.operation_stats.items():
            avg_latency = statistics.mean(latencies)
            if avg_latency > threshold_ms:
                slow_ops.append(operation)

        return slow_ops

    def optimize_code_path(self, operation: str, optimization_func: Callable) -> Dict[str, float]:
        """
        Test and measure optimization improvements.

        Args:
            operation: Operation name
            optimization_func: Optimized version of the function

        Returns:
            Dictionary with before/after statistics
        """
        if operation not in self.operation_stats:
            return {}

        original_latencies = self.operation_stats[operation]
        original_avg = statistics.mean(original_latencies)

        # Benchmark optimized version
        optimized_stats = self.benchmark_operation(f"{operation}_optimized", optimization_func, iterations=50)

        improvement = ((original_avg - optimized_stats["mean_ms"]) / original_avg) * 100

        return {
            "original_avg_ms": original_avg,
            "optimized_avg_ms": optimized_stats["mean_ms"],
            "improvement_percent": improvement,
            "speedup_factor": original_avg / optimized_stats["mean_ms"],
        }

    def analyze_latency_patterns(self, window_minutes: int = 60) -> Dict[str, any]:
        """
        Analyze latency patterns over time.

        Args:
            window_minutes: Time window to analyze

        Returns:
            Dictionary with pattern analysis
        """
        if not self.measurements:
            return {}

        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_measurements = [m for m in self.measurements if m.timestamp >= cutoff_time]

        if not recent_measurements:
            return {}

        # Group by operation
        operation_patterns = {}
        for operation in {m.operation for m in recent_measurements}:
            op_measurements = [m for m in recent_measurements if m.operation == operation]
            latencies = [m.latency_ms for m in op_measurements]

            operation_patterns[operation] = {
                "count": len(latencies),
                "avg_ms": statistics.mean(latencies),
                "trend": self._calculate_trend(latencies),
                "volatility": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            }

        return operation_patterns

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values."""
        if len(values) < 2:
            return "stable"

        # Simple linear regression to determine trend
        x = np.arange(len(values))
        y = np.array(values)

        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]

        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"

    def generate_latency_report(self) -> str:
        """Generate comprehensive latency report."""
        if not self.operation_stats:
            return "No latency measurements available."

        report = "Latency Analysis Report\n"
        report += "=" * 50 + "\n\n"

        # Overall statistics
        all_latencies = []
        for latencies in self.operation_stats.values():
            all_latencies.extend(latencies)

        if all_latencies:
            report += "Overall Statistics:\n"
            report += f"  Total Measurements: {len(all_latencies)}\n"
            report += f"  Average Latency: {statistics.mean(all_latencies):.2f} ms\n"
            report += f"  Median Latency: {statistics.median(all_latencies):.2f} ms\n"
            report += f"  P95 Latency: {np.percentile(all_latencies, 95):.2f} ms\n"
            report += f"  P99 Latency: {np.percentile(all_latencies, 99):.2f} ms\n\n"

        # Per-operation statistics
        report += "Per-Operation Statistics:\n"
        for operation, latencies in self.operation_stats.items():
            avg_latency = statistics.mean(latencies)
            regression = self.get_latency_regression(operation)

            report += f"\n{operation}:\n"
            report += f"  Count: {len(latencies)}\n"
            report += f"  Average: {avg_latency:.2f} ms\n"
            report += f"  Min: {min(latencies):.2f} ms\n"
            report += f"  Max: {max(latencies):.2f} ms\n"

            if regression is not None:
                status = "REGRESSION" if regression > 10 else "OK"
                report += f"  Regression: {regression:.1f}% [{status}]\n"

        # Slow operations
        slow_ops = self.get_slow_operations()
        if slow_ops:
            report += f"\nSlow Operations (>10ms): {', '.join(slow_ops)}\n"

        return report

    def export_measurements(self, filename: str) -> None:
        """Export measurements to CSV file."""
        if not self.measurements:
            return

        data = []
        for measurement in self.measurements:
            data.append(
                {
                    "timestamp": measurement.timestamp,
                    "operation": measurement.operation,
                    "latency_ms": measurement.latency_ms,
                }
            )

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)

    def clear_measurements(self, operation: Optional[str] = None) -> None:
        """Clear measurements for specific operation or all operations."""
        if operation:
            self.measurements = [m for m in self.measurements if m.operation != operation]
            if operation in self.operation_stats:
                del self.operation_stats[operation]
        else:
            self.measurements.clear()
            self.operation_stats.clear()


def main():
    """Example usage of LatencyOptimizer."""
    optimizer = LatencyOptimizer()

    # Example functions to benchmark
    def fast_function():
        return sum(range(1000))

    def slow_function():
        return sum(range(100000))

    def network_simulation():
        time.sleep(0.01)  # Simulate network delay
        return "response"

    print("Latency Optimization Demo")
    print("=" * 40)

    # Measure individual operations
    result1, latency1 = optimizer.measure_latency("fast_operation", fast_function)
    print(f"Fast operation: {latency1:.4f} ms")

    result2, latency2 = optimizer.measure_latency("slow_operation", slow_function)
    print(f"Slow operation: {latency2:.4f} ms")

    # Benchmark operations
    print("\nBenchmarking fast operation...")
    fast_stats = optimizer.benchmark_operation("fast_operation", fast_function, iterations=100)
    for stat, value in fast_stats.items():
        print(f"  {stat}: {value:.4f}")

    print("\nBenchmarking slow operation...")
    slow_stats = optimizer.benchmark_operation("slow_operation", slow_function, iterations=50)
    for stat, value in slow_stats.items():
        print(f"  {stat}: {value:.4f}")

    # Set baselines
    optimizer.set_baseline("fast_operation", fast_stats["mean_ms"])
    optimizer.set_baseline("slow_operation", slow_stats["mean_ms"])

    # Measure some more to check for regression
    for _ in range(10):
        optimizer.measure_latency("fast_operation", fast_function)
        optimizer.measure_latency("slow_operation", slow_function)

    # Check regressions
    print("\nLatency Regressions:")
    for op in ["fast_operation", "slow_operation"]:
        regression = optimizer.get_latency_regression(op)
        if regression:
            print(f"  {op}: {regression:.1f}%")

    # Get slow operations
    slow_ops = optimizer.get_slow_operations(threshold_ms=1.0)
    print(f"\nSlow operations: {slow_ops}")

    # Generate report
    print(f"\n{optimizer.generate_latency_report()}")


if __name__ == "__main__":
    main()
