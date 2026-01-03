#!/usr/bin/env python3
"""
Performance monitoring utility for Traigent SDK.

This utility provides performance profiling, bottleneck detection,
and performance regression monitoring for the codebase.
"""

import cProfile
import functools
import json
import logging
import pstats
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List

import psutil

from traigent.utils.secure_path import safe_open, validate_path
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor and profile performance of Traigent functions."""

    def __init__(self, output_dir: str = "performance_reports"):
        """Initialize performance monitor."""
        self._base_dir = Path.cwd()
        self.output_dir = validate_path(output_dir, self._base_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metrics = {}
        self.profiles = {}

    def profile_function(self, func_name: str = None):
        """Decorator to profile function performance."""

        def decorator(func: Callable) -> Callable:
            name = func_name or f"{func.__module__}.{func.__name__}"

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                profiler = cProfile.Profile()
                start_time = time.perf_counter()
                start_memory = self._get_memory_usage()

                try:
                    profiler.enable()
                    result = func(*args, **kwargs)
                    profiler.disable()

                    end_time = time.perf_counter()
                    end_memory = self._get_memory_usage()

                    # Store metrics
                    execution_time = end_time - start_time
                    memory_delta = end_memory - start_memory

                    self.metrics[name] = {
                        "execution_time": execution_time,
                        "memory_usage": memory_delta,
                        "timestamp": time.time(),
                        "args_count": len(args),
                        "kwargs_count": len(kwargs),
                    }

                    # Store profile if execution time > 1 second
                    if execution_time > 1.0:
                        self.profiles[name] = profiler
                        self._save_profile(name, profiler)

                    # Log slow functions
                    if execution_time > 5.0:
                        logger.warning(
                            f"Slow function detected: {name} took {execution_time:.2f}s"
                        )

                    return result

                except Exception as e:
                    logger.error(f"Error profiling {name}: {e}")
                    profiler.disable()
                    raise

            return wrapper

        return decorator

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except:
            return 0.0

    def _save_profile(self, func_name: str, profiler: cProfile.Profile) -> None:
        """Save profile data to file."""
        stats = pstats.Stats(profiler)
        stats.sort_stats("cumulative")

        # Save detailed stats
        profile_file = validate_path(
            self.output_dir / f"{func_name.replace('.', '_')}_profile.txt",
            self.output_dir,
        )
        with safe_open(profile_file, self.output_dir, mode="w", encoding="utf-8") as f:
            stats.print_stats(file=f)

        # Save top bottlenecks
        bottlenecks_file = validate_path(
            self.output_dir / f"{func_name.replace('.', '_')}_bottlenecks.json",
            self.output_dir,
        )
        bottlenecks = self._extract_bottlenecks(stats)
        with safe_open(
            bottlenecks_file, self.output_dir, mode="w", encoding="utf-8"
        ) as f:
            json.dump(bottlenecks, f, indent=2)

    def _extract_bottlenecks(self, stats: pstats.Stats) -> Dict[str, Any]:
        """Extract top performance bottlenecks."""
        bottlenecks = []

        for func_info, (call_count, _, cumulative_time, _, _) in stats.stats.items():
            if cumulative_time > 0.1:  # Only functions taking >100ms
                bottlenecks.append(
                    {
                        "function": f"{func_info[0]}:{func_info[1]}({func_info[2]})",
                        "call_count": call_count,
                        "cumulative_time": cumulative_time,
                        "avg_time_per_call": (
                            cumulative_time / call_count if call_count > 0 else 0
                        ),
                    }
                )

        # Sort by cumulative time
        bottlenecks.sort(key=lambda x: x["cumulative_time"], reverse=True)
        return {"bottlenecks": bottlenecks[:20]}  # Top 20 bottlenecks

    @contextmanager
    def measure_block(self, block_name: str):
        """Context manager to measure code block performance."""
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()

        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_memory = self._get_memory_usage()

            self.metrics[f"block_{block_name}"] = {
                "execution_time": end_time - start_time,
                "memory_usage": end_memory - start_memory,
                "timestamp": time.time(),
                "type": "code_block",
            }

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.metrics:
            return {"message": "No performance data collected"}

        # Calculate statistics
        execution_times = [
            m["execution_time"] for m in self.metrics.values() if "execution_time" in m
        ]
        memory_usages = [
            m["memory_usage"] for m in self.metrics.values() if "memory_usage" in m
        ]

        report = {
            "summary": {
                "total_functions_profiled": len(self.metrics),
                "avg_execution_time": (
                    sum(execution_times) / len(execution_times)
                    if execution_times
                    else 0
                ),
                "max_execution_time": max(execution_times) if execution_times else 0,
                "total_memory_delta": sum(memory_usages) if memory_usages else 0,
                "functions_with_profiles": len(self.profiles),
            },
            "slow_functions": [
                {"name": name, **metrics}
                for name, metrics in self.metrics.items()
                if metrics.get("execution_time", 0) > 1.0
            ],
            "memory_intensive": [
                {"name": name, **metrics}
                for name, metrics in self.metrics.items()
                if metrics.get("memory_usage", 0) > 100  # >100MB
            ],
            "detailed_metrics": self.metrics,
        }

        # Sort by execution time
        report["slow_functions"].sort(key=lambda x: x["execution_time"], reverse=True)
        report["memory_intensive"].sort(key=lambda x: x["memory_usage"], reverse=True)

        return report

    def save_report(self, filename: str = None) -> str:
        """Save performance report to file."""
        report = self.generate_report()

        if not filename:
            timestamp = int(time.time())
            filename = f"performance_report_{timestamp}.json"

        report_file = validate_path(self.output_dir / filename, self.output_dir)
        with safe_open(report_file, self.output_dir, mode="w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return str(report_file)

    def get_bottlenecks(self, threshold: float = 1.0) -> List[Dict[str, Any]]:
        """Get list of performance bottlenecks."""
        bottlenecks = []

        for name, metrics in self.metrics.items():
            if metrics.get("execution_time", 0) > threshold:
                bottlenecks.append(
                    {
                        "function": name,
                        "execution_time": metrics["execution_time"],
                        "memory_usage": metrics.get("memory_usage", 0),
                        "severity": (
                            "critical"
                            if metrics["execution_time"] > 10
                            else "high" if metrics["execution_time"] > 5 else "medium"
                        ),
                    }
                )

        return sorted(bottlenecks, key=lambda x: x["execution_time"], reverse=True)


# Global performance monitor instance
_performance_monitor = PerformanceMonitor()


def profile(func_name: str = None):
    """Convenience decorator for profiling functions."""
    return _performance_monitor.profile_function(func_name)


def measure_block(block_name: str):
    """Convenience context manager for measuring code blocks."""
    return _performance_monitor.measure_block(block_name)


def get_performance_report() -> Dict[str, Any]:
    """Get current performance report."""
    return _performance_monitor.generate_report()


def save_performance_report(filename: str = None) -> str:
    """Save performance report to file."""
    return _performance_monitor.save_report(filename)


def get_bottlenecks(threshold: float = 1.0) -> List[Dict[str, Any]]:
    """Get performance bottlenecks."""
    return _performance_monitor.get_bottlenecks(threshold)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Performance monitoring utility")
    parser.add_argument(
        "--report", action="store_true", help="Generate performance report"
    )
    parser.add_argument(
        "--bottlenecks",
        type=float,
        default=1.0,
        help="Show bottlenecks with threshold (seconds)",
    )

    args = parser.parse_args()

    if args.report:
        report_file = save_performance_report()
        print(f"Performance report saved to: {report_file}")

    if args.bottlenecks:
        bottlenecks = get_bottlenecks(args.bottlenecks)
        if bottlenecks:
            print(f"\nPerformance bottlenecks (>{args.bottlenecks}s):")
            for bottleneck in bottlenecks:
                print(
                    f"  {bottleneck['severity'].upper()}: {bottleneck['function']} "
                    f"({bottleneck['execution_time']:.2f}s)"
                )
        else:
            print("No performance bottlenecks found.")
