#!/usr/bin/env python3
"""Dynamic analysis tool for profiling assertion coverage.

This tool wraps OptimizationResult and tracks which attributes are accessed
during test assertions, helping identify tests that don't verify behavior-specific fields.

Usage:
    # In conftest.py:
    from tests.optimizer_validation.tools.assertion_coverage_profiler import (
        AssertionCoverageProfiler,
        wrap_result_for_profiling,
    )

    @pytest.fixture
    def profiled_scenario_runner(scenario_runner, request):
        async def runner(scenario):
            func, result = await scenario_runner(scenario)
            if not isinstance(result, Exception):
                result = wrap_result_for_profiling(result, request.node.name)
            return func, result
        return runner

    # After test session:
    profiler.generate_coverage_report()
"""

from __future__ import annotations

import json
import weakref
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Global profiler instance
_profiler: "AssertionCoverageProfiler | None" = None


def get_profiler() -> "AssertionCoverageProfiler":
    """Get or create the global profiler instance."""
    global _profiler
    if _profiler is None:
        _profiler = AssertionCoverageProfiler()
    return _profiler


@dataclass
class TestAttributeAccess:
    """Track attribute accesses for a single test."""

    test_name: str
    accessed_attributes: set[str] = field(default_factory=set)
    accessed_in_assertions: set[str] = field(default_factory=set)
    # Track call context
    access_stack: list[tuple[str, str]] = field(default_factory=list)


class ResultProxy:
    """Proxy wrapper that tracks attribute access on OptimizationResult."""

    # Behavior-critical attributes that should be checked
    CRITICAL_ATTRIBUTES = {
        "trials",
        "successful_trials",
        "best_config",
        "best_score",
        "stop_reason",
        "optimization_id",
        "status",
    }

    # Trial-level critical attributes
    TRIAL_ATTRIBUTES = {
        "config",
        "metrics",
        "status",
        "trial_id",
        "score",
    }

    def __init__(self, result: Any, test_name: str) -> None:
        object.__setattr__(self, "_result", result)
        object.__setattr__(self, "_test_name", test_name)
        object.__setattr__(self, "_accesses", TestAttributeAccess(test_name=test_name))
        get_profiler().register_test(test_name, self)

    def __getattr__(self, name: str) -> Any:
        accesses = object.__getattribute__(self, "_accesses")
        result = object.__getattribute__(self, "_result")

        accesses.accessed_attributes.add(name)

        value = getattr(result, name)

        # Wrap trials list to track trial attribute access
        if name == "trials" and isinstance(value, list):
            return TrialListProxy(value, accesses)

        return value

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            result = object.__getattribute__(self, "_result")
            setattr(result, name, value)

    def __repr__(self) -> str:
        result = object.__getattribute__(self, "_result")
        return repr(result)

    @property
    def _access_info(self) -> TestAttributeAccess:
        return object.__getattribute__(self, "_accesses")


class TrialListProxy(list):
    """Proxy for trials list that tracks element access."""

    def __init__(self, trials: list, accesses: TestAttributeAccess) -> None:
        super().__init__(trials)
        self._accesses = accesses
        self._wrapped_trials = [TrialProxy(t, accesses) for t in trials]

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return [self._wrapped_trials[i] for i in range(*index.indices(len(self)))]
        return self._wrapped_trials[index]

    def __iter__(self):
        return iter(self._wrapped_trials)


class TrialProxy:
    """Proxy wrapper for individual trial objects."""

    def __init__(self, trial: Any, accesses: TestAttributeAccess) -> None:
        object.__setattr__(self, "_trial", trial)
        object.__setattr__(self, "_accesses", accesses)

    def __getattr__(self, name: str) -> Any:
        accesses = object.__getattribute__(self, "_accesses")
        trial = object.__getattribute__(self, "_trial")

        accesses.accessed_attributes.add(f"trial.{name}")

        return getattr(trial, name)

    def __repr__(self) -> str:
        trial = object.__getattribute__(self, "_trial")
        return repr(trial)


class AssertionCoverageProfiler:
    """Profiler that tracks assertion coverage across tests."""

    def __init__(self) -> None:
        self.test_accesses: dict[str, TestAttributeAccess] = {}
        self._proxies: dict[str, weakref.ref] = {}

    def register_test(self, test_name: str, proxy: ResultProxy) -> None:
        """Register a test and its result proxy."""
        self._proxies[test_name] = weakref.ref(proxy)

    def finalize_test(self, test_name: str) -> None:
        """Finalize tracking for a test."""
        proxy_ref = self._proxies.get(test_name)
        if proxy_ref:
            proxy = proxy_ref()
            if proxy:
                self.test_accesses[test_name] = proxy._access_info

    def get_uncovered_tests(self) -> list[dict]:
        """Get tests that don't access critical attributes."""
        uncovered = []

        for test_name, accesses in self.test_accesses.items():
            missing_critical = (
                ResultProxy.CRITICAL_ATTRIBUTES - accesses.accessed_attributes
            )
            missing_trial = set()

            # Check if trials were accessed but trial attributes weren't
            if "trials" in accesses.accessed_attributes:
                trial_accesses = {
                    a for a in accesses.accessed_attributes if a.startswith("trial.")
                }
                trial_attrs = {a.split(".")[1] for a in trial_accesses}
                missing_trial = ResultProxy.TRIAL_ATTRIBUTES - trial_attrs

            if missing_critical or missing_trial:
                uncovered.append(
                    {
                        "test_name": test_name,
                        "accessed": list(accesses.accessed_attributes),
                        "missing_critical": list(missing_critical),
                        "missing_trial": list(missing_trial),
                        "coverage_score": self._compute_coverage_score(accesses),
                    }
                )

        return sorted(uncovered, key=lambda x: x["coverage_score"])

    def _compute_coverage_score(self, accesses: TestAttributeAccess) -> float:
        """Compute coverage score (0-1) based on critical attribute access."""
        critical_accessed = (
            ResultProxy.CRITICAL_ATTRIBUTES & accesses.accessed_attributes
        )
        return len(critical_accessed) / len(ResultProxy.CRITICAL_ATTRIBUTES)

    def generate_report(self, output_path: Path | None = None) -> dict:
        """Generate coverage report."""
        uncovered = self.get_uncovered_tests()

        report = {
            "generated_at": datetime.now().isoformat(),
            "total_tests": len(self.test_accesses),
            "tests_with_coverage_gaps": len(uncovered),
            "average_coverage": (
                sum(t["coverage_score"] for t in uncovered) / len(uncovered)
                if uncovered
                else 1.0
            ),
            "uncovered_tests": uncovered,
            "critical_attributes": list(ResultProxy.CRITICAL_ATTRIBUTES),
            "trial_attributes": list(ResultProxy.TRIAL_ATTRIBUTES),
        }

        if output_path:
            output_path.write_text(json.dumps(report, indent=2))

        return report


def wrap_result_for_profiling(result: Any, test_name: str) -> ResultProxy:
    """Wrap an OptimizationResult for profiling."""
    return ResultProxy(result, test_name)


# Pytest plugin hooks
def pytest_runtest_teardown(item) -> None:
    """Finalize test tracking after each test."""
    profiler = get_profiler()
    profiler.finalize_test(item.name)


def pytest_sessionfinish(session, exitstatus) -> None:
    """Generate report at end of session."""
    profiler = get_profiler()
    if profiler.test_accesses:
        report = profiler.generate_report()
        report_path = Path("tests/optimizer_validation/coverage_report.json")
        report_path.write_text(json.dumps(report, indent=2))
        print(f"\nAssertion coverage report written to {report_path}")


if __name__ == "__main__":
    # Demo usage
    print("Assertion Coverage Profiler")
    print("=" * 40)
    print("This tool is designed to be used as a pytest plugin.")
    print()
    print("To enable, add to conftest.py:")
    print(
        "  pytest_plugins = ['tests.optimizer_validation.tools.assertion_coverage_profiler']"
    )
