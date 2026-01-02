"""Validation and utility tools for optimizer validation tests.

This module provides tools for:
1. Evidence validation (validate_evidence.py)
2. Test weakness detection (test_weakness_analyzer.py)
3. Assertion coverage profiling (assertion_coverage_profiler.py)
4. Mutation-based oracle testing (mutation_oracle.py)

Usage:
    # Static analysis for weak tests
    python -m tests.optimizer_validation.tools.test_weakness_analyzer --output json

    # Mutation testing
    python -m tests.optimizer_validation.tools.mutation_oracle
"""

from .validate_evidence import (
    ValidationIssue,
    ValidationReport,
    validate_all_tests,
    validate_evidence,
)

__all__ = [
    "ValidationIssue",
    "ValidationReport",
    "validate_all_tests",
    "validate_evidence",
]
