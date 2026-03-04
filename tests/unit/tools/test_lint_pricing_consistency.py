"""Unit tests for pricing consistency lint rules."""

from __future__ import annotations

import ast
import textwrap

from tests.optimizer_validation.tools.lint_pricing_consistency import (
    PricingLinter,
    Severity,
    _is_allowlisted,
)


def _lint(source_code: str, file_path: str = "traigent/sample_module.py"):
    """Run PricingLinter on a snippet and return issues."""
    source = textwrap.dedent(source_code)
    tree = ast.parse(source)
    linter = PricingLinter(file_path, source)
    linter.visit(tree)
    return linter.issues


class TestPricingConsistencyLinter:
    """Tests for P001/P002/P003 rules and allowlist behavior."""

    def test_p001_detects_pricing_keys(self) -> None:
        issues = _lint(
            """\
        rates = {"prompt": 0.005, "completion": 0.012}
        """
        )
        assert any(i.code == "P001" and i.severity == Severity.ERROR for i in issues)

    def test_p002_detects_model_pricing_table_with_tuples(self) -> None:
        issues = _lint(
            """\
        cost_per_1m = {
            "gpt-4o": (2.5, 10.0),
            "gpt-4o-mini": (0.15, 0.6),
        }
        """
        )
        assert any(i.code == "P002" and i.severity == Severity.ERROR for i in issues)

    def test_p003_detects_tuple_assignment(self) -> None:
        issues = _lint(
            """\
        cost_per_1m = (2.5, 10.0)
        """
        )
        assert any(i.code == "P003" and i.severity == Severity.WARNING for i in issues)

    def test_numeric_threshold_does_not_flag_one_of_three_numeric(self) -> None:
        issues = _lint(
            """\
        model_table = {
            "gpt-4o": {"input": 0.0025, "meta": "x", "note": "y"},
            "gpt-4o-mini": {"input": 0.00015, "meta": "x", "note": "y"},
        }
        """
        )
        assert not any(i.code == "P002" for i in issues)

    def test_numeric_threshold_flags_two_of_three_numeric(self) -> None:
        issues = _lint(
            """\
        model_table = {
            "gpt-4o": {"input": 0.0025, "output": 0.01, "meta": "x"},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006, "meta": "x"},
        }
        """
        )
        assert any(i.code == "P002" for i in issues)

    def test_allowlist_includes_experimental_paths_by_policy(self) -> None:
        assert _is_allowlisted("traigent/experimental/simple_cloud/platforms/foo.py")
