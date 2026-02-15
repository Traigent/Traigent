"""Unit tests for the LLM test quality scanner.

Tests the three core components:
1. SmellDetector — AST-based test smell detection
2. OracleStrengthAnalyzer — oracle scoring heuristics
3. MutationGuidedAnalyzer — mutant targeting (no LLM calls)
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

from tests.optimizer_validation.tools.llm_test_scanner import (
    LLMTestScanner,
    MutationGuidedAnalyzer,
    OracleStrengthAnalyzer,
    OracleStrengthReport,
    SmellDetector,
    TestSmell,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect(code: str) -> list:
    """Run SmellDetector on a code snippet and return smells."""
    source = textwrap.dedent(code)
    detector = SmellDetector("test_fake.py", source)
    tree = ast.parse(source)
    detector.visit(tree)
    return detector.smells


def _oracle(code: str) -> OracleStrengthReport:
    """Run OracleStrengthAnalyzer on the first test function in code."""
    source = textwrap.dedent(code)
    tree = ast.parse(source)
    analyzer = OracleStrengthAnalyzer()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_"):
                return analyzer.analyze_test(node, source.split("\n"))
    raise ValueError("No test function found in code")


# ---------------------------------------------------------------------------
# SmellDetector tests
# ---------------------------------------------------------------------------


class TestSmellDetector:
    """Test AST-based smell detection."""

    def test_assertion_roulette_detected(self):
        smells = _detect("""\
        def test_example():
            assert len(result) > 0
            assert result[0] is not None
            assert result[0].config != {}
        """)
        roulette = [s for s in smells if s.smell_type == TestSmell.ASSERTION_ROULETTE]
        assert len(roulette) == 1
        assert "3/3" in roulette[0].description

    def test_assertion_roulette_not_triggered_with_messages(self):
        smells = _detect("""\
        def test_example():
            assert len(result) > 0, "need results"
            assert result[0] is not None, "first not none"
            assert result[0].config != {}, "config not empty"
        """)
        roulette = [s for s in smells if s.smell_type == TestSmell.ASSERTION_ROULETTE]
        assert len(roulette) == 0

    def test_assertion_roulette_not_triggered_under_threshold(self):
        smells = _detect("""\
        def test_example():
            assert x > 0
            assert y > 0
        """)
        roulette = [s for s in smells if s.smell_type == TestSmell.ASSERTION_ROULETTE]
        assert len(roulette) == 0

    def test_eager_test_by_name(self):
        smells = _detect("""\
        def test_trials_and_configs_and_metrics():
            pass
        """)
        eager = [s for s in smells if s.smell_type == TestSmell.EAGER_TEST]
        assert len(eager) >= 1

    def test_eager_test_not_triggered_single_and(self):
        smells = _detect("""\
        def test_trials_and_configs():
            pass
        """)
        eager = [s for s in smells if s.smell_type == TestSmell.EAGER_TEST]
        assert len(eager) == 0

    def test_magic_number_detected(self):
        smells = _detect("""\
        def test_example():
            assert result.score > 0.85
        """)
        magic = [s for s in smells if s.smell_type == TestSmell.MAGIC_NUMBER]
        assert len(magic) == 1
        assert "0.85" in str(magic[0].evidence)

    def test_magic_number_ignores_common_values(self):
        smells = _detect("""\
        def test_example():
            assert len(x) >= 1
            assert score > 0
            assert count == 10
        """)
        magic = [s for s in smells if s.smell_type == TestSmell.MAGIC_NUMBER]
        assert len(magic) == 0

    def test_empty_test_detected(self):
        smells = _detect("""\
        def test_example():
            result = some_function()
        """)
        empty = [s for s in smells if s.smell_type == TestSmell.EMPTY_TEST]
        assert len(empty) == 1

    def test_empty_test_not_triggered_with_assert(self):
        smells = _detect("""\
        def test_example():
            result = some_function()
            assert result is not None
        """)
        empty = [s for s in smells if s.smell_type == TestSmell.EMPTY_TEST]
        assert len(empty) == 0

    def test_empty_test_not_triggered_with_pytest_raises(self):
        smells = _detect("""\
        def test_example():
            with pytest.raises(ValueError):
                bad_function()
        """)
        empty = [s for s in smells if s.smell_type == TestSmell.EMPTY_TEST]
        assert len(empty) == 0

    def test_empty_test_not_triggered_with_mock_assert(self):
        smells = _detect("""\
        def test_example():
            mock_obj.assert_called_once()
        """)
        empty = [s for s in smells if s.smell_type == TestSmell.EMPTY_TEST]
        assert len(empty) == 0

    def test_redundant_assert_true(self):
        smells = _detect("""\
        def test_example():
            assert True
        """)
        redundant = [s for s in smells if s.smell_type == TestSmell.REDUNDANT_ASSERTION]
        assert len(redundant) == 1
        assert "assert True" in redundant[0].description

    def test_redundant_self_comparison(self):
        smells = _detect("""\
        def test_example():
            assert x == x
        """)
        redundant = [s for s in smells if s.smell_type == TestSmell.REDUNDANT_ASSERTION]
        assert len(redundant) == 1

    def test_redundant_vacuous_length(self):
        smells = _detect("""\
        def test_example():
            assert len(items) >= 0
        """)
        redundant = [s for s in smells if s.smell_type == TestSmell.REDUNDANT_ASSERTION]
        assert len(redundant) == 1
        assert "always true" in redundant[0].description

    def test_async_tests_detected(self):
        smells = _detect("""\
        async def test_example():
            result = await func()
        """)
        empty = [s for s in smells if s.smell_type == TestSmell.EMPTY_TEST]
        assert len(empty) == 1

    def test_non_test_functions_ignored(self):
        smells = _detect("""\
        def helper_function():
            pass

        def setup_module():
            pass
        """)
        assert len(smells) == 0


# ---------------------------------------------------------------------------
# OracleStrengthAnalyzer tests
# ---------------------------------------------------------------------------


class TestOracleStrengthAnalyzer:
    """Test oracle scoring heuristics."""

    def test_strong_oracle_high_score(self):
        report = _oracle("""\
        def test_example():
            assert len(result.trials) >= 1
            assert result.best_config is not None
            assert result.stop_reason == "max_trials"
            assert result.best_score > 0
            assert result.status == "completed"
        """)
        assert report.oracle_score >= 0.7
        assert report.has_behavior_verification is True
        assert "trials" in report.checked_attributes
        assert "best_config" in report.checked_attributes
        assert "stop_reason" in report.checked_attributes

    def test_no_assertions_zero_score(self):
        report = _oracle("""\
        def test_example():
            result = some_function()
        """)
        assert report.oracle_score == 0.0
        assert report.assertion_count == 0

    def test_isinstance_only_penalised(self):
        report = _oracle("""\
        def test_example():
            assert isinstance(result, dict)
            assert isinstance(result["key"], str)
        """)
        assert "only-isinstance-checks" in report.weak_patterns

    def test_vacuous_length_penalised(self):
        report = _oracle("""\
        def test_example():
            assert len(result.trials) >= 0
        """)
        assert "vacuous-length-checks" in report.weak_patterns

    def test_validator_reliance_only_penalised(self):
        report = _oracle("""\
        def test_example():
            assert not isinstance(result, Exception)
            validation = result_validator(scenario, result)
            assert validation.passed
        """)
        assert "validator-reliance-only" in report.weak_patterns

    def test_exception_guard_only_penalised(self):
        report = _oracle("""\
        def test_example():
            assert not isinstance(result, Exception)
        """)
        assert "exception-guard-only" in report.weak_patterns

    def test_missing_critical_checks(self):
        report = _oracle("""\
        def test_example():
            assert result.trials is not None
        """)
        # Only 'trials' is checked; others should be missing
        assert "best_config" in report.missing_critical_checks
        assert "best_score" in report.missing_critical_checks
        assert "stop_reason" in report.missing_critical_checks

    def test_score_clamped_to_zero(self):
        """Many weak patterns shouldn't push score below 0."""
        report = _oracle("""\
        def test_example():
            assert not isinstance(result, Exception)
        """)
        assert report.oracle_score >= 0.0

    def test_score_clamped_to_one(self):
        """Score shouldn't exceed 1.0 even with many attributes."""
        report = _oracle("""\
        def test_example():
            assert result.trials is not None
            assert result.best_config is not None
            assert result.best_score > 0
            assert result.stop_reason == "max_trials"
            assert result.status == "completed"
            assert result.metrics is not None
            assert result.config is not None
            assert result.score > 0
        """)
        assert report.oracle_score <= 1.0


# ---------------------------------------------------------------------------
# MutationGuidedAnalyzer tests (no LLM calls)
# ---------------------------------------------------------------------------


class TestMutationGuidedAnalyzer:
    """Test mutant targeting (no LLM backend)."""

    def test_targets_missing_trials(self):
        report = OracleStrengthReport(
            test_name="test_x",
            oracle_score=0.3,
            weak_patterns=[],
            checked_attributes=set(),
            missing_critical_checks={"trials", "best_config"},
            assertion_count=1,
            has_behavior_verification=False,
        )
        mutants = MutationGuidedAnalyzer._target_mutants(report)
        assert any("empty trials" in m for m in mutants)
        assert any("best_config" in m for m in mutants)

    def test_targets_weak_patterns(self):
        report = OracleStrengthReport(
            test_name="test_x",
            oracle_score=0.3,
            weak_patterns=["validator-reliance-only", "vacuous-length-checks"],
            checked_attributes=set(),
            missing_critical_checks=set(),
            assertion_count=1,
            has_behavior_verification=False,
        )
        mutants = MutationGuidedAnalyzer._target_mutants(report)
        assert any("Validator passes" in m for m in mutants)
        assert any("empty configs" in m for m in mutants)

    def test_no_mutants_for_strong_oracle(self):
        report = OracleStrengthReport(
            test_name="test_x",
            oracle_score=0.9,
            weak_patterns=[],
            checked_attributes={
                "trials",
                "best_config",
                "best_score",
                "stop_reason",
                "status",
            },
            missing_critical_checks=set(),
            assertion_count=5,
            has_behavior_verification=True,
        )
        mutants = MutationGuidedAnalyzer._target_mutants(report)
        assert len(mutants) == 0

    def test_analyze_returns_none_without_llm(self):
        analyzer = MutationGuidedAnalyzer(llm_backend=None)
        report = OracleStrengthReport(
            test_name="test_x",
            oracle_score=0.3,
            weak_patterns=["validator-reliance-only"],
            checked_attributes=set(),
            missing_critical_checks={"trials"},
            assertion_count=1,
            has_behavior_verification=False,
        )
        result = analyzer.analyze("def test_x(): pass", "test_x", report)
        assert result is None

    def test_parse_response_valid_json(self):
        content = '{"suggestions": [{"assertion": "assert x", "kills_mutant": "M1", "rationale": "reason"}], "confidence": 0.9}'
        result = MutationGuidedAnalyzer._parse_response(content, "test_x", ["M1"])
        assert result is not None
        assert result.test_name == "test_x"
        assert result.confidence == 0.9
        assert len(result.suggested_assertions) == 1

    def test_parse_response_with_markdown_fences(self):
        content = '```json\n{"suggestions": [], "confidence": 0.5}\n```'
        result = MutationGuidedAnalyzer._parse_response(content, "test_x", [])
        assert result is not None
        assert result.confidence == 0.5

    def test_parse_response_invalid_json(self):
        result = MutationGuidedAnalyzer._parse_response("not json", "test_x", [])
        assert result is None


# ---------------------------------------------------------------------------
# Full scanner integration tests
# ---------------------------------------------------------------------------


class TestLLMTestScanner:
    """Integration tests for the scanner."""

    def test_scan_file_produces_all_sections(self, tmp_path: Path):
        test_file = tmp_path / "test_sample.py"
        test_file.write_text(textwrap.dedent("""\
        def test_with_assert():
            assert len(result.trials) >= 1

        def test_empty():
            result = func()

        async def test_async_validator():
            assert not isinstance(result, Exception)
            validation = result_validator(scenario, result)
            assert validation.passed
        """))

        scanner = LLMTestScanner(enable_llm=False)
        result = scanner.scan_file(test_file)

        assert "test_smells" in result
        assert "oracle_reports" in result
        assert "mutation_suggestions" in result
        assert len(result["oracle_reports"]) == 3
        assert len(result["mutation_suggestions"]) == 0  # no LLM

    def test_scan_directory(self, tmp_path: Path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "test_a.py").write_text("def test_one(): assert True\n")
        (tmp_path / "sub" / "test_b.py").write_text("def test_two(): assert 1 == 1\n")
        (tmp_path / "sub" / "not_a_test.py").write_text("x = 1\n")

        scanner = LLMTestScanner(enable_llm=False)
        result = scanner.scan_directory(tmp_path)

        assert result["scanned_files"] == 2
        assert result["summary"]["total_tests"] == 2

    def test_text_report_format(self, tmp_path: Path):
        test_file = tmp_path / "test_sample.py"
        test_file.write_text("def test_x(): assert True\n")

        scanner = LLMTestScanner(enable_llm=False)
        result = scanner.scan_directory(tmp_path)
        report = scanner.format_report(result, "text")

        assert "LLM TEST QUALITY SCANNER REPORT" in report
        assert "Scanned files" in report

    def test_json_report_format(self, tmp_path: Path):
        test_file = tmp_path / "test_sample.py"
        test_file.write_text("def test_x(): assert True\n")

        scanner = LLMTestScanner(enable_llm=False)
        result = scanner.scan_directory(tmp_path)
        report = scanner.format_report(result, "json")

        import json

        data = json.loads(report)
        assert "scanned_files" in data
        assert "summary" in data

    def test_syntax_error_handled(self, tmp_path: Path):
        test_file = tmp_path / "test_bad.py"
        test_file.write_text("def test_x(:\n")

        scanner = LLMTestScanner(enable_llm=False)
        result = scanner.scan_file(test_file)

        assert result["error"] == "SyntaxError"
        assert result["test_smells"] == []

    def test_tools_directory_skipped(self, tmp_path: Path):
        (tmp_path / "tools").mkdir()
        (tmp_path / "tools" / "test_helper.py").write_text("def test_x(): pass\n")
        (tmp_path / "test_real.py").write_text("def test_y(): assert True\n")

        scanner = LLMTestScanner(enable_llm=False)
        result = scanner.scan_directory(tmp_path)

        assert result["scanned_files"] == 1
