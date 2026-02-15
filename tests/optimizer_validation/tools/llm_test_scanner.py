#!/usr/bin/env python3
"""LLM-powered test quality scanner.

Hybrid AST + LLM tool for detecting test smells, scoring oracle strength,
and suggesting assertion improvements via mutation-guided analysis.

Research foundation:
- MuTAP (IST 2024): Re-prompt LLM with surviving mutants
- Meta ACH (FSE 2025): Targeted mutant generation at scale
- AugmenTest (ICST 2025): LLM oracle strengthening from documentation
- SymPrompt (FSE 2024): Code-aware prompting for coverage
- Test Smells in LLM Tests (arxiv 2024): AST-based smell taxonomy

Usage:
    # AST-only (fast, no API key needed)
    python -m tests.optimizer_validation.tools.llm_test_scanner

    # With LLM oracle strengthening suggestions
    python -m tests.optimizer_validation.tools.llm_test_scanner --enable-llm

    # Scan specific directory, save JSON report
    python -m tests.optimizer_validation.tools.llm_test_scanner \
        -d tests/optimizer_validation/dimensions -o json -s report.json
"""

from __future__ import annotations

import ast
import json
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class TestSmell(Enum):
    """Test smell types from SE literature."""

    ASSERTION_ROULETTE = "assertion-roulette"
    EAGER_TEST = "eager-test"
    MAGIC_NUMBER = "magic-number"
    EMPTY_TEST = "empty-test"
    REDUNDANT_ASSERTION = "redundant-assertion"


@dataclass
class SmellDetection:
    """Single detected test smell."""

    smell_type: TestSmell
    file_path: str
    test_name: str
    line_number: int
    severity: str  # high, medium, low
    description: str
    evidence: list[str] = field(default_factory=list)
    suggested_fix: str | None = None


@dataclass
class OracleStrengthReport:
    """Oracle strength analysis for a single test."""

    test_name: str
    oracle_score: float  # 0.0 (weakest) to 1.0 (strongest)
    weak_patterns: list[str]
    checked_attributes: set[str]
    missing_critical_checks: set[str]
    assertion_count: int
    has_behavior_verification: bool


@dataclass
class MutationSuggestion:
    """LLM-generated suggestion for improving a test oracle."""

    test_name: str
    surviving_mutants: list[str]
    suggested_assertions: list[dict[str, str]]
    rationale: str
    confidence: float


# ---------------------------------------------------------------------------
# Component 1: Test Smell Detector (AST-based, no LLM)
# ---------------------------------------------------------------------------

# Numbers commonly used in test setup that aren't "magic"
_COMMON_NUMBERS = frozenset({0, 1, -1, 2, 3, 5, 10, 100, 0.5})


class SmellDetector(ast.NodeVisitor):
    """AST-based test smell detector.

    Detects five smell types from the SE literature:
    - Assertion Roulette: 3+ assertions without explanatory messages
    - Eager Test: tests verifying multiple distinct behaviors
    - Magic Number: unexplained numeric literals in assertions
    - Empty Test: tests with no assertions at all
    - Redundant Assertion: tautologies (assert True, assert x == x)
    """

    def __init__(self, file_path: str, source: str) -> None:
        self.file_path = file_path
        self.source = source
        self.smells: list[SmellDetection] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name.startswith("test_"):
            self._check_all(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if node.name.startswith("test_"):
            self._check_all(node)
        self.generic_visit(node)

    # -- checks --------------------------------------------------------

    def _check_all(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._check_assertion_roulette(node)
        self._check_eager_test(node)
        self._check_magic_numbers(node)
        self._check_empty_test(node)
        self._check_redundant_assertions(node)

    def _check_assertion_roulette(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        """3+ assertions without explanatory messages."""
        assertions = [n for n in ast.walk(node) if isinstance(n, ast.Assert)]
        if len(assertions) < 3:
            return
        no_msg = sum(1 for a in assertions if a.msg is None)
        if no_msg >= 3:
            self.smells.append(
                SmellDetection(
                    smell_type=TestSmell.ASSERTION_ROULETTE,
                    file_path=self.file_path,
                    test_name=node.name,
                    line_number=node.lineno,
                    severity="medium",
                    description=(
                        f"{no_msg}/{len(assertions)} assertions lack messages"
                    ),
                    suggested_fix=(
                        "Add descriptive messages: assert cond, 'explanation'"
                    ),
                )
            )

    def _check_eager_test(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Test verifying multiple distinct behaviors."""
        # Heuristic 1: name contains 2+ "and" segments
        if node.name.count("_and_") >= 2:
            self.smells.append(
                SmellDetection(
                    smell_type=TestSmell.EAGER_TEST,
                    file_path=self.file_path,
                    test_name=node.name,
                    line_number=node.lineno,
                    severity="low",
                    description="Test name suggests multiple behaviors",
                    suggested_fix="Split into focused single-behavior tests",
                )
            )

        # Heuristic 2: multiple scenario_runner / await calls on SUT
        runner_calls = sum(
            1
            for n in ast.walk(node)
            if isinstance(n, ast.Call)
            and isinstance(getattr(n, "func", None), ast.Name)
            and getattr(n.func, "id", "") == "scenario_runner"
        )
        if runner_calls > 1:
            self.smells.append(
                SmellDetection(
                    smell_type=TestSmell.EAGER_TEST,
                    file_path=self.file_path,
                    test_name=node.name,
                    line_number=node.lineno,
                    severity="medium",
                    description=(f"scenario_runner called {runner_calls} times"),
                    suggested_fix="Split into separate tests, one per run",
                )
            )

    def _check_magic_numbers(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        """Unexplained numeric literals inside assertions."""
        hits: list[tuple[int, object]] = []
        for child in ast.walk(node):
            if not isinstance(child, ast.Assert):
                continue
            for inner in ast.walk(child):
                if (
                    isinstance(inner, ast.Constant)
                    and isinstance(inner.value, (int, float))
                    and inner.value not in _COMMON_NUMBERS
                ):
                    hits.append((child.lineno, inner.value))
        if hits:
            self.smells.append(
                SmellDetection(
                    smell_type=TestSmell.MAGIC_NUMBER,
                    file_path=self.file_path,
                    test_name=node.name,
                    line_number=hits[0][0],
                    severity="low",
                    description=f"{len(hits)} magic number(s) in assertions",
                    evidence=[f"line {ln}: {v}" for ln, v in hits[:5]],
                    suggested_fix="Extract to named constants with comments",
                )
            )

    def _check_empty_test(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Test with no assertions at all."""
        has_assert = any(isinstance(n, ast.Assert) for n in ast.walk(node))
        if has_assert:
            return

        # pytest.raises is a valid assertion pattern
        has_raises = any(
            isinstance(n, ast.With)
            and any(
                isinstance(item.context_expr, ast.Call)
                and isinstance(getattr(item.context_expr, "func", None), ast.Attribute)
                and getattr(item.context_expr.func, "attr", "") == "raises"
                for item in n.items
            )
            for n in ast.walk(node)
        )
        if has_raises:
            return

        # Allow mock assert methods (mock.assert_called_once, etc.)
        has_mock_assert = any(
            isinstance(n, ast.Attribute)
            and isinstance(n.attr, str)
            and n.attr.startswith("assert_")
            for n in ast.walk(node)
        )
        if has_mock_assert:
            return

        self.smells.append(
            SmellDetection(
                smell_type=TestSmell.EMPTY_TEST,
                file_path=self.file_path,
                test_name=node.name,
                line_number=node.lineno,
                severity="high",
                description="Test has no assertions",
                suggested_fix="Add explicit assertions or use pytest.raises",
            )
        )

    def _check_redundant_assertions(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        """Tautologies: assert True, assert x == x, assert len(x) >= 0."""
        for child in ast.walk(node):
            if not isinstance(child, ast.Assert):
                continue

            # assert True / assert False
            if isinstance(child.test, ast.Constant) and child.test.value is True:
                self.smells.append(
                    SmellDetection(
                        smell_type=TestSmell.REDUNDANT_ASSERTION,
                        file_path=self.file_path,
                        test_name=node.name,
                        line_number=child.lineno,
                        severity="high",
                        description="Redundant: assert True",
                        suggested_fix="Replace with a meaningful check",
                    )
                )

            # assert x == x / assert x is x
            if isinstance(child.test, ast.Compare) and len(child.test.ops) == 1:
                if isinstance(child.test.ops[0], (ast.Eq, ast.Is)):
                    left = ast.unparse(child.test.left)
                    right = ast.unparse(child.test.comparators[0])
                    if left == right:
                        self.smells.append(
                            SmellDetection(
                                smell_type=TestSmell.REDUNDANT_ASSERTION,
                                file_path=self.file_path,
                                test_name=node.name,
                                line_number=child.lineno,
                                severity="medium",
                                description=f"Redundant: {left} == {right}",
                                suggested_fix="Remove self-comparison",
                            )
                        )

            # assert len(x) >= 0  (always true)
            code = ast.unparse(child.test)
            if "len(" in code and ">= 0" in code:
                self.smells.append(
                    SmellDetection(
                        smell_type=TestSmell.REDUNDANT_ASSERTION,
                        file_path=self.file_path,
                        test_name=node.name,
                        line_number=child.lineno,
                        severity="high",
                        description="Vacuous: len(x) >= 0 is always true",
                        suggested_fix="Use a meaningful lower bound: >= 1",
                    )
                )


# ---------------------------------------------------------------------------
# Component 2: Oracle Strength Analyzer (AST + heuristics)
# ---------------------------------------------------------------------------

# Attributes that strong oracles should verify
CRITICAL_RESULT_ATTRS = frozenset(
    {"trials", "best_config", "best_score", "stop_reason", "status"}
)
CRITICAL_TRIAL_ATTRS = frozenset({"config", "metrics", "status", "score"})


class OracleStrengthAnalyzer:
    """Score oracle strength for test assertions.

    Formula (inspired by AugmenTest, ICST 2025):
        score = base + attr_bonus - pattern_penalty
    Where:
        base = 0.5 if assertions > 0 else 0.0
        attr_bonus = min(0.1 * |critical_attrs_checked|, 0.5)
        pattern_penalty = 0.2 * |weak_patterns|
    """

    def analyze_test(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        source_lines: list[str],
    ) -> OracleStrengthReport:
        assertions = [n for n in ast.walk(node) if isinstance(n, ast.Assert)]
        checked = self._extract_checked_attributes(node)
        weak: list[str] = []

        if self._is_isinstance_dominant(assertions):
            weak.append("only-isinstance-checks")
        if self._has_vacuous_length(assertions):
            weak.append("vacuous-length-checks")
        if self._is_hasattr_dominant(assertions):
            weak.append("hasattr-without-value-check")

        # Validator-reliance-only: uses validator but no explicit attr checks
        func_source = "\n".join(
            source_lines[node.lineno - 1 : node.end_lineno or node.lineno]
        )
        if "validation.passed" in func_source and not (checked & CRITICAL_RESULT_ATTRS):
            weak.append("validator-reliance-only")

        # Not-isinstance as sole guard
        if self._is_exception_guard_only(assertions, func_source):
            weak.append("exception-guard-only")

        missing = CRITICAL_RESULT_ATTRS - checked
        score = self._compute_score(checked, len(assertions), weak)

        return OracleStrengthReport(
            test_name=node.name,
            oracle_score=score,
            weak_patterns=weak,
            checked_attributes=checked,
            missing_critical_checks=missing,
            assertion_count=len(assertions),
            has_behavior_verification=bool(checked & CRITICAL_RESULT_ATTRS),
        )

    # -- scoring -------------------------------------------------------

    @staticmethod
    def _compute_score(
        checked: set[str],
        assertion_count: int,
        weak: list[str],
    ) -> float:
        if assertion_count == 0:
            return 0.0
        score = 0.5
        score += min(len(checked & CRITICAL_RESULT_ATTRS) * 0.1, 0.5)
        score -= len(weak) * 0.2
        return max(0.0, min(1.0, round(score, 2)))

    # -- attribute extraction -----------------------------------------

    @staticmethod
    def _extract_checked_attributes(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> set[str]:
        checked: set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Assert):
                for attr in ast.walk(child):
                    if isinstance(attr, ast.Attribute) and isinstance(attr.attr, str):
                        checked.add(attr.attr)
        return checked

    # -- weak pattern detectors ---------------------------------------

    @staticmethod
    def _is_isinstance_dominant(assertions: list[ast.Assert]) -> bool:
        if not assertions:
            return False
        count = 0
        for a in assertions:
            for n in ast.walk(a):
                if (
                    isinstance(n, ast.Call)
                    and isinstance(getattr(n, "func", None), ast.Name)
                    and getattr(n.func, "id", "") == "isinstance"
                ):
                    count += 1
        return count > len(assertions) / 2

    @staticmethod
    def _has_vacuous_length(assertions: list[ast.Assert]) -> bool:
        for a in assertions:
            code = ast.unparse(a.test)
            if "len(" in code and ">= 0" in code:
                return True
        return False

    @staticmethod
    def _is_hasattr_dominant(assertions: list[ast.Assert]) -> bool:
        if not assertions:
            return False
        count = 0
        for a in assertions:
            for n in ast.walk(a):
                if (
                    isinstance(n, ast.Call)
                    and isinstance(getattr(n, "func", None), ast.Name)
                    and getattr(n.func, "id", "") == "hasattr"
                ):
                    count += 1
        return count > len(assertions) / 2

    @staticmethod
    def _is_exception_guard_only(
        assertions: list[ast.Assert], func_source: str
    ) -> bool:
        """Check if 'assert not isinstance(result, Exception)' is the sole check."""
        if not assertions:
            return False
        guard_count = sum(
            1
            for a in assertions
            if "isinstance" in ast.unparse(a.test)
            and "Exception" in ast.unparse(a.test)
        )
        return guard_count == len(assertions)


# ---------------------------------------------------------------------------
# Component 3: Mutation-Guided Analyzer (opt-in LLM)
# ---------------------------------------------------------------------------

_ORACLE_SYSTEM_PROMPT = """\
You are a test oracle quality expert. Your task is to strengthen weak test \
oracles by suggesting specific assertions that catch semantic bugs \
(represented as mutants).

Guidelines:
1. Suggest EXPLICIT Python assertions (not prose)
2. Include assertion messages explaining what is being checked
3. Focus on behavioral verification, not structural checks
4. Prefer specific value/range checks over existence checks
5. Consider the test name to infer intended behavior

Respond ONLY with valid JSON matching the schema below.
"""

_ORACLE_USER_TEMPLATE = """\
## Test Code
```python
{test_code}
```

## Oracle Strength Analysis
- Score: {score}/1.0
- Weak patterns: {weak_patterns}
- Attributes checked: {checked_attrs}
- Missing critical checks: {missing_checks}

## Surviving Mutants
{mutants_list}

## Task
Suggest 1-3 assertions to kill the surviving mutants.

**Output JSON**:
{{
    "suggestions": [
        {{
            "assertion": "assert len(result.trials) >= 1, 'Should execute at least one trial'",
            "kills_mutant": "MUTANT: Returns empty trials list",
            "rationale": "Fails when trials list is empty"
        }}
    ],
    "confidence": 0.85
}}
"""


class MutationGuidedAnalyzer:
    """Generate oracle improvement suggestions via mutation + LLM.

    Workflow (inspired by MuTAP, IST 2024):
    1. Generate targeted mutants based on missing checks
    2. Prompt LLM with test code + analysis + mutants
    3. Parse structured assertion suggestions
    """

    def __init__(self, llm_backend: Any | None = None) -> None:
        self._llm = llm_backend

    def analyze(
        self,
        test_code: str,
        test_name: str,
        report: OracleStrengthReport,
    ) -> MutationSuggestion | None:
        if self._llm is None:
            return None

        mutants = self._target_mutants(report)
        if not mutants:
            return None

        prompt = _ORACLE_USER_TEMPLATE.format(
            test_code=test_code,
            score=f"{report.oracle_score:.2f}",
            weak_patterns=", ".join(report.weak_patterns) or "none",
            checked_attrs=", ".join(report.checked_attributes) or "none",
            missing_checks=", ".join(report.missing_critical_checks) or "none",
            mutants_list="\n".join(f"- {m}" for m in mutants),
        )

        try:
            response = self._llm.completion(
                model=os.getenv("TRAIGENT_SCANNER_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": _ORACLE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=800,
            )
            return self._parse_response(
                response.choices[0].message.content, test_name, mutants
            )
        except Exception:
            return None

    # -- mutant targeting (from Meta ACH, FSE 2025) --------------------

    @staticmethod
    def _target_mutants(report: OracleStrengthReport) -> list[str]:
        mutants: list[str] = []
        if "trials" in report.missing_critical_checks:
            mutants.append("MUTANT: Returns empty trials list (trials = [])")
        if "best_config" in report.missing_critical_checks:
            mutants.append("MUTANT: Returns None for best_config")
        if "best_score" in report.missing_critical_checks:
            mutants.append("MUTANT: Returns negated best_score")
        if "stop_reason" in report.missing_critical_checks:
            mutants.append("MUTANT: Returns wrong stop_reason")
        if "status" in report.missing_critical_checks:
            mutants.append("MUTANT: Returns incorrect status")
        if "vacuous-length-checks" in report.weak_patterns:
            mutants.append("MUTANT: Returns trials with empty configs")
        if "validator-reliance-only" in report.weak_patterns:
            mutants.append("MUTANT: Validator passes but no trials executed")
        return mutants

    @staticmethod
    def _parse_response(
        content: str, test_name: str, mutants: list[str]
    ) -> MutationSuggestion | None:
        try:
            # Strip markdown fences if present
            text = content.strip()
            if text.startswith("```"):
                text = "\n".join(text.split("\n")[1:])
            if text.endswith("```"):
                text = "\n".join(text.split("\n")[:-1])
            data = json.loads(text)
            return MutationSuggestion(
                test_name=test_name,
                surviving_mutants=mutants,
                suggested_assertions=data.get("suggestions", []),
                rationale="; ".join(
                    s.get("rationale", "") for s in data.get("suggestions", [])
                ),
                confidence=data.get("confidence", 0.5),
            )
        except (json.JSONDecodeError, KeyError):
            return None


# ---------------------------------------------------------------------------
# Main scanner
# ---------------------------------------------------------------------------


class LLMTestScanner:
    """Hybrid AST + LLM test quality scanner."""

    def __init__(
        self,
        enable_llm: bool = False,
        max_llm_calls: int = 20,
    ) -> None:
        self.enable_llm = enable_llm
        self.max_llm_calls = max_llm_calls
        self._llm = None

        if enable_llm:
            try:
                import litellm

                self._llm = litellm
            except ImportError:
                print(
                    "Warning: litellm not installed. LLM features disabled.",
                    file=sys.stderr,
                )
                self.enable_llm = False

    def scan_file(self, file_path: Path) -> dict[str, Any]:
        source = file_path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {
                "file_path": str(file_path),
                "test_smells": [],
                "oracle_reports": [],
                "mutation_suggestions": [],
                "error": "SyntaxError",
            }

        source_lines = source.split("\n")

        # Phase 1: smell detection (AST, fast)
        detector = SmellDetector(str(file_path), source)
        detector.visit(tree)

        # Phase 2: oracle strength (AST + heuristics)
        oracle_analyzer = OracleStrengthAnalyzer()
        oracle_reports: list[OracleStrengthReport] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("test_"):
                    oracle_reports.append(
                        oracle_analyzer.analyze_test(node, source_lines)
                    )

        # Phase 3: mutation-guided suggestions (LLM, slow)
        suggestions: list[MutationSuggestion] = []
        if self.enable_llm and self._llm:
            analyzer = MutationGuidedAnalyzer(self._llm)
            weak = sorted(
                [r for r in oracle_reports if r.oracle_score < 0.6],
                key=lambda r: r.oracle_score,
            )
            for report in weak[: self.max_llm_calls]:
                code = self._extract_test_source(tree, source_lines, report.test_name)
                if code:
                    sug = analyzer.analyze(code, report.test_name, report)
                    if sug:
                        suggestions.append(sug)

        return {
            "file_path": str(file_path),
            "test_smells": detector.smells,
            "oracle_reports": oracle_reports,
            "mutation_suggestions": suggestions,
        }

    def scan_directory(self, directory: Path) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for fp in sorted(directory.rglob("test_*.py")):
            # Skip tooling and __pycache__
            parts = fp.relative_to(directory).parts
            if "tools" in parts or "__pycache__" in parts:
                continue
            results.append(self.scan_file(fp))
        return {
            "scanned_files": len(results),
            "results": results,
            "summary": self._summarise(results),
        }

    # -- report --------------------------------------------------------

    def format_report(
        self, scan: dict[str, Any], fmt: Literal["text", "json"] = "text"
    ) -> str:
        if fmt == "json":
            return json.dumps(scan, indent=2, default=_json_default)

        s = scan["summary"]
        lines = [
            "=" * 72,
            "  LLM TEST QUALITY SCANNER REPORT",
            "=" * 72,
            "",
            f"  Scanned files : {scan['scanned_files']}",
            f"  Total tests   : {s['total_tests']}",
            f"  Test smells   : {s['total_smells']}",
            f"  Weak oracles  : {s['weak_oracles']} (score < 0.6)",
            f"  Strong oracles: {s['strong_oracles']} (score >= 0.8)",
            "",
        ]

        # Smell breakdown
        if s["smell_breakdown"]:
            lines.append("  Smell Breakdown:")
            for name, count in sorted(
                s["smell_breakdown"].items(), key=lambda x: -x[1]
            ):
                lines.append(f"    {name:30s} {count}")
            lines.append("")

        # Weak pattern breakdown
        if s["weak_pattern_breakdown"]:
            lines.append("  Weak Oracle Patterns:")
            for name, count in sorted(
                s["weak_pattern_breakdown"].items(), key=lambda x: -x[1]
            ):
                lines.append(f"    {name:30s} {count}")
            lines.append("")

        # Oracle score histogram
        lines.append("  Oracle Score Distribution:")
        bins = s.get("score_histogram", {})
        for label in [
            "0.0-0.3 (very weak)",
            "0.3-0.6 (weak)",
            "0.6-0.8 (moderate)",
            "0.8-1.0 (strong)",
        ]:
            count = bins.get(label, 0)
            bar = "#" * min(count, 60)
            lines.append(f"    {label:22s} {count:4d} {bar}")
        lines.append("")

        # Top weakest tests
        all_reports = [
            (r["file_path"], rep)
            for r in scan["results"]
            for rep in r["oracle_reports"]
        ]
        weakest = sorted(all_reports, key=lambda x: x[1].oracle_score)[:10]
        if weakest:
            lines.append("  Top 10 Weakest Oracles:")
            lines.append("  " + "-" * 68)
            for _fp, rep in weakest:
                lines.append(f"    {rep.oracle_score:.2f}  {rep.test_name}")
                if rep.weak_patterns:
                    lines.append(f"          patterns: {', '.join(rep.weak_patterns)}")
            lines.append("")

        # LLM suggestions
        all_sug = [sug for r in scan["results"] for sug in r["mutation_suggestions"]]
        if all_sug:
            lines.append(f"  LLM Suggestions ({len(all_sug)} tests):")
            lines.append("  " + "-" * 68)
            for sug in all_sug[:10]:
                lines.append(f"    {sug.test_name}:")
                for s in sug.suggested_assertions:
                    lines.append(f"      + {s.get('assertion', '?')}")
                    lines.append(f"        kills: {s.get('kills_mutant', '?')}")
                lines.append("")

        lines.append("=" * 72)
        return "\n".join(lines)

    # -- helpers -------------------------------------------------------

    @staticmethod
    def _extract_test_source(
        tree: ast.Module,
        source_lines: list[str],
        test_name: str,
    ) -> str | None:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == test_name and node.end_lineno:
                    return "\n".join(source_lines[node.lineno - 1 : node.end_lineno])
        return None

    @staticmethod
    def _summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
        total_tests = 0
        total_smells = 0
        weak = 0
        strong = 0
        smell_counts: dict[str, int] = {}
        pattern_counts: dict[str, int] = {}
        scores: list[float] = []

        for r in results:
            for s in r["test_smells"]:
                total_smells += 1
                key = (
                    s.smell_type.value
                    if isinstance(s.smell_type, TestSmell)
                    else str(s.smell_type)
                )
                smell_counts[key] = smell_counts.get(key, 0) + 1

            for rep in r["oracle_reports"]:
                total_tests += 1
                scores.append(rep.oracle_score)
                if rep.oracle_score < 0.6:
                    weak += 1
                if rep.oracle_score >= 0.8:
                    strong += 1
                for p in rep.weak_patterns:
                    pattern_counts[p] = pattern_counts.get(p, 0) + 1

        # Build histogram
        histogram: dict[str, int] = {
            "0.0-0.3 (very weak)": 0,
            "0.3-0.6 (weak)": 0,
            "0.6-0.8 (moderate)": 0,
            "0.8-1.0 (strong)": 0,
        }
        for sc in scores:
            if sc < 0.3:
                histogram["0.0-0.3 (very weak)"] += 1
            elif sc < 0.6:
                histogram["0.3-0.6 (weak)"] += 1
            elif sc < 0.8:
                histogram["0.6-0.8 (moderate)"] += 1
            else:
                histogram["0.8-1.0 (strong)"] += 1

        return {
            "total_tests": total_tests,
            "total_smells": total_smells,
            "weak_oracles": weak,
            "strong_oracles": strong,
            "smell_breakdown": smell_counts,
            "weak_pattern_breakdown": pattern_counts,
            "score_histogram": histogram,
            "avg_score": round(sum(scores) / len(scores), 3) if scores else 0.0,
        }


def _json_default(obj: Any) -> Any:
    """JSON serializer for dataclasses and enums."""
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "__dataclass_fields__"):
        from dataclasses import asdict

        return asdict(obj)
    return str(obj)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="LLM-powered test quality scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=Path,
        default=Path("tests/optimizer_validation"),
        help="Directory to scan (default: tests/optimizer_validation)",
    )
    parser.add_argument(
        "-o",
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "-s",
        "--save",
        type=Path,
        help="Save report to file",
    )
    parser.add_argument(
        "--enable-llm",
        action="store_true",
        help="Enable LLM oracle strengthening (requires litellm + API key)",
    )
    parser.add_argument(
        "--max-llm-calls",
        type=int,
        default=20,
        help="Maximum LLM calls per scan (default: 20)",
    )
    args = parser.parse_args()

    scanner = LLMTestScanner(
        enable_llm=args.enable_llm,
        max_llm_calls=args.max_llm_calls,
    )

    if not args.directory.is_dir():
        print(f"Error: {args.directory} is not a directory", file=sys.stderr)
        return 1

    result = scanner.scan_directory(args.directory)
    report = scanner.format_report(result, args.output)

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(report, encoding="utf-8")
        print(f"Report saved to {args.save}")
    else:
        print(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
