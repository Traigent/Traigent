#!/usr/bin/env python3
"""Analyze optimizer validation tests for weak/vulnerable patterns.

This tool combines static and optional dynamic analysis to flag tests that
exhibit the root causes and issue types defined in the optimizer validation
meta-analysis (IT-VRO, IT-CBM, IT-VTA; RC-MA, RC-WA, RC-MV).

Usage:
    python -m tests.optimizer_validation.tools.weak_test_analyzer
    python -m tests.optimizer_validation.tools.weak_test_analyzer --report report.json
    python -m tests.optimizer_validation.tools.weak_test_analyzer --json --output weak_test_report.json
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCENARIO_FACTORIES = {
    "TestScenario",
    "basic_scenario",
    "config_space_scenario",
    "constrained_scenario",
    "evaluator_scenario",
    "multi_objective_scenario",
    "failure_scenario",
}

BEHAVIOR_KEYWORDS = {
    "timeout",
    "max_trials",
    "max_trial",
    "stop",
    "constraint",
    "mapping",
    "metric",
    "objective",
    "direction",
    "weight",
    "injection",
    "seed",
    "reproducibility",
    "deterministic",
    "parallel",
}

RESULT_FIELDS = {
    "stop_reason",
    "metrics",
    "best_config",
    "best_score",
    "config",
    "trials",
}


@dataclass
class ScenarioSummary:
    """Extracted scenario details for mismatch checks."""

    name: str | None = None
    max_trials: int | None = None
    timeout: float | None = None
    expected_outcome: str | None = None
    expected_stop_reason: str | None = None
    algorithm: str | None = None
    config_space_size: int | None = None
    has_continuous: bool | None = None


@dataclass
class StaticTestInfo:
    """Static analysis data for a single test function."""

    nodeid: str
    file_path: str
    class_name: str | None
    function: str
    lineno: int
    docstring: str | None
    calls: set[str] = field(default_factory=set)
    assert_count: int = 0
    has_validation_assert: bool = False
    has_exception_assert: bool = False
    has_behavior_assert: bool = False
    has_stop_reason_assert: bool = False
    has_metrics_assert: bool = False
    has_config_assert: bool = False
    has_trials_assert: bool = False
    has_vacuous_assert: bool = False
    uses_expected_failure: bool = False
    scenarios: list[ScenarioSummary] = field(default_factory=list)
    vacuous_examples: list[str] = field(default_factory=list)


@dataclass
class DynamicTestInfo:
    """Dynamic analysis data derived from pytest JSON report evidence."""

    nodeid: str
    outcome: str
    evidence_found: bool
    evidence: dict[str, Any] | None = None


@dataclass
class Finding:
    """A single weak-test finding."""

    test_id: str
    issue_type: str
    root_cause: str
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


def _get_call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _safe_literal_eval(node: ast.AST) -> Any | None:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _collect_names_attrs(node: ast.AST) -> tuple[set[str], set[str]]:
    names: set[str] = set()
    attrs: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.add(child.id)
        elif isinstance(child, ast.Attribute):
            attrs.add(child.attr)
    return names, attrs


def _assert_contains_call(node: ast.AST, call_name: str) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and _get_call_name(child) == call_name:
            return True
    return False


def _assert_isinstance_result(expr: ast.AST) -> bool:
    # Handle direct or negated isinstance(result, Exception) patterns.
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
        expr = expr.operand
    if isinstance(expr, ast.Call) and _get_call_name(expr) == "isinstance":
        if expr.args and isinstance(expr.args[0], ast.Name):
            return expr.args[0].id == "result"
    return False


def _is_assert_true(expr: ast.AST) -> bool:
    return isinstance(expr, ast.Constant) and expr.value is True


def _is_len_ge_zero(expr: ast.AST) -> bool:
    if not isinstance(expr, ast.Compare):
        return False
    if not isinstance(expr.left, ast.Call):
        return False
    if _get_call_name(expr.left) != "len":
        return False
    if not expr.comparators:
        return False
    comparator = expr.comparators[0]
    if not isinstance(comparator, ast.Constant):
        return False
    if comparator.value not in (0, -1):
        return False
    return isinstance(expr.ops[0], (ast.GtE, ast.Gt))


def _is_tautology(expr: ast.AST) -> bool:
    # Detect `assert X or not X`
    if not isinstance(expr, ast.BoolOp):
        return False
    if not isinstance(expr.op, ast.Or):
        return False
    if len(expr.values) != 2:
        return False
    left, right = expr.values
    if isinstance(right, ast.UnaryOp) and isinstance(right.op, ast.Not):
        return ast.dump(left) == ast.dump(right.operand)
    if isinstance(left, ast.UnaryOp) and isinstance(left.op, ast.Not):
        return ast.dump(right) == ast.dump(left.operand)
    return False


def _extract_expected_outcome(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _extract_expected_result(call: ast.Call) -> dict[str, Any]:
    expected: dict[str, Any] = {}
    for kw in call.keywords:
        if kw.arg == "outcome":
            expected["outcome"] = _extract_expected_outcome(kw.value)
        elif kw.arg == "expected_stop_reason":
            expected["expected_stop_reason"] = _safe_literal_eval(kw.value)
        elif kw.arg == "min_trials":
            expected["min_trials"] = _safe_literal_eval(kw.value)
        elif kw.arg == "max_trials":
            expected["max_trials"] = _safe_literal_eval(kw.value)
    return expected


def _extract_algorithm(mock_mode_config: dict[str, Any] | None) -> str | None:
    if not mock_mode_config:
        return None
    optimizer = mock_mode_config.get("optimizer")
    if optimizer:
        return str(optimizer)
    sampler = mock_mode_config.get("sampler")
    if sampler:
        return f"optuna_{sampler}"
    return None


def _compute_config_space_size(
    config_space: dict[str, Any] | None,
) -> tuple[int | None, bool | None]:
    if not config_space:
        return 0, False
    combos = 1
    has_continuous = False
    for value in config_space.values():
        if isinstance(value, tuple) and len(value) == 2:
            has_continuous = True
        elif isinstance(value, list):
            combos *= len(value)
        else:
            return None, None
    if has_continuous:
        return None, True
    return combos, False


def _extract_assignments(node: ast.AST) -> dict[str, Any]:
    assignments: dict[str, Any] = {}
    for child in ast.walk(node):
        if isinstance(child, ast.Assign) and len(child.targets) == 1:
            target = child.targets[0]
            if isinstance(target, ast.Name):
                if (
                    isinstance(child.value, ast.Call)
                    and _get_call_name(child.value) == "ExpectedResult"
                ):
                    assignments[target.id] = _extract_expected_result(child.value)
                else:
                    literal = _safe_literal_eval(child.value)
                    if literal is not None:
                        assignments[target.id] = literal
    return assignments


def _extract_scenarios(
    node: ast.AST, assignments: dict[str, Any]
) -> list[ScenarioSummary]:
    scenarios: list[ScenarioSummary] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and _get_call_name(child) in SCENARIO_FACTORIES:
            scenario = ScenarioSummary()
            for kw in child.keywords:
                if kw.arg == "name":
                    scenario.name = _safe_literal_eval(kw.value)
                elif kw.arg == "max_trials":
                    scenario.max_trials = _safe_literal_eval(kw.value)
                elif kw.arg == "timeout":
                    scenario.timeout = _safe_literal_eval(kw.value)
                elif kw.arg == "config_space":
                    config_space = _safe_literal_eval(kw.value)
                    if config_space is None and isinstance(kw.value, ast.Name):
                        config_space = assignments.get(kw.value.id)
                    scenario.config_space_size, scenario.has_continuous = (
                        _compute_config_space_size(
                            config_space if isinstance(config_space, dict) else None
                        )
                    )
                elif kw.arg == "mock_mode_config":
                    mock_mode = _safe_literal_eval(kw.value)
                    if mock_mode is None and isinstance(kw.value, ast.Name):
                        mock_mode = assignments.get(kw.value.id)
                    scenario.algorithm = _extract_algorithm(
                        mock_mode if isinstance(mock_mode, dict) else None
                    )
                elif kw.arg == "expected":
                    if (
                        isinstance(kw.value, ast.Call)
                        and _get_call_name(kw.value) == "ExpectedResult"
                    ):
                        expected = _extract_expected_result(kw.value)
                    elif isinstance(kw.value, ast.Name):
                        expected = assignments.get(kw.value.id, {})
                    else:
                        expected = {}
                    if expected:
                        scenario.expected_outcome = expected.get("outcome")
                        scenario.expected_stop_reason = expected.get(
                            "expected_stop_reason"
                        )
                        if scenario.max_trials is None:
                            scenario.max_trials = expected.get("max_trials")
            scenarios.append(scenario)
    return scenarios


class StaticAnalyzer(ast.NodeVisitor):
    """AST visitor that extracts static test analysis."""

    def __init__(self, file_path: Path, root_dir: Path) -> None:
        self.file_path = file_path
        self.root_dir = root_dir
        self.current_class: str | None = None
        self.tests: list[StaticTestInfo] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        prev = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = prev

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._maybe_analyze_test(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._maybe_analyze_test(node)

    def _maybe_analyze_test(self, node: ast.AST) -> None:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return
        if not node.name.startswith("test_"):
            return

        rel_path = self.file_path.relative_to(self.root_dir)
        if self.current_class:
            nodeid = f"{rel_path}::{self.current_class}::{node.name}"
        else:
            nodeid = f"{rel_path}::{node.name}"

        docstring = ast.get_docstring(node)
        assignments = _extract_assignments(node)
        scenarios = _extract_scenarios(node, assignments)

        calls: set[str] = set()
        assert_nodes: list[ast.Assert] = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                name = _get_call_name(child)
                if name:
                    calls.add(name)
            elif isinstance(child, ast.Assert):
                assert_nodes.append(child)

        info = StaticTestInfo(
            nodeid=nodeid,
            file_path=str(rel_path),
            class_name=self.current_class,
            function=node.name,
            lineno=node.lineno,
            docstring=docstring,
            calls=calls,
            scenarios=scenarios,
        )

        for assert_node in assert_nodes:
            info.assert_count += 1
            expr = assert_node.test

            if _assert_isinstance_result(expr):
                info.has_exception_assert = True

            if _is_assert_true(expr):
                info.has_vacuous_assert = True
                info.vacuous_examples.append("assert True")
            if _is_tautology(expr):
                info.has_vacuous_assert = True
                info.vacuous_examples.append("assert X or not X")
            if _is_len_ge_zero(expr):
                info.has_vacuous_assert = True
                info.vacuous_examples.append("assert len(x) >= 0")

            names, attrs = _collect_names_attrs(expr)
            if "passed" in attrs or _assert_contains_call(expr, "result_validator"):
                info.has_validation_assert = True

            if "stop_reason" in attrs or "stop_reason" in names:
                info.has_stop_reason_assert = True
            if "metrics" in attrs or "metrics" in names:
                info.has_metrics_assert = True
            if "config" in attrs or "config" in names:
                info.has_config_assert = True
            if "trials" in attrs or "trials" in names:
                info.has_trials_assert = True

        info.has_behavior_assert = any(
            [
                info.has_stop_reason_assert,
                info.has_metrics_assert,
                info.has_config_assert,
            ]
        )

        # Detect ExpectedOutcome.FAILURE usage
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute) and child.attr == "FAILURE":
                info.uses_expected_failure = True
                break

        self.tests.append(info)


def _extract_evidence(stdout: str) -> dict[str, Any] | None:
    if not stdout:
        return None
    marker = '{"type": "TEST_EVIDENCE"'
    start = stdout.find(marker)
    if start == -1:
        return None
    decoder = json.JSONDecoder()
    try:
        evidence, _ = decoder.raw_decode(stdout[start:])
        if isinstance(evidence, dict) and evidence.get("type") == "TEST_EVIDENCE":
            return evidence
    except json.JSONDecodeError:
        return None
    return None


def _load_dynamic_report(report_path: Path) -> dict[str, DynamicTestInfo]:
    data = json.loads(report_path.read_text())
    tests = data.get("tests", [])
    dynamic: dict[str, DynamicTestInfo] = {}
    for test in tests:
        nodeid = test.get("nodeid")
        call = test.get("call", {})
        stdout = call.get("stdout", "") or ""
        outcome = call.get("outcome", test.get("outcome", "unknown"))
        evidence = _extract_evidence(stdout)
        info = DynamicTestInfo(
            nodeid=nodeid,
            outcome=outcome,
            evidence_found=evidence is not None,
            evidence=evidence,
        )
        dynamic[nodeid] = info
    return dynamic


def _add_finding(
    findings: list[Finding],
    seen: set[tuple[str, str, str]],
    finding: Finding,
) -> None:
    key = (finding.test_id, finding.issue_type, finding.message)
    if key in seen:
        return
    seen.add(key)
    findings.append(finding)


def _classify_static(info: StaticTestInfo) -> list[Finding]:
    findings: list[Finding] = []
    seen: set[tuple[str, str, str]] = set()

    name_text = " ".join(
        [
            info.function.lower(),
            (info.docstring or "").lower(),
        ]
    )

    calls_validator = "result_validator" in info.calls

    if calls_validator and not info.has_validation_assert:
        _add_finding(
            findings,
            seen,
            Finding(
                test_id=info.nodeid,
                issue_type="IT-VRO",
                root_cause="RC-MA",
                severity="high",
                message="result_validator called but validation result not asserted",
                details={"file": info.file_path, "line": info.lineno},
            ),
        )

    if calls_validator and not info.has_behavior_assert:
        _add_finding(
            findings,
            seen,
            Finding(
                test_id=info.nodeid,
                issue_type="IT-VRO",
                root_cause="RC-MA",
                severity="medium",
                message="test relies on validator without behavior-specific assertions",
                details={
                    "file": info.file_path,
                    "line": info.lineno,
                    "has_trials_assert": info.has_trials_assert,
                },
            ),
        )

    if info.uses_expected_failure and not (
        info.has_exception_assert or info.has_validation_assert
    ):
        _add_finding(
            findings,
            seen,
            Finding(
                test_id=info.nodeid,
                issue_type="IT-CBM",
                root_cause="RC-MA",
                severity="high",
                message="expected failure without asserting exception or validator result",
                details={"file": info.file_path, "line": info.lineno},
            ),
        )

    if info.has_vacuous_assert:
        _add_finding(
            findings,
            seen,
            Finding(
                test_id=info.nodeid,
                issue_type="IT-VTA",
                root_cause="RC-WA",
                severity="low",
                message="vacuous assertion detected",
                details={
                    "file": info.file_path,
                    "line": info.lineno,
                    "examples": info.vacuous_examples,
                },
            ),
        )

    keyword_hit = any(keyword in name_text for keyword in BEHAVIOR_KEYWORDS)
    if keyword_hit and not info.has_behavior_assert:
        _add_finding(
            findings,
            seen,
            Finding(
                test_id=info.nodeid,
                issue_type="IT-VRO",
                root_cause="RC-MA",
                severity="medium",
                message="test name/docstring implies behavior but lacks behavior-specific asserts",
                details={"file": info.file_path, "line": info.lineno},
            ),
        )

    for scenario in info.scenarios:
        if (
            scenario.expected_stop_reason == "max_trials_reached"
            and scenario.algorithm == "grid"
            and scenario.config_space_size is not None
            and scenario.max_trials is not None
            and scenario.config_space_size < scenario.max_trials
        ):
            _add_finding(
                findings,
                seen,
                Finding(
                    test_id=info.nodeid,
                    issue_type="IT-CBM",
                    root_cause="RC-MA",
                    severity="medium",
                    message="max_trials stop expected but grid space exhausts first",
                    details={
                        "file": info.file_path,
                        "line": info.lineno,
                        "config_space_size": scenario.config_space_size,
                        "max_trials": scenario.max_trials,
                    },
                ),
            )
        if "timeout" in name_text and (
            scenario.timeout is None or scenario.timeout == 0
        ):
            _add_finding(
                findings,
                seen,
                Finding(
                    test_id=info.nodeid,
                    issue_type="IT-CBM",
                    root_cause="RC-MA",
                    severity="low",
                    message="timeout test without timeout configured",
                    details={"file": info.file_path, "line": info.lineno},
                ),
            )

    return findings


def _classify_dynamic(info: DynamicTestInfo) -> list[Finding]:
    findings: list[Finding] = []
    seen: set[tuple[str, str, str]] = set()
    if not info.evidence_found or not info.evidence:
        return findings

    evidence = info.evidence
    expected = evidence.get("expected", {})
    actual = evidence.get("actual", {})
    scenario = evidence.get("scenario", {})
    validation_checks = evidence.get("validation_checks", [])

    if info.outcome == "passed" and evidence.get("passed") is False:
        _add_finding(
            findings,
            seen,
            Finding(
                test_id=info.nodeid,
                issue_type="IT-VRO",
                root_cause="RC-MA",
                severity="high",
                message="validator reported failure but test passed",
                details={"actual": actual, "expected": expected},
            ),
        )

    if info.outcome == "passed":
        for check in validation_checks:
            if check.get("passed") is False:
                _add_finding(
                    findings,
                    seen,
                    Finding(
                        test_id=info.nodeid,
                        issue_type="IT-VRO",
                        root_cause="RC-MA",
                        severity="high",
                        message="validation check failed but test passed",
                        details={
                            "check": check.get("check"),
                            "message": check.get("message"),
                        },
                    ),
                )
                break

    if info.outcome == "passed":
        expected_outcome = expected.get("outcome")
        actual_type = actual.get("type")
        if expected_outcome == "FAILURE" and actual_type == "success":
            _add_finding(
                findings,
                seen,
                Finding(
                    test_id=info.nodeid,
                    issue_type="IT-CBM",
                    root_cause="RC-MA",
                    severity="medium",
                    message="expected failure but run succeeded without failing test",
                    details={
                        "expected_outcome": expected_outcome,
                        "actual_type": actual_type,
                    },
                ),
            )

    expected_stop = expected.get("expected_stop_reason")
    actual_stop = actual.get("stop_reason")
    if info.outcome == "passed" and expected_stop and actual_stop != expected_stop:
        _add_finding(
            findings,
            seen,
            Finding(
                test_id=info.nodeid,
                issue_type="IT-CBM",
                root_cause="RC-MA",
                severity="medium",
                message="expected stop_reason mismatch with passing test",
                details={
                    "expected_stop_reason": expected_stop,
                    "actual_stop_reason": actual_stop,
                },
            ),
        )

    expected_min_trials = expected.get("min_trials")
    actual_trials = actual.get("trial_count")
    if (
        info.outcome == "passed"
        and isinstance(expected_min_trials, int)
        and isinstance(actual_trials, int)
        and actual_trials < expected_min_trials
    ):
        _add_finding(
            findings,
            seen,
            Finding(
                test_id=info.nodeid,
                issue_type="IT-CBM",
                root_cause="RC-MA",
                severity="medium",
                message="trial_count below expected minimum but test passed",
                details={
                    "expected_min_trials": expected_min_trials,
                    "actual_trials": actual_trials,
                },
            ),
        )

    config_space = scenario.get("config_space", {})
    summary = config_space.get("_summary", {})
    total_space = summary.get("total_space")
    algorithm = scenario.get("algorithm")
    max_trials = scenario.get("max_trials")
    if (
        info.outcome == "passed"
        and expected_stop == "max_trials_reached"
        and algorithm == "grid"
        and isinstance(total_space, int)
        and isinstance(max_trials, int)
        and total_space < max_trials
    ):
        _add_finding(
            findings,
            seen,
            Finding(
                test_id=info.nodeid,
                issue_type="IT-CBM",
                root_cause="RC-MA",
                severity="medium",
                message="grid search exhausts space before max_trials stop",
                details={"total_space": total_space, "max_trials": max_trials},
            ),
        )

    return findings


def _aggregate(findings: Iterable[Finding]) -> dict[str, Any]:
    counts_by_issue: dict[str, int] = {}
    counts_by_root: dict[str, int] = {}
    counts_by_severity: dict[str, int] = {}
    for f in findings:
        counts_by_issue[f.issue_type] = counts_by_issue.get(f.issue_type, 0) + 1
        counts_by_root[f.root_cause] = counts_by_root.get(f.root_cause, 0) + 1
        counts_by_severity[f.severity] = counts_by_severity.get(f.severity, 0) + 1
    return {
        "by_issue_type": counts_by_issue,
        "by_root_cause": counts_by_root,
        "by_severity": counts_by_severity,
    }


def _render_text(findings: list[Finding], metadata: dict[str, Any]) -> str:
    summary = _aggregate(findings)
    lines = [
        "WEAK TEST ANALYSIS REPORT",
        "=" * 60,
        f"Generated: {metadata['generated_at']}",
        f"Static tests analyzed: {metadata['static_tests']}",
        f"Dynamic tests analyzed: {metadata['dynamic_tests']}",
        f"Total findings: {len(findings)}",
        "",
        "Findings by issue type:",
    ]
    for issue, count in summary["by_issue_type"].items():
        lines.append(f"  - {issue}: {count}")
    lines.append("")
    lines.append("Findings by root cause:")
    for root, count in summary["by_root_cause"].items():
        lines.append(f"  - {root}: {count}")
    lines.append("")
    lines.append("Findings by severity:")
    for sev, count in summary["by_severity"].items():
        lines.append(f"  - {sev}: {count}")
    lines.append("")
    lines.append("Sample findings:")
    for f in findings[:20]:
        lines.append(f"- {f.issue_type} {f.test_id}: {f.message}")
    if len(findings) > 20:
        lines.append(f"... {len(findings) - 20} more")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-dir",
        default="tests/optimizer_validation",
        help="Root directory to scan for tests.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional pytest JSON report for dynamic analysis.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON path.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON to stdout instead of text.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    test_dir = Path(args.test_dir).resolve()
    root_dir = Path.cwd().resolve()

    static_tests: list[StaticTestInfo] = []
    for path in test_dir.rglob("test_*.py"):
        try:
            source = path.read_text()
        except OSError:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        analyzer = StaticAnalyzer(path, root_dir)
        analyzer.visit(tree)
        static_tests.extend(analyzer.tests)

    dynamic_tests: dict[str, DynamicTestInfo] = {}
    if args.report:
        report_path = Path(args.report)
        if report_path.exists():
            dynamic_tests = _load_dynamic_report(report_path)

    findings: list[Finding] = []
    seen: set[tuple[str, str, str]] = set()

    for test in static_tests:
        for finding in _classify_static(test):
            _add_finding(findings, seen, finding)

    for nodeid, dyn in dynamic_tests.items():
        for finding in _classify_dynamic(dyn):
            _add_finding(findings, seen, finding)

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "static_tests": len(static_tests),
        "dynamic_tests": len(dynamic_tests),
        "report": args.report,
    }
    payload = {
        "metadata": metadata,
        "summary": _aggregate(findings),
        "findings": [
            {
                "test_id": f.test_id,
                "issue_type": f.issue_type,
                "root_cause": f.root_cause,
                "severity": f.severity,
                "message": f.message,
                "details": f.details,
            }
            for f in findings
        ],
    }

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(payload, indent=2))

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(_render_text(findings, metadata))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
