#!/usr/bin/env python3
"""Signature Mismatch Detection Script.

Scans the Traigent codebase to detect signature mismatches between:
1. Callables passed to Traigent APIs (custom_evaluator, scoring_function, constraints, etc.)
2. Expected interfaces defined by Traigent

Usage:
    python scripts/check_signature_mismatches.py
    python scripts/check_signature_mismatches.py --fix  # Show suggested fixes
    python scripts/check_signature_mismatches.py --verbose  # Show all findings
"""

import argparse
import ast
import inspect
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from traigent.utils.secure_path import PathTraversalError, safe_read_text, validate_path
# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class InterfaceContract:
    """Defines an expected interface for a Traigent API parameter."""

    name: str
    expected_params: list[str]
    expected_return: str
    description: str
    alternative: str | None = None  # Suggest alternative if mismatch detected


# Known interface contracts
CONTRACTS = {
    "custom_evaluator": InterfaceContract(
        name="custom_evaluator",
        expected_params=["func", "config", "example"],
        expected_return="ExampleResult",
        description="Full control evaluator that receives the function, config, and example",
        alternative="scoring_function or metric_functions",
    ),
    "scoring_function": InterfaceContract(
        name="scoring_function",
        expected_params=["prediction", "expected", "input_data"],
        expected_return="dict[str, float]",
        description="Metric evaluator that scores predictions against expected values",
    ),
    "metric_functions": InterfaceContract(
        name="metric_functions",
        expected_params=["prediction", "expected", "input_data"],
        expected_return="float",
        description="Individual metric function that returns a single score",
    ),
    "constraints": InterfaceContract(
        name="constraints",
        expected_params=["config"],  # or ["config", "metrics"]
        expected_return="bool",
        description="Constraint function that validates configurations",
    ),
}

# Patterns that suggest metric evaluator interface
METRIC_EVALUATOR_PATTERNS = {
    "prediction",
    "expected",
    "input_data",
    "ground_truth",
    "label",
}

# Patterns that suggest custom_evaluator interface
CUSTOM_EVALUATOR_PATTERNS = {
    "func",
    "config",
    "example",
    "function",
    "cfg",
}


@dataclass
class SignatureFinding:
    """A detected signature issue."""

    file_path: str
    line_number: int
    api_param: str
    callable_name: str
    actual_params: list[str]
    expected_params: list[str]
    severity: str  # "error", "warning", "info"
    message: str
    suggested_fix: str | None = None


def find_python_files(root: Path, exclude_dirs: set[str] | None = None) -> list[Path]:
    """Find all Python files in the project."""
    if exclude_dirs is None:
        exclude_dirs = {".venv", "venv", "__pycache__", ".git", "node_modules", "build", "dist"}

    python_files = []
    for path in root.rglob("*.py"):
        # Skip excluded directories
        if any(excluded in path.parts for excluded in exclude_dirs):
            continue
        python_files.append(path)
    return python_files


def extract_api_usages(file_path: Path) -> list[dict[str, Any]]:
    """Extract Traigent API usages from a Python file."""
    usages = []

    try:
        safe_path = validate_path(file_path, PROJECT_ROOT, must_exist=True)
        content = safe_read_text(safe_path, PROJECT_ROOT)
    except (PathTraversalError, FileNotFoundError, OSError, UnicodeDecodeError):
        return usages

    # Parse AST
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return usages

    # Find decorator calls and function calls with our target parameters
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Check for @traigent.optimize() or EvaluationOptions()
            call_name = _get_call_name(node)
            if call_name in ("traigent.optimize", "optimize", "EvaluationOptions"):
                for keyword in node.keywords:
                    if keyword.arg in CONTRACTS:
                        usages.append(
                            {
                                "file_path": str(file_path),
                                "line_number": keyword.lineno,
                                "api_param": keyword.arg,
                                "value_node": keyword.value,
                                "content": content,
                            }
                        )

    return usages


def _get_call_name(node: ast.Call) -> str:
    """Get the full name of a call (e.g., 'traigent.optimize')."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    elif isinstance(node.func, ast.Attribute):
        parts = []
        current = node.func
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))
    return ""


def resolve_callable_signature(
    value_node: ast.expr, file_path: str, content: str
) -> tuple[str, list[str]] | None:
    """Try to resolve the callable and get its signature.

    Returns (callable_name, param_names) or None if unresolvable.
    """
    # Case 1: Direct function reference - func_name
    if isinstance(value_node, ast.Name):
        callable_name = value_node.id
        params = _find_function_params_in_file(callable_name, content)
        if params is not None:
            return callable_name, params
        return callable_name, []

    # Case 2: Class instantiation - ClassName()
    if isinstance(value_node, ast.Call):
        if isinstance(value_node.func, ast.Name):
            class_name = value_node.func.id
            params = _find_class_call_params_in_file(class_name, content)
            if params is not None:
                return class_name, params
            # Try to find the class definition and get __call__ params
            return class_name, []
        elif isinstance(value_node.func, ast.Attribute):
            # module.ClassName()
            class_name = value_node.func.attr
            return class_name, []

    # Case 3: Lambda - lambda x, y: ...
    if isinstance(value_node, ast.Lambda):
        params = [arg.arg for arg in value_node.args.args]
        return "lambda", params

    # Case 4: Attribute access - module.func
    if isinstance(value_node, ast.Attribute):
        return value_node.attr, []

    return None


def _find_function_params_in_file(func_name: str, content: str) -> list[str] | None:
    """Find function definition and extract parameters."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return [arg.arg for arg in node.args.args]
    return None


def _find_class_call_params_in_file(class_name: str, content: str) -> list[str] | None:
    """Find class definition and extract __call__ parameters."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            # Find __call__ method
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__call__":
                    params = [arg.arg for arg in item.args.args]
                    # Remove 'self'
                    if params and params[0] == "self":
                        params = params[1:]
                    return params
    return None


def analyze_signature(
    api_param: str, actual_params: list[str], callable_name: str
) -> tuple[str, str, str | None]:
    """Analyze if the signature matches the expected interface.

    Returns (severity, message, suggested_fix).
    """
    contract = CONTRACTS.get(api_param)
    if not contract:
        return "info", f"Unknown API parameter: {api_param}", None

    actual_set = set(actual_params)

    # Check for metric evaluator signature used with custom_evaluator
    if api_param == "custom_evaluator":
        metric_overlap = actual_set & METRIC_EVALUATOR_PATTERNS
        if len(metric_overlap) >= 2:
            return (
                "error",
                f"Signature mismatch: {callable_name} has metric evaluator parameters "
                f"({', '.join(sorted(metric_overlap))}) but is used as custom_evaluator.\n"
                f"Expected: custom_evaluator(func, config, example) -> ExampleResult\n"
                f"Got:      {callable_name}({', '.join(actual_params)})",
                f"Change 'custom_evaluator={callable_name}(...)' to 'scoring_function={callable_name}(...)'",
            )

    # Check for custom_evaluator signature used with scoring_function
    if api_param == "scoring_function":
        custom_overlap = actual_set & CUSTOM_EVALUATOR_PATTERNS
        if len(custom_overlap) >= 2:
            return (
                "error",
                f"Signature mismatch: {callable_name} has custom_evaluator parameters "
                f"({', '.join(sorted(custom_overlap))}) but is used as scoring_function.\n"
                f"Expected: scoring_function(prediction, expected, input_data) -> dict\n"
                f"Got:      {callable_name}({', '.join(actual_params)})",
                f"Change 'scoring_function={callable_name}(...)' to 'custom_evaluator={callable_name}(...)'",
            )

    # Check parameter count
    expected_count = len(contract.expected_params)
    actual_count = len(actual_params)

    if actual_params and actual_count < expected_count - 1:
        return (
            "warning",
            f"Parameter count mismatch: {callable_name} has {actual_count} params, "
            f"expected ~{expected_count} for {api_param}",
            None,
        )

    return "info", f"Signature looks OK for {api_param}", None


def check_file_for_mismatches(file_path: Path) -> list[SignatureFinding]:
    """Check a single file for signature mismatches."""
    findings = []
    usages = extract_api_usages(file_path)

    for usage in usages:
        result = resolve_callable_signature(
            usage["value_node"], usage["file_path"], usage["content"]
        )

        if result is None:
            continue

        callable_name, actual_params = result

        # Skip if we couldn't resolve parameters
        if not actual_params:
            # Try to find the class/function in imported modules
            # For now, just note that we couldn't resolve
            findings.append(
                SignatureFinding(
                    file_path=usage["file_path"],
                    line_number=usage["line_number"],
                    api_param=usage["api_param"],
                    callable_name=callable_name,
                    actual_params=[],
                    expected_params=CONTRACTS[usage["api_param"]].expected_params,
                    severity="info",
                    message=f"Could not resolve signature for {callable_name}",
                )
            )
            continue

        severity, message, suggested_fix = analyze_signature(
            usage["api_param"], actual_params, callable_name
        )

        findings.append(
            SignatureFinding(
                file_path=usage["file_path"],
                line_number=usage["line_number"],
                api_param=usage["api_param"],
                callable_name=callable_name,
                actual_params=actual_params,
                expected_params=CONTRACTS[usage["api_param"]].expected_params,
                severity=severity,
                message=message,
                suggested_fix=suggested_fix,
            )
        )

    return findings


def check_dynamic_imports(file_path: Path) -> list[SignatureFinding]:
    """Check evaluators loaded via importlib.util.spec_from_file_location.

    Detects patterns like:
        _evaluator_path = Path(__file__).parent.parent / "eval" / "evaluator.py"
        _spec = importlib.util.spec_from_file_location("evaluator", _evaluator_path)
        ...
        SomeEvaluator = _evaluator_module.SomeEvaluator
    """
    findings = []

    try:
        safe_path = validate_path(file_path, PROJECT_ROOT, must_exist=True)
        content = safe_read_text(safe_path, PROJECT_ROOT)
    except (OSError, UnicodeDecodeError):
        return findings

    # Look for _evaluator_path pattern
    eval_path_pattern = r'_evaluator_path\s*=\s*Path\(__file__\)\.parent\.parent\s*/\s*"eval"\s*/\s*"evaluator\.py"'
    if not re.search(eval_path_pattern, content):
        return findings

    # Found the pattern - resolve the evaluator file path
    eval_file = file_path.parent.parent / "eval" / "evaluator.py"
    if not eval_file.exists():
        return findings

    try:
        safe_eval = validate_path(eval_file, PROJECT_ROOT, must_exist=True)
        eval_content = safe_read_text(safe_eval, PROJECT_ROOT)
    except (OSError, UnicodeDecodeError):
        return findings

    # Find all evaluator class names extracted from the module
    # Pattern: SomeEvaluator = _evaluator_module.SomeEvaluator
    extract_pattern = r"(\w+Evaluator)\s*=\s*_evaluator_module\.(\w+)"
    extract_matches = re.finditer(extract_pattern, content)

    evaluator_classes = {}
    for match in extract_matches:
        local_name = match.group(1)
        class_name = match.group(2)
        evaluator_classes[local_name] = class_name

    # Now find usages of these evaluators
    for local_name, class_name in evaluator_classes.items():
        # Look for custom_evaluator=LocalName() or scoring_function=LocalName()
        usage_pattern = rf"(custom_evaluator|scoring_function)\s*=\s*{local_name}\s*\("
        usage_matches = re.finditer(usage_pattern, content)

        for match in usage_matches:
            api_param = match.group(1)
            line_number = content[: match.start()].count("\n") + 1

            # Get the __call__ signature from the evaluator file
            params = _find_class_call_params_in_file(class_name, eval_content)

            if params:
                severity, message, suggested_fix = analyze_signature(
                    api_param, params, class_name
                )

                findings.append(
                    SignatureFinding(
                        file_path=str(file_path),
                        line_number=line_number,
                        api_param=api_param,
                        callable_name=class_name,
                        actual_params=params,
                        expected_params=CONTRACTS[api_param].expected_params,
                        severity=severity,
                        message=message,
                        suggested_fix=suggested_fix,
                    )
                )

    return findings


def check_imported_evaluators(file_path: Path) -> list[SignatureFinding]:
    """Check evaluators that are imported from other files."""
    findings = []

    try:
        safe_path = validate_path(file_path, PROJECT_ROOT, must_exist=True)
        content = safe_read_text(safe_path, PROJECT_ROOT)
    except (OSError, UnicodeDecodeError):
        return findings

    # Look for patterns like:
    # custom_evaluator=SomeEvaluator()
    # where SomeEvaluator is imported
    pattern = r"(custom_evaluator|scoring_function)\s*=\s*(\w+)\s*\("
    matches = re.finditer(pattern, content)

    for match in matches:
        api_param = match.group(1)
        evaluator_name = match.group(2)
        line_number = content[: match.start()].count("\n") + 1

        # Try to find the import
        import_pattern = rf"from\s+[\w.]+\s+import\s+.*{evaluator_name}"
        import_match = re.search(import_pattern, content)

        if import_match:
            # Try to resolve the imported module
            import_line = import_match.group(0)
            module_match = re.search(r"from\s+([\w.]+)\s+import", import_line)
            if module_match:
                module_path = module_match.group(1)
                # Convert module path to file path
                evaluator_file = PROJECT_ROOT / module_path.replace(".", "/")

                # Try both .py and looking in parent eval directory
                for eval_path in [
                    evaluator_file.with_suffix(".py"),
                    evaluator_file / "evaluator.py",
                    evaluator_file.parent / "eval" / "evaluator.py",
                ]:
                    if eval_path.exists():
                        try:
                            safe_eval = validate_path(
                                eval_path, PROJECT_ROOT, must_exist=True
                            )
                            eval_content = safe_read_text(safe_eval, PROJECT_ROOT)
                            params = _find_class_call_params_in_file(
                                evaluator_name, eval_content
                            )
                            if params:
                                severity, message, suggested_fix = analyze_signature(
                                    api_param, params, evaluator_name
                                )
                                if severity == "error":
                                    findings.append(
                                        SignatureFinding(
                                            file_path=str(file_path),
                                            line_number=line_number,
                                            api_param=api_param,
                                            callable_name=evaluator_name,
                                            actual_params=params,
                                            expected_params=CONTRACTS[
                                                api_param
                                            ].expected_params,
                                            severity=severity,
                                            message=message,
                                            suggested_fix=suggested_fix,
                                        )
                                    )
                                break
                        except (OSError, UnicodeDecodeError):
                            continue

    return findings


def scan_codebase(root: Path, verbose: bool = False) -> list[SignatureFinding]:
    """Scan the entire codebase for signature mismatches."""
    all_findings = []
    python_files = find_python_files(root)

    for file_path in python_files:
        # Check within-file definitions
        findings = check_file_for_mismatches(file_path)
        all_findings.extend(findings)

        # Check imported evaluators
        imported_findings = check_imported_evaluators(file_path)
        all_findings.extend(imported_findings)

        # Check dynamic imports (importlib.util.spec_from_file_location pattern)
        dynamic_findings = check_dynamic_imports(file_path)
        all_findings.extend(dynamic_findings)

    # Filter based on severity
    if not verbose:
        all_findings = [f for f in all_findings if f.severity in ("error", "warning")]

    return all_findings


def print_findings(findings: list[SignatureFinding], show_fix: bool = False) -> None:
    """Print findings in a readable format."""
    if not findings:
        print("\n No signature mismatches detected!")
        return

    # Group by severity
    errors = [f for f in findings if f.severity == "error"]
    warnings = [f for f in findings if f.severity == "warning"]

    print("\n" + "=" * 70)
    print("SIGNATURE MISMATCH REPORT")
    print("=" * 70)

    if errors:
        print(f"\n ERRORS ({len(errors)} found)")
        print("-" * 70)
        for finding in errors:
            print(f"\nFile: {finding.file_path}:{finding.line_number}")
            print(f"API Parameter: {finding.api_param}")
            print(f"Callable: {finding.callable_name}")
            print(f"Actual params: ({', '.join(finding.actual_params)})")
            print(f"Expected params: ({', '.join(finding.expected_params)})")
            print(f"\n{finding.message}")
            if show_fix and finding.suggested_fix:
                print(f"\n Suggested fix: {finding.suggested_fix}")

    if warnings:
        print(f"\n WARNINGS ({len(warnings)} found)")
        print("-" * 70)
        for finding in warnings:
            print(f"\nFile: {finding.file_path}:{finding.line_number}")
            print(f"API Parameter: {finding.api_param}")
            print(f"Callable: {finding.callable_name}")
            print(f"\n{finding.message}")

    print("\n" + "=" * 70)
    print(f"Summary: {len(errors)} errors, {len(warnings)} warnings")
    print("=" * 70)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Detect signature mismatches in Traigent API usages"
    )
    parser.add_argument(
        "--fix", action="store_true", help="Show suggested fixes for mismatches"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show all findings including info-level"
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=PROJECT_ROOT,
        help="Path to scan (default: project root)",
    )
    parser.add_argument(
        "--json", action="store_true", help="Output as JSON"
    )

    args = parser.parse_args()

    print(f"Scanning {args.path} for signature mismatches...")
    findings = scan_codebase(args.path, verbose=args.verbose)

    if args.json:
        import json

        output = [
            {
                "file": f.file_path,
                "line": f.line_number,
                "api_param": f.api_param,
                "callable": f.callable_name,
                "actual_params": f.actual_params,
                "expected_params": f.expected_params,
                "severity": f.severity,
                "message": f.message,
                "fix": f.suggested_fix,
            }
            for f in findings
        ]
        print(json.dumps(output, indent=2))
    else:
        print_findings(findings, show_fix=args.fix)

    # Return error code if errors found
    errors = [f for f in findings if f.severity == "error"]
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
