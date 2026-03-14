#!/usr/bin/env python3
"""Validate release-review evidence JSON files.

Supports both:
- v2 evidence JSON files (single JSON object)
- legacy markdown tracking files with embedded JSON evidence snippets
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

SHA_PATTERN = re.compile(r"^[a-f0-9]{7,40}$")
UTC_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]


V2_REQUIRED_FIELDS = [
    "schema_version",
    "component",
    "review_type",
    "agent_type",
    "reviewer_model",
    "commit_sha",
    "files_reviewed",
    "findings",
    "strengths",
    "checks_performed",
    "tests",
    "review_summary",
    "decision",
    "timestamp_utc",
]

FILE_ARTIFACT_REQUIRED_FIELDS = [
    "schema_version",
    "component",
    "review_type",
    "agent_type",
    "reviewer_model",
    "file",
    "commit_sha",
    "decision",
    "notes",
    "findings",
    "strengths",
    "checks_performed",
    "timestamp_utc",
]

AGENT_TYPES = {"codex_cli", "claude_cli", "copilot_cli"}
FILE_REVIEW_ANGLES = {
    "security_authz",
    "correctness_regression",
    "async_concurrency_performance",
    "dto_api_contract",
}

LEGACY_REQUIRED_FIELDS = [
    "format",
    "commits",
    "tests",
    "models",
    "reviewer",
    "timestamp",
    "followups",
    "accepted_risks",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def validate_commit_sha(sha: str, repo_path: Path) -> bool:
    if not SHA_PATTERN.match(sha):
        return False

    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def validate_timestamp(value: str) -> bool:
    if not UTC_PATTERN.match(value):
        return False
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
        return True
    except ValueError:
        return False


def validate_path_value(path: Any, *, field_name: str, errors: list[str]) -> None:
    if not isinstance(path, str) or not path.strip():
        errors.append(f"{field_name} must be a non-empty string path")
        return
    cleaned = path.strip().replace("\\", "/")
    if cleaned.startswith("/"):
        errors.append(f"{field_name} must be a relative repository path")
    if ".." in cleaned.split("/"):
        errors.append(f"{field_name} must not contain parent-directory traversal")


def validate_strengths(
    payload: dict[str, Any],
    *,
    field_name: str,
    errors: list[str],
) -> list[dict[str, Any]]:
    strengths = payload.get(field_name)
    if not isinstance(strengths, list):
        errors.append(f"{field_name} must be an array")
        return []

    valid_items: list[dict[str, Any]] = []
    for idx, strength in enumerate(strengths):
        prefix = f"{field_name}[{idx}]"
        if not isinstance(strength, dict):
            errors.append(f"{prefix} must be an object")
            continue
        for key in ["id", "severity", "line", "title", "description"]:
            if key not in strength:
                errors.append(f"{prefix} missing required field: {key}")

        severity = strength.get("severity")
        if severity not in {"S0", "S1", "S2"}:
            errors.append(f"{prefix}.severity must be one of S0,S1,S2")

        line = strength.get("line")
        if not isinstance(line, int) or line < 1:
            errors.append(f"{prefix}.line must be a positive integer")

        title = strength.get("title")
        if not isinstance(title, str) or not title.strip():
            errors.append(f"{prefix}.title must be a non-empty string")

        description = strength.get("description")
        if not isinstance(description, str) or len(description.strip()) < 10:
            errors.append(f"{prefix}.description must be at least 10 characters")

        valid_items.append(strength)
    return valid_items


def validate_checks_performed(
    payload: dict[str, Any],
    *,
    field_name: str,
    errors: list[str],
) -> list[dict[str, Any]]:
    checks = payload.get(field_name)
    if not isinstance(checks, list):
        errors.append(f"{field_name} must be an array")
        return []

    valid_items: list[dict[str, Any]] = []
    categories = {
        "security",
        "correctness",
        "reliability",
        "performance",
        "maintainability",
        "test_quality",
        "operability",
        "documentation",
    }
    results = {"pass", "fail", "warning", "not_applicable"}

    for idx, check in enumerate(checks):
        prefix = f"{field_name}[{idx}]"
        if not isinstance(check, dict):
            errors.append(f"{prefix} must be an object")
            continue
        for key in ["check_id", "category", "result", "evidence"]:
            if key not in check:
                errors.append(f"{prefix} missing required field: {key}")

        check_id = check.get("check_id")
        if not isinstance(check_id, str) or not check_id.strip():
            errors.append(f"{prefix}.check_id must be a non-empty string")

        category = check.get("category")
        if category not in categories:
            errors.append(
                f"{prefix}.category must be one of: {', '.join(sorted(categories))}"
            )

        result = check.get("result")
        if result not in results:
            errors.append(
                f"{prefix}.result must be one of: {', '.join(sorted(results))}"
            )

        evidence = check.get("evidence")
        if not isinstance(evidence, str) or len(evidence.strip()) < 10:
            errors.append(f"{prefix}.evidence must be at least 10 characters")

        valid_items.append(check)
    return valid_items


def validate_v2_payload(payload: Any, repo_path: Path) -> ValidationResult:
    errors: list[str] = []

    if not isinstance(payload, dict):
        return ValidationResult(False, ["payload is not a JSON object"])

    for field in V2_REQUIRED_FIELDS:
        if field not in payload:
            errors.append(f"missing required field: {field}")

    if errors:
        return ValidationResult(False, errors)

    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, int) or schema_version < 2:
        errors.append("schema_version must be an integer >= 2")

    if (
        not isinstance(payload.get("component"), str)
        or not payload["component"].strip()
    ):
        errors.append("component must be a non-empty string")

    if payload.get("review_type") not in {
        "primary",
        "secondary",
        "tertiary",
        "reconciliation",
        "captain",
    }:
        errors.append(
            "review_type must be one of: primary, secondary, tertiary, reconciliation, captain"
        )

    agent_type = payload.get("agent_type")
    if agent_type not in AGENT_TYPES:
        errors.append("agent_type must be one of: codex_cli, claude_cli, copilot_cli")

    if (
        not isinstance(payload.get("reviewer_model"), str)
        or not payload["reviewer_model"].strip()
    ):
        errors.append("reviewer_model must be a non-empty string")

    files_reviewed = payload.get("files_reviewed")
    if not isinstance(files_reviewed, list):
        errors.append("files_reviewed must be an array")
    else:
        for idx, file_path in enumerate(files_reviewed):
            validate_path_value(file_path, field_name=f"files_reviewed[{idx}]", errors=errors)

    commit_sha = payload.get("commit_sha")
    if not isinstance(commit_sha, str) or not validate_commit_sha(
        commit_sha, repo_path
    ):
        errors.append("commit_sha must be a valid reachable git commit SHA")

    if not isinstance(payload.get("timestamp_utc"), str) or not validate_timestamp(
        payload["timestamp_utc"]
    ):
        errors.append("timestamp_utc must be UTC ISO-8601 format: YYYY-MM-DDTHH:MM:SSZ")

    decision = payload.get("decision")
    if decision not in {"approved", "changes_required", "blocked"}:
        errors.append("decision must be approved|changes_required|blocked")

    review_summary = payload.get("review_summary")
    if not isinstance(review_summary, str) or len(review_summary.strip()) < 50:
        errors.append("review_summary must be a non-empty string with at least 50 characters")

    strengths = validate_strengths(payload, field_name="strengths", errors=errors)
    checks_performed = validate_checks_performed(
        payload, field_name="checks_performed", errors=errors
    )

    if payload.get("review_type") in {
        "primary",
        "secondary",
        "tertiary",
        "reconciliation",
    }:
        if isinstance(files_reviewed, list) and len(files_reviewed) == 0:
            errors.append(
                "primary/secondary/tertiary/reconciliation evidence must include at least one files_reviewed entry"
            )
        if len(strengths) == 0:
            errors.append(
                "primary/secondary/tertiary/reconciliation evidence must include at least one strengths entry"
            )
        if len(checks_performed) == 0:
            errors.append(
                "primary/secondary/tertiary/reconciliation evidence must include at least one checks_performed entry"
            )

    findings = payload.get("findings")
    if not isinstance(findings, list):
        errors.append("findings must be an array")
    else:
        for idx, finding in enumerate(findings):
            prefix = f"findings[{idx}]"
            if not isinstance(finding, dict):
                errors.append(f"{prefix} must be an object")
                continue
            for field in ["id", "severity", "file", "line", "title", "repro", "status"]:
                if field not in finding:
                    errors.append(f"{prefix} missing required field: {field}")

            if finding.get("severity") not in {"P0", "P1", "P2", "P3"}:
                errors.append(f"{prefix}.severity must be one of P0,P1,P2,P3")

            line = finding.get("line")
            if not isinstance(line, int) or line < 1:
                errors.append(f"{prefix}.line must be a positive integer")
            validate_path_value(finding.get("file"), field_name=f"{prefix}.file", errors=errors)
            if finding.get("status") not in {"open", "resolved", "waived"}:
                errors.append(f"{prefix}.status must be open|resolved|waived")

    tests = payload.get("tests")
    if not isinstance(tests, list):
        errors.append("tests must be an array")
    else:
        for idx, test in enumerate(tests):
            prefix = f"tests[{idx}]"
            if not isinstance(test, dict):
                errors.append(f"{prefix} must be an object")
                continue
            for field in ["command", "exit_code", "summary"]:
                if field not in test:
                    errors.append(f"{prefix} missing required field: {field}")
            if "exit_code" in test and not isinstance(test["exit_code"], int):
                errors.append(f"{prefix}.exit_code must be an integer")

        if (
            payload.get("review_type") in {"primary", "secondary", "tertiary"}
            and len(tests) == 0
        ):
            errors.append(
                "primary/secondary/tertiary evidence must include at least one executed test entry"
            )

    return ValidationResult(valid=not errors, errors=errors)


def validate_file_review_payload(payload: Any, repo_path: Path) -> ValidationResult:
    errors: list[str] = []

    if not isinstance(payload, dict):
        return ValidationResult(False, ["payload is not a JSON object"])

    for field in FILE_ARTIFACT_REQUIRED_FIELDS:
        if field not in payload:
            errors.append(f"missing required field: {field}")

    if errors:
        return ValidationResult(False, errors)

    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, int) or schema_version < 2:
        errors.append("schema_version must be an integer >= 2")

    if (
        not isinstance(payload.get("component"), str)
        or not payload["component"].strip()
    ):
        errors.append("component must be a non-empty string")

    if payload.get("review_type") not in {
        "primary",
        "secondary",
        "tertiary",
        "reconciliation",
    }:
        errors.append(
            "review_type must be one of: primary, secondary, tertiary, reconciliation"
        )

    agent_type = payload.get("agent_type")
    if agent_type not in AGENT_TYPES:
        errors.append("agent_type must be one of: codex_cli, claude_cli, copilot_cli")

    if (
        not isinstance(payload.get("reviewer_model"), str)
        or not payload["reviewer_model"].strip()
    ):
        errors.append("reviewer_model must be a non-empty string")

    validate_path_value(payload.get("file"), field_name="file", errors=errors)

    angles_reviewed = payload.get("angles_reviewed")
    if angles_reviewed is not None:
        if not isinstance(angles_reviewed, list):
            errors.append("angles_reviewed must be an array when present")
        else:
            invalid_angles = [
                angle for angle in angles_reviewed if angle not in FILE_REVIEW_ANGLES
            ]
            if invalid_angles:
                errors.append(
                    "angles_reviewed contains invalid values: "
                    + ", ".join(sorted(str(angle) for angle in invalid_angles))
                )

    commit_sha = payload.get("commit_sha")
    if not isinstance(commit_sha, str) or not validate_commit_sha(
        commit_sha, repo_path
    ):
        errors.append("commit_sha must be a valid reachable git commit SHA")

    if not isinstance(payload.get("timestamp_utc"), str) or not validate_timestamp(
        payload["timestamp_utc"]
    ):
        errors.append("timestamp_utc must be UTC ISO-8601 format: YYYY-MM-DDTHH:MM:SSZ")

    decision = payload.get("decision")
    if decision not in {"approved", "changes_required", "blocked"}:
        errors.append("decision must be approved|changes_required|blocked")

    notes = payload.get("notes")
    if not isinstance(notes, str) or len(notes.strip()) < 20:
        errors.append("notes must be a non-empty string with at least 20 characters")

    findings = payload.get("findings")
    if not isinstance(findings, list):
        errors.append("findings must be an array")
        findings = []
    else:
        for idx, finding in enumerate(findings):
            prefix = f"findings[{idx}]"
            if not isinstance(finding, dict):
                errors.append(f"{prefix} must be an object")
                continue
            for field in ["id", "severity", "line", "title", "repro", "status"]:
                if field not in finding:
                    errors.append(f"{prefix} missing required field: {field}")

            if finding.get("severity") not in {"P0", "P1", "P2", "P3"}:
                errors.append(f"{prefix}.severity must be one of P0,P1,P2,P3")

            line = finding.get("line")
            if not isinstance(line, int) or line < 1:
                errors.append(f"{prefix}.line must be a positive integer")

            if finding.get("status") not in {"open", "resolved", "waived"}:
                errors.append(f"{prefix}.status must be open|resolved|waived")

    strengths = validate_strengths(payload, field_name="strengths", errors=errors)
    checks_performed = validate_checks_performed(
        payload, field_name="checks_performed", errors=errors
    )

    if decision == "approved":
        if len(strengths) == 0:
            errors.append("approved file artifacts must include at least one strengths entry")
        if len(checks_performed) == 0:
            errors.append(
                "approved file artifacts must include at least one checks_performed entry"
            )
        if len(findings) == 0 and (not isinstance(notes, str) or len(notes.strip()) < 20):
            errors.append(
                "approved file artifacts with empty findings must include notes with at least 20 characters"
            )

    return ValidationResult(valid=not errors, errors=errors)


def validate_legacy_payload(payload: Any) -> ValidationResult:
    errors: list[str] = []

    if not isinstance(payload, dict):
        return ValidationResult(False, ["legacy payload is not a JSON object"])

    for field in LEGACY_REQUIRED_FIELDS:
        if field not in payload:
            errors.append(f"missing legacy field: {field}")

    tests = payload.get("tests")
    if tests is not None and not isinstance(tests, dict):
        errors.append("legacy tests must be an object")

    commits = payload.get("commits")
    if commits is not None and not isinstance(commits, list):
        errors.append("legacy commits must be an array")

    return ValidationResult(valid=not errors, errors=errors)


def extract_json_objects(text: str) -> list[str]:
    objects: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escape = False

    for idx, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    objects.append(text[start : idx + 1])
                    start = -1

    return objects


def validate_markdown_file(path: Path, repo_path: Path) -> ValidationResult:
    text = path.read_text()
    raw_objects = extract_json_objects(text)

    matched = 0
    errors: list[str] = []

    for idx, raw in enumerate(raw_objects):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue

        if not isinstance(payload, dict):
            continue

        if set(V2_REQUIRED_FIELDS).issubset(payload.keys()):
            matched += 1
            result = validate_v2_payload(payload, repo_path)
            if not result.valid:
                errors.extend([f"entry {idx}: {err}" for err in result.errors])
            continue

        if set(FILE_ARTIFACT_REQUIRED_FIELDS).issubset(payload.keys()):
            matched += 1
            result = validate_file_review_payload(payload, repo_path)
            if not result.valid:
                errors.extend([f"entry {idx}: {err}" for err in result.errors])
            continue

        if "format" in payload and "tests" in payload:
            matched += 1
            result = validate_legacy_payload(payload)
            if not result.valid:
                errors.extend([f"entry {idx}: {err}" for err in result.errors])

    if matched == 0:
        return ValidationResult(
            False, [f"no JSON evidence entries found in markdown file: {path}"]
        )

    return ValidationResult(valid=not errors, errors=errors)


def validate_file(path: Path, repo_path: Path) -> ValidationResult:
    if not path.exists():
        return ValidationResult(False, [f"file not found: {path}"])

    if path.suffix.lower() == ".md":
        return validate_markdown_file(path, repo_path)

    try:
        payload = load_json(path)
    except json.JSONDecodeError as exc:
        return ValidationResult(False, [f"invalid JSON in {path}: {exc}"])

    if (
        isinstance(payload, dict)
        and "format" in payload
        and "tests" in payload
        and "component" not in payload
    ):
        return validate_legacy_payload(payload)

    if isinstance(payload, dict) and set(FILE_ARTIFACT_REQUIRED_FIELDS).issubset(
        payload.keys()
    ):
        return validate_file_review_payload(payload, repo_path)

    return validate_v2_payload(payload, repo_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate release review evidence")
    parser.add_argument(
        "--file", required=True, help="Evidence file path (.json or .md)"
    )
    parser.add_argument(
        "--repo-path", default=".", help="Git repo root for commit validation"
    )
    args = parser.parse_args()

    result = validate_file(Path(args.file), Path(args.repo_path))

    if result.valid:
        print("✅ Evidence is valid")
        return 0

    print("❌ Evidence is invalid")
    for err in result.errors:
        print(f"  - {err}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
