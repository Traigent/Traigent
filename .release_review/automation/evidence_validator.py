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
    "component",
    "review_type",
    "reviewer_model",
    "commit_sha",
    "findings",
    "tests",
    "decision",
    "timestamp_utc",
]

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


def validate_v2_payload(payload: Any, repo_path: Path) -> ValidationResult:
    errors: list[str] = []

    if not isinstance(payload, dict):
        return ValidationResult(False, ["payload is not a JSON object"])

    for field in V2_REQUIRED_FIELDS:
        if field not in payload:
            errors.append(f"missing required field: {field}")

    if errors:
        return ValidationResult(False, errors)

    if not isinstance(payload.get("component"), str) or not payload["component"].strip():
        errors.append("component must be a non-empty string")

    if payload.get("review_type") not in {"primary", "secondary", "reconciliation", "captain"}:
        errors.append("review_type must be one of: primary, secondary, reconciliation, captain")

    if not isinstance(payload.get("reviewer_model"), str) or not payload["reviewer_model"].strip():
        errors.append("reviewer_model must be a non-empty string")

    commit_sha = payload.get("commit_sha")
    if not isinstance(commit_sha, str) or not validate_commit_sha(commit_sha, repo_path):
        errors.append("commit_sha must be a valid reachable git commit SHA")

    if not isinstance(payload.get("timestamp_utc"), str) or not validate_timestamp(payload["timestamp_utc"]):
        errors.append("timestamp_utc must be UTC ISO-8601 format: YYYY-MM-DDTHH:MM:SSZ")

    decision = payload.get("decision")
    if decision not in {"approved", "changes_required", "blocked"}:
        errors.append("decision must be approved|changes_required|blocked")

    findings = payload.get("findings")
    if not isinstance(findings, list):
        errors.append("findings must be an array")
    else:
        for idx, finding in enumerate(findings):
            prefix = f"findings[{idx}]"
            if not isinstance(finding, dict):
                errors.append(f"{prefix} must be an object")
                continue
            for field in ["id", "severity", "file", "line", "title", "repro"]:
                if field not in finding:
                    errors.append(f"{prefix} missing required field: {field}")

            if finding.get("severity") not in {"P0", "P1", "P2", "P3"}:
                errors.append(f"{prefix}.severity must be one of P0,P1,P2,P3")

            line = finding.get("line")
            if not isinstance(line, int) or line < 1:
                errors.append(f"{prefix}.line must be a positive integer")

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

        if "format" in payload and "tests" in payload:
            matched += 1
            result = validate_legacy_payload(payload)
            if not result.valid:
                errors.extend([f"entry {idx}: {err}" for err in result.errors])

    if matched == 0:
        return ValidationResult(False, [f"no JSON evidence entries found in markdown file: {path}"])

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

    if isinstance(payload, dict) and "format" in payload and "tests" in payload and "component" not in payload:
        return validate_legacy_payload(payload)

    return validate_v2_payload(payload, repo_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate release review evidence")
    parser.add_argument("--file", required=True, help="Evidence file path (.json or .md)")
    parser.add_argument("--repo-path", default=".", help="Git repo root for commit validation")
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
