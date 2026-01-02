#!/usr/bin/env python3
"""Evidence Validator for Release Review Protocol.

Parses and validates evidence format in review tracking.
Used by captain to ensure all evidence meets standards.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class ParsedEvidence:
    """Parsed evidence data."""

    format: str
    commits: list[str]
    tests_command: str | None
    tests_status: str | None
    tests_passed: int | None
    tests_total: int | None
    model: str
    reviewer: str
    timestamp: str
    followups: str | None = None
    accepted_risks: str | None = None
    legacy_summary: str | None = None

    @property
    def tests_pass_rate(self) -> float:
        """Calculate test pass rate."""
        if not self.tests_total:
            return 0.0
        return self.tests_passed / self.tests_total


class EvidenceValidator:
    """Validate evidence format and content."""

    VALID_STATUSES = {"PASS", "FAIL", "SKIP", "UNKNOWN"}
    REQUIRED_JSON_FIELDS = {
        "format",
        "commits",
        "tests",
        "models",
        "reviewer",
        "timestamp",
        "followups",
        "accepted_risks",
    }
    REQUIRED_TEST_FIELDS = {"command", "status", "passed", "total"}

    # Pattern for standard evidence format
    # Example: Tests: 47/47 | Commits: abc123 | Model: Claude/Opus4.5 | Time: 2025-12-13T10:00:00Z
    STANDARD_PATTERN = re.compile(
        r"Tests:\s*(\d+)/(\d+)\s*"
        r"(?:passed\s*)?\|?\s*"
        r"Commits?:\s*([a-f0-9]{7,40})\s*\|?\s*"
        r"Model:\s*([^\|]+?)\s*\|?\s*"
        r"(?:Time(?:stamp)?:\s*)?(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z?)",
        re.IGNORECASE,
    )

    # Flexible pattern for various evidence formats
    FLEXIBLE_PATTERNS = {
        "tests": re.compile(
            r"Tests?:\s*(\d+)/(\d+)\s*(?:passed)?",
            re.IGNORECASE,
        ),
        "commit": re.compile(
            r"Commits?:\s*([a-f0-9]{7,40})",
            re.IGNORECASE,
        ),
        "model": re.compile(
            r"Model:\s*([^\|]+?)(?:\s*\||$)",
            re.IGNORECASE,
        ),
        "timestamp": re.compile(
            r"(?:Time(?:stamp)?:\s*)?(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z?)",
            re.IGNORECASE,
        ),
        "followups": re.compile(
            r"Follow-?ups?:\s*([^\|]+?)(?:\s*\||$)",
            re.IGNORECASE,
        ),
        "risks": re.compile(
            r"(?:Accepted\s*)?Risks?:\s*([^\|]+?)(?:\s*\||$)",
            re.IGNORECASE,
        ),
    }

    def validate(self, evidence_text: str, allow_legacy_text: bool = False) -> dict[str, Any]:
        """Parse and validate evidence format.

        Args:
            evidence_text: Evidence string to validate
            allow_legacy_text: Whether to accept legacy non-JSON evidence

        Returns:
            Validation result with parsed data or error
        """
        if not evidence_text or not evidence_text.strip():
            return {
                "valid": False,
                "error": "Evidence is empty",
                "parsed": None,
            }

        stripped = evidence_text.strip()
        if stripped.startswith("{"):
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as e:
                return {
                    "valid": False,
                    "error": f"Invalid JSON evidence: {e}",
                    "parsed": None,
                }
            return self._validate_json(payload)

        if allow_legacy_text:
            return self._flexible_parse(evidence_text)

        return {
            "valid": False,
            "error": "Evidence must be JSON (see CAPTAIN_PROTOCOL.md)",
            "parsed": None,
        }

    def _validate_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Validate JSON evidence payload."""
        if not isinstance(payload, dict):
            return {
                "valid": False,
                "error": "Evidence JSON must be an object",
                "parsed": None,
            }

        missing = self.REQUIRED_JSON_FIELDS - payload.keys()
        if missing:
            return {
                "valid": False,
                "error": f"Missing required JSON fields: {sorted(missing)}",
                "parsed": None,
            }

        fmt = payload.get("format")
        if fmt not in {"standard", "legacy"}:
            return {
                "valid": False,
                "error": "format must be 'standard' or 'legacy'",
                "parsed": None,
            }

        commits = payload.get("commits")
        if not isinstance(commits, list):
            return {
                "valid": False,
                "error": "commits must be a list",
                "parsed": None,
            }

        if fmt == "standard":
            if not commits:
                return {
                    "valid": False,
                    "error": "standard evidence requires at least one commit",
                    "parsed": None,
                }
            for sha in commits:
                if not re.fullmatch(r"[a-f0-9]{7,40}", str(sha)):
                    return {
                        "valid": False,
                        "error": f"Invalid commit SHA: {sha}",
                        "parsed": None,
                    }

        tests = payload.get("tests")
        if not isinstance(tests, dict):
            return {
                "valid": False,
                "error": "tests must be an object",
                "parsed": None,
            }
        missing_tests = self.REQUIRED_TEST_FIELDS - tests.keys()
        if missing_tests:
            return {
                "valid": False,
                "error": f"Missing tests fields: {sorted(missing_tests)}",
                "parsed": None,
            }

        tests_command = tests.get("command")
        tests_status = tests.get("status")
        tests_passed = tests.get("passed")
        tests_total = tests.get("total")

        if tests_status not in self.VALID_STATUSES:
            return {
                "valid": False,
                "error": f"Invalid tests.status: {tests_status}",
                "parsed": None,
            }

        if fmt == "standard":
            if not isinstance(tests_command, str) or not tests_command.strip():
                return {
                    "valid": False,
                    "error": "standard evidence requires tests.command",
                    "parsed": None,
                }
            if not isinstance(tests_passed, int) or not isinstance(tests_total, int):
                return {
                    "valid": False,
                    "error": "standard evidence requires integer tests.passed/total",
                    "parsed": None,
                }
        else:
            if tests_command in ("", None):
                tests_command = None
            if tests_passed in ("", None):
                tests_passed = None
            if tests_total in ("", None):
                tests_total = None

        model = payload.get("models", "").strip()
        reviewer = payload.get("reviewer", "").strip()
        timestamp = payload.get("timestamp", "")

        if fmt == "standard":
            if not model or not reviewer:
                return {
                    "valid": False,
                    "error": "standard evidence requires models and reviewer",
                    "parsed": None,
                }
            ts_result = self.validate_timestamp(timestamp)
            if not ts_result["valid"]:
                return {
                    "valid": False,
                    "error": ts_result["error"],
                    "parsed": None,
                }

        parsed = ParsedEvidence(
            format=fmt,
            commits=[str(c) for c in commits],
            tests_command=tests_command,
            tests_status=tests_status,
            tests_passed=tests_passed,
            tests_total=tests_total,
            model=model or "UNKNOWN",
            reviewer=reviewer or "UNKNOWN",
            timestamp=timestamp or "UNKNOWN",
            followups=payload.get("followups"),
            accepted_risks=payload.get("accepted_risks"),
            legacy_summary=payload.get("legacy_summary"),
        )

        return {
            "valid": True,
            "parsed": parsed,
            "error": None,
            "format": "json",
        }

    def _flexible_parse(self, evidence_text: str) -> dict[str, Any]:
        """Try to parse evidence using flexible patterns.

        Args:
            evidence_text: Evidence string

        Returns:
            Validation result
        """
        extracted: dict[str, Any] = {}
        missing: list[str] = []

        # Extract tests
        tests_match = self.FLEXIBLE_PATTERNS["tests"].search(evidence_text)
        if tests_match:
            extracted["tests_passed"] = int(tests_match.group(1))
            extracted["tests_total"] = int(tests_match.group(2))
        else:
            missing.append("tests")

        # Extract commit
        commit_match = self.FLEXIBLE_PATTERNS["commit"].search(evidence_text)
        if commit_match:
            extracted["commit_sha"] = commit_match.group(1)
        else:
            missing.append("commit")

        # Extract model
        model_match = self.FLEXIBLE_PATTERNS["model"].search(evidence_text)
        if model_match:
            extracted["model"] = model_match.group(1).strip()
        else:
            missing.append("model")

        # Extract timestamp
        ts_match = self.FLEXIBLE_PATTERNS["timestamp"].search(evidence_text)
        if ts_match:
            extracted["timestamp"] = ts_match.group(1)
        else:
            missing.append("timestamp")

        # Optional: followups
        followups_match = self.FLEXIBLE_PATTERNS["followups"].search(evidence_text)
        if followups_match:
            extracted["followups"] = followups_match.group(1).strip()

        # Optional: risks
        risks_match = self.FLEXIBLE_PATTERNS["risks"].search(evidence_text)
        if risks_match:
            extracted["accepted_risks"] = risks_match.group(1).strip()

        # Check required fields
        if missing:
            return {
                "valid": False,
                "error": f"Missing required fields: {missing}",
                "parsed": None,
                "extracted": extracted,
                "missing": missing,
            }

        try:
            parsed = ParsedEvidence(
                format="legacy",
                commits=[extracted["commit_sha"]],
                tests_command=None,
                tests_status="UNKNOWN",
                tests_passed=extracted["tests_passed"],
                tests_total=extracted["tests_total"],
                model=extracted["model"],
                reviewer="UNKNOWN",
                timestamp=extracted["timestamp"],
                followups=extracted.get("followups"),
                accepted_risks=extracted.get("accepted_risks"),
                legacy_summary=evidence_text.strip(),
            )
            return {
                "valid": True,
                "parsed": parsed,
                "error": None,
                "format": "legacy_text",
            }
        except Exception as e:
            return {
                "valid": False,
                "error": f"Failed to create ParsedEvidence: {e}",
                "parsed": None,
            }

    def validate_timestamp(self, timestamp: str) -> dict[str, Any]:
        """Validate timestamp format and recency.

        Args:
            timestamp: ISO-8601 timestamp string

        Returns:
            Validation result
        """
        try:
            # Handle with or without Z suffix
            ts = timestamp.rstrip("Z")
            dt = datetime.fromisoformat(ts)

            # Check if timestamp is reasonable (not in future, not too old)
            now = datetime.now()
            age_hours = (now - dt).total_seconds() / 3600

            return {
                "valid": True,
                "datetime": dt,
                "age_hours": age_hours,
                "warning": "Timestamp is over 24 hours old" if age_hours > 24 else None,
            }
        except ValueError as e:
            return {
                "valid": False,
                "error": f"Invalid timestamp format: {e}",
            }

    def format_evidence_json(
        self,
        tests_passed: int,
        tests_total: int,
        commit_sha: str,
        model: str,
        reviewer: str,
        timestamp: str | None = None,
        followups: str = "None",
        accepted_risks: str = "None",
        tests_command: str = "pytest tests/ -q",
        status: str = "PASS",
        fmt: str = "standard",
    ) -> str:
        """Format evidence as machine-validated JSON.

        Args:
            tests_passed: Number of tests passed
            tests_total: Total number of tests
            commit_sha: Git commit SHA
            model: Model name
            reviewer: Reviewer string
            timestamp: Optional timestamp (defaults to now)
            followups: Follow-up items
            accepted_risks: Accepted risks
            tests_command: Test command executed
            status: Test status (PASS/FAIL/SKIP/UNKNOWN)
            fmt: Evidence format (standard/legacy)

        Returns:
            Formatted evidence string
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat() + "Z"

        payload = {
            "format": fmt,
            "commits": [commit_sha],
            "tests": {
                "command": tests_command,
                "status": status,
                "passed": tests_passed,
                "total": tests_total,
            },
            "models": model,
            "reviewer": reviewer,
            "timestamp": timestamp,
            "followups": followups,
            "accepted_risks": accepted_risks,
        }
        return json.dumps(payload, separators=(",", ":"))

    def format_evidence(
        self,
        tests_passed: int,
        tests_total: int,
        commit_sha: str,
        model: str,
        reviewer: str,
        timestamp: str | None = None,
        followups: str = "None",
        accepted_risks: str = "None",
        tests_command: str = "pytest tests/ -q",
        status: str = "PASS",
        fmt: str = "standard",
    ) -> str:
        """Backwards-compatible alias for JSON evidence formatting."""
        return self.format_evidence_json(
            tests_passed=tests_passed,
            tests_total=tests_total,
            commit_sha=commit_sha,
            model=model,
            reviewer=reviewer,
            timestamp=timestamp,
            followups=followups,
            accepted_risks=accepted_risks,
            tests_command=tests_command,
            status=status,
            fmt=fmt,
        )


def main() -> None:
    """CLI entry point."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: evidence_validator.py '<evidence_string>'")
        print(
            'Example: evidence_validator.py '
            '\'{"format":"standard","commits":["abc123"],'
            '"tests":{"command":"pytest tests/unit/core -q","status":"PASS","passed":47,"total":47},'
            '"models":"Claude","reviewer":"claude + captain","timestamp":"2025-12-13T10:00:00Z",'
            '"followups":"None","accepted_risks":"None"}\''
        )
        sys.exit(1)

    evidence = " ".join(sys.argv[1:])
    validator = EvidenceValidator()
    result = validator.validate(evidence)

    if result["valid"]:
        print("✅ Evidence is valid")
        parsed = result["parsed"]
        print(f"   Tests: {parsed.tests_passed}/{parsed.tests_total} ({parsed.tests_pass_rate:.1%})")
        print(f"   Commits: {', '.join(parsed.commits)}")
        print(f"   Model: {parsed.model}")
        print(f"   Reviewer: {parsed.reviewer}")
        print(f"   Timestamp: {parsed.timestamp}")
    else:
        print(f"❌ Evidence is invalid: {result['error']}")
        if "missing" in result:
            print(f"   Missing fields: {result['missing']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
