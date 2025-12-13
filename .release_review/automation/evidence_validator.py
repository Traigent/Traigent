#!/usr/bin/env python3
"""Evidence Validator for Release Review Protocol.

Parses and validates evidence format in review tracking.
Used by captain to ensure all evidence meets standards.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class ParsedEvidence:
    """Parsed evidence data."""

    tests_passed: int
    tests_total: int
    commit_sha: str
    model: str
    timestamp: str
    followups: str | None = None
    accepted_risks: str | None = None

    @property
    def tests_pass_rate(self) -> float:
        """Calculate test pass rate."""
        if self.tests_total == 0:
            return 0.0
        return self.tests_passed / self.tests_total


class EvidenceValidator:
    """Validate evidence format and content."""

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

    def validate(self, evidence_text: str) -> dict[str, Any]:
        """Parse and validate evidence format.

        Args:
            evidence_text: Evidence string to validate

        Returns:
            Validation result with parsed data or error
        """
        if not evidence_text or not evidence_text.strip():
            return {
                "valid": False,
                "error": "Evidence is empty",
                "parsed": None,
            }

        # Try standard pattern first
        match = self.STANDARD_PATTERN.search(evidence_text)
        if match:
            try:
                parsed = ParsedEvidence(
                    tests_passed=int(match.group(1)),
                    tests_total=int(match.group(2)),
                    commit_sha=match.group(3),
                    model=match.group(4).strip(),
                    timestamp=match.group(5),
                )
                return {
                    "valid": True,
                    "parsed": parsed,
                    "error": None,
                    "format": "standard",
                }
            except (ValueError, IndexError) as e:
                return {
                    "valid": False,
                    "error": f"Failed to parse matched evidence: {e}",
                    "parsed": None,
                }

        # Try flexible parsing
        result = self._flexible_parse(evidence_text)
        return result

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
                tests_passed=extracted["tests_passed"],
                tests_total=extracted["tests_total"],
                commit_sha=extracted["commit_sha"],
                model=extracted["model"],
                timestamp=extracted["timestamp"],
                followups=extracted.get("followups"),
                accepted_risks=extracted.get("accepted_risks"),
            )
            return {
                "valid": True,
                "parsed": parsed,
                "error": None,
                "format": "flexible",
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

    def format_evidence(
        self,
        tests_passed: int,
        tests_total: int,
        commit_sha: str,
        model: str,
        timestamp: str | None = None,
        followups: str = "None",
        accepted_risks: str = "None",
    ) -> str:
        """Format evidence in standard format.

        Args:
            tests_passed: Number of tests passed
            tests_total: Total number of tests
            commit_sha: Git commit SHA
            model: Model name
            timestamp: Optional timestamp (defaults to now)
            followups: Follow-up items
            accepted_risks: Accepted risks

        Returns:
            Formatted evidence string
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat() + "Z"

        return (
            f"Tests: {tests_passed}/{tests_total} passed | "
            f"Commits: {commit_sha} | "
            f"Model: {model} | "
            f"Time: {timestamp} | "
            f"Follow-ups: {followups} | "
            f"Accepted risks: {accepted_risks}"
        )


def main() -> None:
    """CLI entry point."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: evidence_validator.py '<evidence_string>'")
        print('Example: evidence_validator.py "Tests: 47/47 | Commits: abc123 | Model: Claude | Time: 2025-12-13T10:00:00Z"')
        sys.exit(1)

    evidence = " ".join(sys.argv[1:])
    validator = EvidenceValidator()
    result = validator.validate(evidence)

    if result["valid"]:
        print("✅ Evidence is valid")
        parsed = result["parsed"]
        print(f"   Tests: {parsed.tests_passed}/{parsed.tests_total} ({parsed.tests_pass_rate:.1%})")
        print(f"   Commit: {parsed.commit_sha}")
        print(f"   Model: {parsed.model}")
        print(f"   Timestamp: {parsed.timestamp}")
    else:
        print(f"❌ Evidence is invalid: {result['error']}")
        if "missing" in result:
            print(f"   Missing fields: {result['missing']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
