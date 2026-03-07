#!/usr/bin/env python3
"""Validate release-review docs for deprecated path/command usage."""

from __future__ import annotations

import re
from pathlib import Path

TARGET_FILES = [
    Path(".release_review/CAPTAIN_PROTOCOL.md"),
    Path(".release_review/PRE_RELEASE_REVIEW_PLAN.md"),
    Path(".release_review/START_REVIEW.md"),
    Path(".release_review/PRE_RELEASE_REVIEW_TRACKING.md"),
    Path(".release_review/RELEASE_QUALITY_PLAYBOOK.md"),
    Path(".release_review/automation/README.md"),
]

REQUIRED_PROMPT_FILES = [
    Path(".release_review/prompts/codex_cli__captain.md"),
    Path(".release_review/prompts/codex_cli__primary.md"),
    Path(".release_review/prompts/claude_cli__secondary.md"),
    Path(".release_review/prompts/codex_cli__tertiary.md"),
    Path(".release_review/prompts/copilot_cli__tertiary.md"),
    Path(".release_review/prompts/codex_cli__reconciliation.md"),
]

REQUIRED_PROMPT_TOKENS = {
    Path(".release_review/prompts/codex_cli__captain.md"): [
        "strengths",
        "checks_performed",
        "review_summary",
    ],
    Path(".release_review/prompts/codex_cli__primary.md"): [
        "strengths",
        "checks_performed",
        "review_summary",
    ],
    Path(".release_review/prompts/claude_cli__secondary.md"): [
        "strengths",
        "checks_performed",
        "review_summary",
    ],
    Path(".release_review/prompts/codex_cli__tertiary.md"): [
        "strengths",
        "checks_performed",
        "review_summary",
    ],
    Path(".release_review/prompts/copilot_cli__tertiary.md"): [
        "strengths",
        "checks_performed",
        "review_summary",
    ],
    Path(".release_review/prompts/codex_cli__reconciliation.md"): [
        "strengths",
        "checks_performed",
        "consensus",
    ],
}

DEPRECATED_PATTERNS = [
    (
        re.compile(r"\.release_review/artifacts/"),
        "use `.release_review/runs/<release_id>/...` instead of `.release_review/artifacts/`",
    ),
    (
        re.compile(r"\bpython\s+\.release_review/"),
        "use `python3` instead of `python` for release-review commands",
    ),
    (
        re.compile(r"\.release_review/prompts/(captain_system|primary_reviewer|secondary_adversarial_reviewer|tertiary_independent_reviewer|reconciliation)\.md"),
        "use agent+review_type prompt matrix files (`<agent_type>__<review_type>.md`)",
    ),
]


def main() -> int:
    violations: list[str] = []

    for file in TARGET_FILES:
        if not file.exists():
            violations.append(f"missing required file: {file}")
            continue
        content = file.read_text()
        for pattern, message in DEPRECATED_PATTERNS:
            for match in pattern.finditer(content):
                line = content.count("\n", 0, match.start()) + 1
                violations.append(f"{file}:{line}: {message}")

    for prompt_file in REQUIRED_PROMPT_FILES:
        if not prompt_file.exists():
            violations.append(f"missing required prompt file: {prompt_file}")
            continue
        content = prompt_file.read_text()
        if not content.strip():
            violations.append(f"required prompt file is empty: {prompt_file}")
            continue
        lowered = content.lower()
        for token in REQUIRED_PROMPT_TOKENS.get(prompt_file, []):
            if token not in lowered:
                violations.append(
                    f"{prompt_file}: missing required prompt token '{token}'"
                )

    if violations:
        print("❌ Release-review consistency check failed")
        for item in violations:
            print(f"  - {item}")
        return 1

    print("✅ Release-review consistency check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
