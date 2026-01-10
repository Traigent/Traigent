#!/usr/bin/env python3
"""Generate fresh pre-release review tracking file.

This script scans traigent/ and tests/ directories to create a new tracking
file with accurate file counts, SHA256 hashes, and fresh evidence placeholders.

Usage:
    python generate_tracking.py --version v0.10.0
    python generate_tracking.py --version v0.10.0 --output custom_path.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


# Priority scoring weights (same as protocol)
WEIGHT_CENTRALITY = 0.40
WEIGHT_SEVERITY = 0.35
WEIGHT_LIKELIHOOD = 0.25

# Default L/S/C scores for modules (can be overridden)
MODULE_SCORES: dict[str, tuple[int, int, int]] = {
    # (Likelihood, Severity, Centrality)
    "integrations": (5, 5, 5),
    "config": (4, 5, 5),
    "core": (4, 5, 5),
    "optimizers": (4, 5, 5),
    "invokers": (4, 5, 5),
    "storage": (4, 4, 5),
    "utils": (4, 4, 5),
    "evaluators": (4, 4, 5),
    "api": (3, 4, 5),
    "security": (4, 5, 3),
    "metrics": (3, 4, 4),
    "cli": (3, 3, 4),
    "cloud": (4, 3, 3),
    "agents": (3, 3, 3),
    "adapters": (2, 3, 3),
    "hooks": (3, 3, 2),
    "analytics": (3, 2, 2),
    "telemetry": (2, 2, 2),
    "visualization": (2, 2, 2),
    "tvl": (2, 2, 2),
    "plugins": (2, 2, 2),
    "experimental": (3, 2, 1),
}


def compute_priority(l_score: int, s_score: int, c_score: int) -> int:
    """Compute priority score (0-100) from L/S/C scores."""
    raw = WEIGHT_CENTRALITY * c_score + WEIGHT_SEVERITY * s_score + WEIGHT_LIKELIHOOD * l_score
    return round(raw * 20)


def get_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()[:12]  # First 12 chars for brevity
    except (OSError, IOError):
        return "ERROR"


def count_lines(filepath: Path) -> int:
    """Count lines in a file."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)
    except (OSError, IOError):
        return 0


def get_git_info() -> tuple[str, str]:
    """Get current git commit SHA and branch."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return sha, branch
    except subprocess.CalledProcessError:
        return "UNKNOWN", "UNKNOWN"


def get_last_modified(filepath: Path) -> str:
    """Get last modification date of a file."""
    try:
        mtime = filepath.stat().st_mtime
        return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
    except OSError:
        return "UNKNOWN"


def scan_module(module_path: Path) -> dict:
    """Scan a module directory and collect stats."""
    files = list(module_path.rglob("*.py"))
    total_lines = sum(count_lines(f) for f in files)

    # Compute combined hash of all files
    combined = hashlib.sha256()
    for f in sorted(files):
        combined.update(get_file_hash(f).encode())

    # Get most recent modification
    if files:
        last_mod = max(get_last_modified(f) for f in files)
    else:
        last_mod = "N/A"

    return {
        "path": str(module_path),
        "file_count": len(files),
        "line_count": total_lines,
        "last_modified": last_mod,
        "combined_hash": combined.hexdigest()[:12],
        "files": [str(f.relative_to(module_path.parent.parent)) for f in files],
    }


def scan_tests(tests_path: Path, module_name: str) -> dict:
    """Scan test files for a specific module."""
    # Look for tests in multiple locations
    test_patterns = [
        tests_path / "unit" / module_name,
        tests_path / "unit" / f"test_{module_name}.py",
        tests_path / "integration" / f"test_{module_name}*.py",
    ]

    test_files = []
    for pattern in test_patterns:
        if pattern.is_dir():
            test_files.extend(pattern.rglob("test_*.py"))
        elif pattern.exists():
            test_files.append(pattern)
        elif "*" in str(pattern):
            test_files.extend(pattern.parent.glob(pattern.name))

    # Count test functions (approximate)
    test_count = 0
    for tf in test_files:
        try:
            content = tf.read_text(encoding="utf-8", errors="ignore")
            test_count += content.count("def test_")
            test_count += content.count("async def test_")
        except (OSError, IOError):
            pass

    return {
        "test_files": [str(f) for f in test_files],
        "test_count": test_count,
    }


def generate_evidence_placeholder(generated_time: str) -> str:
    """Generate a fresh evidence JSON placeholder."""
    evidence = {
        "format": "standard",
        "generated": generated_time,
        "commits": [],
        "tests": {
            "command": None,
            "status": "NOT_RUN",
            "passed": None,
            "total": None,
        },
        "models": None,
        "reviewer": None,
        "timestamp": None,
        "followups": None,
        "accepted_risks": None,
    }
    return json.dumps(evidence)


def generate_tracking_file(
    version: str,
    repo_root: Path,
    output_path: Path | None = None,
) -> str:
    """Generate a fresh tracking file."""
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    date_str = now.strftime("%Y%m%d")

    git_sha, git_branch = get_git_info()

    traigent_path = repo_root / "traigent"
    tests_path = repo_root / "tests"

    # Scan all modules
    modules = {}
    for module_dir in sorted(traigent_path.iterdir()):
        if module_dir.is_dir() and not module_dir.name.startswith("_"):
            module_name = module_dir.name
            modules[module_name] = scan_module(module_dir)
            modules[module_name]["tests"] = scan_tests(tests_path, module_name)

    # Also scan root-level files
    root_files = list(traigent_path.glob("*.py"))

    # Build the markdown content
    lines = []

    # Header
    lines.append(f"# Pre-Release Review Tracking (Traigent SDK {version})")
    lines.append("")
    lines.append(f"**Generated**: {timestamp}")
    lines.append(f"**Generator**: generate_tracking.py v1.0")
    lines.append(f"**Baseline commit**: {git_sha} ({git_branch})")
    lines.append(f"**Total modules**: {len(modules)}")
    lines.append(f"**Total files**: {sum(m['file_count'] for m in modules.values()) + len(root_files)}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Roles section
    lines.append("## Roles")
    lines.append("")
    lines.append("- Release captain: Claude Code (Opus 4.5)")
    lines.append("- Human release owner (final sign-off): TBD")
    lines.append("- Target release date: TBD")
    lines.append(f"- Branch/tag: `release-review/{version}` (baseline: `{version}-rc1` @ {git_sha})")
    lines.append(f"- Tracking file: `PRE_RELEASE_REVIEW_TRACKING_{version}_{date_str}.md`")
    lines.append("")

    # File Manifest
    lines.append("## File Manifest (traigent/)")
    lines.append("")
    lines.append("| Module | Files | Lines | Last Modified | Hash |")
    lines.append("|--------|-------|-------|---------------|------|")

    for name, info in sorted(modules.items()):
        lines.append(
            f"| `traigent/{name}/` | {info['file_count']} | {info['line_count']:,} | "
            f"{info['last_modified']} | `{info['combined_hash']}` |"
        )

    lines.append("")

    # Test Coverage Mapping
    lines.append("## Test Coverage Mapping")
    lines.append("")
    lines.append("| Component | Test Files | Test Functions |")
    lines.append("|-----------|------------|----------------|")

    for name, info in sorted(modules.items()):
        test_info = info.get("tests", {})
        test_file_count = len(test_info.get("test_files", []))
        test_count = test_info.get("test_count", 0)
        lines.append(f"| `traigent/{name}/` | {test_file_count} | {test_count} |")

    lines.append("")

    # SDK Runtime Components
    lines.append("## SDK Runtime Components")
    lines.append("")
    lines.append("| Component | Priority | L/S/C | Files | Scope | Status | Evidence |")
    lines.append("|-----------|----------|-------|-------|-------|--------|----------|")

    # Sort by priority (descending)
    sorted_modules = sorted(
        modules.items(),
        key=lambda x: compute_priority(*MODULE_SCORES.get(x[0], (2, 2, 2))),
        reverse=True,
    )

    for name, info in sorted_modules:
        l, s, c = MODULE_SCORES.get(name, (2, 2, 2))
        priority = compute_priority(l, s, c)
        evidence = generate_evidence_placeholder(timestamp)
        lines.append(
            f"| {name.capitalize()} | {priority} | {l}/{s}/{c} | {info['file_count']} | "
            f"`traigent/{name}/` | **Not started** | {evidence} |"
        )

    lines.append("")

    # Root-level files
    if root_files:
        lines.append("## Root-Level Files (traigent/*.py)")
        lines.append("")
        lines.append("| File | Lines | Hash |")
        lines.append("|------|-------|------|")
        for f in sorted(root_files):
            lines.append(f"| `{f.name}` | {count_lines(f)} | `{get_file_hash(f)}` |")
        lines.append("")

    # Review Notes Log
    lines.append("## Review Notes Log (append-only)")
    lines.append("")
    lines.append(f"### {version} Review - NOT STARTED")
    lines.append("")
    lines.append(f"- {timestamp}: **Tracking file generated** — Fresh tracking file created with file manifest.")
    lines.append("")

    content = "\n".join(lines)

    # Write to file if output path provided
    if output_path:
        output_path.write_text(content, encoding="utf-8")
        print(f"Generated: {output_path}")

    return content


def main():
    parser = argparse.ArgumentParser(
        description="Generate fresh pre-release review tracking file"
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Release version (e.g., v0.10.0)",
    )
    parser.add_argument(
        "--output",
        help="Output file path (default: auto-generated with timestamp)",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root directory",
    )

    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()

    # Generate output path if not specified
    if args.output:
        output_path = Path(args.output)
    else:
        date_str = datetime.now().strftime("%Y%m%d")
        output_path = (
            repo_root / ".release_review" /
            f"PRE_RELEASE_REVIEW_TRACKING_{args.version}_{date_str}.md"
        )

    generate_tracking_file(args.version, repo_root, output_path)

    # Print summary
    print(f"\nTo activate this tracking file:")
    print(f"  cd {repo_root / '.release_review'}")
    print(f"  ln -sf {output_path.name} PRE_RELEASE_REVIEW_TRACKING.md")


if __name__ == "__main__":
    main()
