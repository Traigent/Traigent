#!/usr/bin/env python3
"""Safe tag migration script for Traigent traceability tags.

This script performs verified regex replacements for migrating deprecated tags
to the new taxonomy. It operates in dry-run mode by default.

Usage:
    python tools/migrate_tags.py --dry-run     # Preview changes
    python tools/migrate_tags.py --apply       # Apply changes
    python tools/migrate_tags.py --verify      # Verify current state
"""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from traigent.utils.secure_path import (
    PathTraversalError,
    safe_read_text,
    safe_write_text,
    validate_path,
)

@dataclass
class MigrationRule:
    """Defines a safe tag migration pattern."""

    name: str
    description: str
    # Pattern must match the ENTIRE traceability line for safety
    pattern: re.Pattern
    replacement: Callable[[re.Match], str]
    # Files this rule applies to (glob pattern)
    file_patterns: list[str]


# Define migration rules with explicit file targeting for safety
MIGRATION_RULES: list[MigrationRule] = [
    # === CrossCutting → Infra migrations (infrastructure utilities) ===
    MigrationRule(
        name="crosscutting-retry-to-infra",
        description="retry.py: CrossCutting → Infra (network retry is infrastructure)",
        pattern=re.compile(
            r"(# Traceability: )CONC-Layer-CrossCutting( CONC-Quality-Reliability CONC-Quality-Performance FUNC-CLOUD-HYBRID FUNC-SECURITY REQ-CLOUD-009 REQ-SEC-010 SYNC-CloudHybrid)"
        ),
        replacement=lambda m: f"{m.group(1)}CONC-Layer-Infra{m.group(2)}",
        file_patterns=["**/retry.py"],
    ),
    MigrationRule(
        name="crosscutting-logging-to-infra",
        description="logging.py: CrossCutting → Infra (logging infrastructure)",
        pattern=re.compile(
            r"(# Traceability: )CONC-Layer-CrossCutting( CONC-Quality-Observability CONC-Quality-Maintainability FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow)"
        ),
        replacement=lambda m: f"{m.group(1)}CONC-Layer-Infra{m.group(2)}",
        file_patterns=["**/utils/logging.py"],
    ),
    MigrationRule(
        name="crosscutting-error-handler-to-infra",
        description="error_handler.py: CrossCutting → Infra (error handling infrastructure)",
        pattern=re.compile(
            r"(# Traceability: )CONC-Layer-CrossCutting( CONC-Quality-Reliability CONC-Quality-Security FUNC-ORCH-LIFECYCLE FUNC-CLOUD-HYBRID FUNC-SECURITY REQ-ORCH-003 REQ-CLOUD-009 REQ-SEC-010 SYNC-OptimizationFlow SYNC-CloudHybrid)"
        ),
        replacement=lambda m: f"{m.group(1)}CONC-Layer-Infra{m.group(2)}",
        file_patterns=["**/error_handler.py"],
    ),
    MigrationRule(
        name="crosscutting-exceptions-to-infra",
        description="exceptions.py: CrossCutting → Infra (exception definitions)",
        pattern=re.compile(
            r"(# Traceability: )CONC-Layer-CrossCutting( CONC-Quality-Reliability CONC-Quality-Security FUNC-ORCH-LIFECYCLE FUNC-CLOUD-HYBRID FUNC-SECURITY REQ-ORCH-003 REQ-CLOUD-009 REQ-SEC-010 SYNC-OptimizationFlow SYNC-CloudHybrid)"
        ),
        replacement=lambda m: f"{m.group(1)}CONC-Layer-Infra{m.group(2)}",
        file_patterns=["**/exceptions.py"],
    ),
    MigrationRule(
        name="crosscutting-env-config-to-infra",
        description="env_config.py: CrossCutting → Infra (environment configuration)",
        pattern=re.compile(
            r"(# Traceability: )CONC-Layer-CrossCutting( CONC-Quality-Security CONC-Quality-Maintainability FUNC-INVOKERS FUNC-SECURITY REQ-INJ-002 REQ-SEC-010 SYNC-OptimizationFlow)"
        ),
        replacement=lambda m: f"{m.group(1)}CONC-Layer-Infra{m.group(2)}",
        file_patterns=["**/env_config.py"],
    ),
    MigrationRule(
        name="crosscutting-validation-to-infra",
        description="validation.py: CrossCutting → Infra (input validation infrastructure)",
        pattern=re.compile(
            r"(# Traceability: )CONC-Layer-CrossCutting( CONC-Quality-Maintainability CONC-Quality-Reliability FUNC-INVOKERS REQ-INJ-002 SYNC-OptimizationFlow)"
        ),
        replacement=lambda m: f"{m.group(1)}CONC-Layer-Infra{m.group(2)}",
        file_patterns=["**/validation.py"],
    ),
    MigrationRule(
        name="crosscutting-hashing-to-infra",
        description="hashing.py: CrossCutting → Infra (hashing utilities)",
        pattern=re.compile(
            r"(# Traceability: )CONC-Layer-CrossCutting( CONC-Quality-Reliability CONC-Quality-Maintainability FUNC-ANALYTICS FUNC-STORAGE REQ-ANLY-011 REQ-STOR-007 SYNC-StorageLogging)"
        ),
        replacement=lambda m: f"{m.group(1)}CONC-Layer-Infra{m.group(2)}",
        file_patterns=["**/hashing.py"],
    ),
    MigrationRule(
        name="crosscutting-reproducibility-to-infra",
        description="reproducibility.py: CrossCutting → Infra (reproducibility infrastructure)",
        pattern=re.compile(
            r"(# Traceability: )CONC-Layer-CrossCutting( CONC-Quality-Reliability CONC-Quality-Maintainability FUNC-STORAGE FUNC-ANALYTICS REQ-STOR-007 REQ-ANLY-011 SYNC-StorageLogging)"
        ),
        replacement=lambda m: f"{m.group(1)}CONC-Layer-Infra{m.group(2)}",
        file_patterns=["**/reproducibility.py"],
    ),
    # === CrossCutting → Core migrations (domain-specific utilities) ===
    MigrationRule(
        name="crosscutting-optimization-logger-to-core",
        description="optimization_logger.py: CrossCutting → Core (domain-specific optimization logging)",
        pattern=re.compile(
            r"(# Traceability: )CONC-Layer-CrossCutting( CONC-Quality-Observability CONC-Quality-Security FUNC-ORCH-LIFECYCLE FUNC-STORAGE FUNC-ANALYTICS REQ-ORCH-003 REQ-STOR-007 REQ-ANLY-011 SYNC-OptimizationFlow SYNC-StorageLogging)"
        ),
        replacement=lambda m: f"{m.group(1)}CONC-Layer-Core{m.group(2)}",
        file_patterns=["**/optimization_logger.py"],
    ),
    MigrationRule(
        name="crosscutting-local-analytics-to-core",
        description="local_analytics.py: CrossCutting → Core (analytics is domain logic)",
        pattern=re.compile(
            r"(# Traceability: )CONC-Layer-CrossCutting( CONC-Quality-Observability CONC-Quality-Performance FUNC-ANALYTICS REQ-ANLY-011 SYNC-Observability)"
        ),
        replacement=lambda m: f"{m.group(1)}CONC-Layer-Core{m.group(2)}",
        file_patterns=["**/local_analytics.py"],
    ),
    MigrationRule(
        name="crosscutting-cost-calculator-to-core",
        description="cost_calculator.py: CrossCutting → Core (cost calculation is domain logic)",
        pattern=re.compile(
            r"(# Traceability: )CONC-Layer-CrossCutting( CONC-Quality-Performance CONC-Quality-Observability FUNC-ANALYTICS REQ-ANLY-011 SYNC-Observability)"
        ),
        replacement=lambda m: f"{m.group(1)}CONC-Layer-Core{m.group(2)}",
        file_patterns=["**/cost_calculator.py"],
    ),
    MigrationRule(
        name="crosscutting-diagnostics-to-core",
        description="diagnostics.py: CrossCutting → Core (diagnostics is domain logic)",
        pattern=re.compile(
            r"(# Traceability: )CONC-Layer-CrossCutting( CONC-Quality-Observability CONC-Quality-Maintainability FUNC-ANALYTICS REQ-ANLY-011 SYNC-Observability)"
        ),
        replacement=lambda m: f"{m.group(1)}CONC-Layer-Core{m.group(2)}",
        file_patterns=["**/diagnostics.py"],
    ),
    MigrationRule(
        name="crosscutting-optuna-metrics-to-core",
        description="optuna_metrics.py: CrossCutting → Core (telemetry is domain observability)",
        pattern=re.compile(
            r"(# Traceability: )CONC-Layer-CrossCutting( CONC-Quality-Observability CONC-Quality-Reliability FUNC-ANALYTICS REQ-ANLY-011 SYNC-Observability)"
        ),
        replacement=lambda m: f"{m.group(1)}CONC-Layer-Core{m.group(2)}",
        file_patterns=["**/optuna_metrics.py"],
    ),
    # === CrossCutting → Data migrations (version/schema) ===
    MigrationRule(
        name="crosscutting-version-to-data",
        description="_version.py: CrossCutting → Data (version is metadata/schema)",
        pattern=re.compile(
            r"(# Traceability: )CONC-Layer-CrossCutting( CONC-Quality-Maintainability FUNC-API-ENTRY REQ-API-001)"
        ),
        replacement=lambda m: f"{m.group(1)}CONC-Layer-Data{m.group(2)}",
        file_patterns=["**/_version.py"],
    ),
    # === CrossCutting → Infra for utils/__init__.py (export aggregator) ===
    MigrationRule(
        name="crosscutting-utils-init-to-infra",
        description="utils/__init__.py: CrossCutting → Infra (module export aggregator)",
        pattern=re.compile(
            r"(# Traceability: )CONC-Layer-CrossCutting( CONC-Quality-Maintainability CONC-Quality-Observability FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow)"
        ),
        replacement=lambda m: f"{m.group(1)}CONC-Layer-Infra{m.group(2)}",
        file_patterns=["**/utils/__init__.py"],
    ),
    # === Experimental → Integration migrations (platform adapters) ===
    MigrationRule(
        name="experimental-platform-to-integration",
        description="experimental/platforms/*: Experimental → Integration (LLM provider adapters)",
        pattern=re.compile(
            r"(# Traceability: )CONC-Layer-Experimental( CONC-Quality-Compatibility FUNC-AGENTS FUNC-INTEGRATIONS REQ-AGNT-013 REQ-INT-008 SYNC-OptimizationFlow)"
        ),
        replacement=lambda m: f"{m.group(1)}CONC-Layer-Integration{m.group(2)}",
        file_patterns=["**/experimental/simple_cloud/platforms/*.py"],
    ),
    # === Experimental → Core migrations (simulator) ===
    MigrationRule(
        name="experimental-simulator-to-core",
        description="experimental/simulator.py: Experimental → Core (simulation is domain logic)",
        pattern=re.compile(
            r"(# Traceability: )CONC-Layer-Experimental( CONC-Quality-Reliability FUNC-CLOUD-HYBRID REQ-CLOUD-009 SYNC-CloudHybrid)"
        ),
        replacement=lambda m: f"{m.group(1)}CONC-Layer-Core{m.group(2)}",
        file_patterns=["**/simulator.py", "**/experimental/__init__.py"],
    ),
    # === Reduce 3 quality tags to 2 ===
    MigrationRule(
        name="reduce-quality-deployment",
        description="security/deployment.py: Remove Observability (Security+Reliability are primary)",
        pattern=re.compile(
            r"(# Traceability: CONC-Layer-Infra )CONC-Quality-Security CONC-Quality-Reliability CONC-Quality-Observability( FUNC-SECURITY REQ-SEC-010 SYNC-CloudHybrid)"
        ),
        replacement=lambda m: f"{m.group(1)}CONC-Quality-Security CONC-Quality-Reliability{m.group(2)}",
        file_patterns=["**/deployment.py"],
    ),
]


def find_matching_files(base_path: Path, patterns: list[str]) -> list[Path]:
    """Find all files matching any of the glob patterns."""
    files = []
    for pattern in patterns:
        files.extend(base_path.glob(pattern))
    return [f for f in files if f.is_file() and "__pycache__" not in str(f)]


def apply_rule(
    file_path: Path,
    rule: MigrationRule,
    base_dir: Path,
    dry_run: bool = True,
) -> tuple[bool, str]:
    """Apply a migration rule to a file.

    Returns:
        (changed, message) tuple
    """
    try:
        safe_path = validate_path(file_path, base_dir, must_exist=True)
        content = safe_read_text(safe_path, base_dir, encoding="utf-8")
    except (PathTraversalError, FileNotFoundError, OSError) as e:
        return False, f"Error reading {file_path}: {e}"

    # Check if pattern matches
    match = rule.pattern.search(content)
    if not match:
        return False, ""

    # Perform replacement
    new_content = rule.pattern.sub(rule.replacement, content)

    if new_content == content:
        return False, ""

    # Verify the replacement is valid
    old_line = match.group(0)
    new_match = rule.pattern.search(new_content)
    if new_match:
        return False, f"ERROR: Pattern still matches after replacement in {file_path}"

    # Extract new traceability line for verification
    new_trace_match = re.search(r"# Traceability:.+", new_content)
    new_line = new_trace_match.group(0) if new_trace_match else "UNKNOWN"

    if dry_run:
        return True, f"[DRY-RUN] {safe_path}\n  OLD: {old_line}\n  NEW: {new_line}"
    else:
        safe_write_text(safe_path, new_content, base_dir, encoding="utf-8")
        return True, f"[APPLIED] {safe_path}\n  OLD: {old_line}\n  NEW: {new_line}"


def run_migrations(base_paths: list[Path], dry_run: bool = True) -> dict:
    """Run all migration rules across the codebase."""
    results = {
        "changed": [],
        "errors": [],
        "skipped": 0,
    }

    for rule in MIGRATION_RULES:
        print(f"\n=== Rule: {rule.name} ===")
        print(f"    {rule.description}")

        for base_path in base_paths:
            files = find_matching_files(base_path, rule.file_patterns)
            for file_path in files:
                changed, message = apply_rule(file_path, rule, base_path, dry_run)
                if message:
                    if "ERROR" in message:
                        results["errors"].append(message)
                        print(f"  ❌ {message}")
                    elif changed:
                        results["changed"].append(str(file_path))
                        print(f"  ✓ {message}")
                else:
                    results["skipped"] += 1

    return results


def verify_state(base_paths: list[Path]) -> dict:
    """Verify the current state of tags against taxonomy."""
    issues = {
        "deprecated_crosscutting": [],
        "deprecated_experimental": [],
        "too_many_quality": [],
        "missing_layer": [],
    }

    for base_path in base_paths:
        for file_path in base_path.rglob("*.py"):
            if "__pycache__" in str(file_path):
                continue
            try:
                safe_path = validate_path(file_path, base_path, must_exist=True)
                content = safe_path.read_text(encoding="utf-8")
            except (PathTraversalError, FileNotFoundError, OSError):
                continue

            trace_match = re.search(r"# Traceability:(.+)", content)
            if not trace_match:
                continue

            line = trace_match.group(1)

            if "CONC-Layer-CrossCutting" in line:
                issues["deprecated_crosscutting"].append(str(safe_path))
            if "CONC-Layer-Experimental" in line:
                issues["deprecated_experimental"].append(str(safe_path))

            quality_tags = re.findall(r"CONC-Quality-\w+", line)
            if len(quality_tags) > 2:
                issues["too_many_quality"].append((str(safe_path), quality_tags))

            layer_tags = re.findall(r"CONC-Layer-\w+", line)
            if len(layer_tags) != 1:
                issues["missing_layer"].append((str(safe_path), layer_tags))

    return issues


def main():
    parser = argparse.ArgumentParser(description="Safe tag migration for Traigent")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without applying"
    )
    parser.add_argument("--apply", action="store_true", help="Apply changes")
    parser.add_argument("--verify", action="store_true", help="Verify current state")
    parser.add_argument("--base", type=str, default=".", help="Base path to search")
    args = parser.parse_args()

    base = Path(args.base)
    try:
        base = validate_path(base, Path.cwd(), must_exist=True)
    except (PathTraversalError, FileNotFoundError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    base_paths = [base / "traigent", base / "src"]
    base_paths = [p for p in base_paths if p.exists()]

    if not base_paths:
        print(f"Error: No traigent/ or src/ found under {base}")
        sys.exit(1)

    if args.verify:
        print("=== Verifying current tag state ===")
        issues = verify_state(base_paths)

        print(
            f"\nDeprecated CONC-Layer-CrossCutting: {len(issues['deprecated_crosscutting'])}"
        )
        for f in issues["deprecated_crosscutting"]:
            print(f"  - {f}")

        print(
            f"\nDeprecated CONC-Layer-Experimental: {len(issues['deprecated_experimental'])}"
        )
        for f in issues["deprecated_experimental"]:
            print(f"  - {f}")

        print(f"\nFiles with 3+ quality tags: {len(issues['too_many_quality'])}")
        for f, tags in issues["too_many_quality"]:
            print(f"  - {f}: {tags}")

        total_issues = (
            len(issues["deprecated_crosscutting"])
            + len(issues["deprecated_experimental"])
            + len(issues["too_many_quality"])
        )
        print(f"\n{'❌' if total_issues else '✓'} Total issues: {total_issues}")
        sys.exit(1 if total_issues else 0)

    elif args.apply:
        print("=== Applying migrations ===")
        results = run_migrations(base_paths, dry_run=False)
        print(f"\n✓ Changed: {len(results['changed'])} files")
        print(f"  Skipped: {results['skipped']} (no match)")
        if results["errors"]:
            print(f"❌ Errors: {len(results['errors'])}")
            sys.exit(1)

    else:  # Default to dry-run
        print("=== Dry-run mode (use --apply to make changes) ===")
        results = run_migrations(base_paths, dry_run=True)
        print(f"\n✓ Would change: {len(results['changed'])} files")
        print(f"  Would skip: {results['skipped']} (no match)")
        if results["errors"]:
            print(f"❌ Errors: {len(results['errors'])}")


if __name__ == "__main__":
    main()
