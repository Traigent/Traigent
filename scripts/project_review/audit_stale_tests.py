#!/usr/bin/env python3
"""Build a stale-test inventory and Sonnet-ready review batches.

This script inventories pytest test files under ``tests/``, scores likely stale
or quarantined files using static heuristics, and emits:

- a machine-readable inventory (JSON/CSV)
- a markdown summary of high-signal candidates
- Sonnet-ready candidate batches for deeper human-in-the-loop review

The goal is to narrow 13k+ collected tests down to a small file-level review
set. It is intentionally conservative: "delete" recommendations are never
automatic. Files are classified into review buckets such as:

- ``move-out-of-tests``
- ``merge-or-delete-after-review``
- ``review-quarantine``
- ``keep-quarantined-runtime``
- ``keep-track-debt``
- ``keep``
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import tomllib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

MANUAL_DIR = Path("tests/manual_validation")
REFERENCE_GLOBS = {
    "workflows": [".github/workflows/**/*.yml", ".github/workflows/**/*.yaml"],
    "docs": ["docs/**/*.md", "README.md", "CLAUDE.md", "AGENTS.md"],
    "scripts": ["scripts/**/*.py", "scripts/**/*.md"],
    "config": ["pyproject.toml"],
}
RUNTIME_QUARANTINE_HINTS = (
    "tests/unit/bridges/",
    "tests/unit/test_bridge_wrapper.py",
    "tests/unit/evaluators/test_js_evaluator.py",
    "tests/unit/evaluators/test_js_evaluator_budget.py",
    "tests/unit/evaluators/test_js_evaluator_stop_conditions.py",
    "tests/unit/core/test_constraints_enforced.py",
    "tests/unit/evaluators/test_litellm_integration.py",
    "tests/unit/core/test_orchestrator.py",
)
PATTERN_MAP = {
    "localhost": re.compile(r"localhost|127\.0\.0\.1", re.IGNORECASE),
    "local_backend": re.compile(r"backend server.*localhost", re.IGNORECASE),
    "manual_gate": re.compile(r"RUN_MANUAL_VALIDATION|manual validation", re.IGNORECASE),
    "needs_update": re.compile(r"needs update", re.IGNORECASE),
    "not_implemented": re.compile(r"not yet implemented|feature needed", re.IGNORECASE),
    "main_guard": re.compile(r"if __name__ == ['\"]__main__['\"]"),
}


@dataclass(slots=True)
class InventoryEntry:
    path: str
    top_level: str
    line_count: int
    file_bytes: int
    test_function_count: int
    test_class_count: int
    fixture_count: int
    import_count: int
    skip_count: int
    skipif_count: int
    importorskip_count: int
    xfail_count: int
    todo_count: int
    print_count: int
    has_shebang: bool
    has_main_guard: bool
    has_localhost: bool
    has_manual_gate: bool
    has_not_implemented: bool
    has_needs_update: bool
    ignored_by_default: bool
    ignore_pattern: str
    workflow_ref_count: int
    docs_ref_count: int
    scripts_ref_count: int
    config_ref_count: int
    same_basename_count: int
    same_basename_other_paths: list[str] = field(default_factory=list)
    import_roots: list[str] = field(default_factory=list)
    evidence_snippets: list[str] = field(default_factory=list)
    stale_score: int = 0
    recommendation: str = "keep"
    reasons: list[str] = field(default_factory=list)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_pytest_ignores(root: Path) -> list[str]:
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    addopts = pyproject["tool"]["pytest"]["ini_options"]["addopts"]
    ignores: list[str] = []
    for item in addopts:
        if isinstance(item, str) and item.startswith("--ignore="):
            ignores.append(item.removeprefix("--ignore=").strip("/"))
    return ignores


def iter_reference_texts(root: Path) -> dict[str, list[str]]:
    refs: dict[str, list[str]] = {}
    for label, patterns in REFERENCE_GLOBS.items():
        blobs: list[str] = []
        for pattern in patterns:
            for path in sorted(root.glob(pattern)):
                if path.is_file():
                    blobs.append(path.read_text(encoding="utf-8", errors="replace"))
        refs[label] = blobs
    return refs


def path_is_ignored(rel_path: str, ignore_paths: list[str]) -> tuple[bool, str]:
    normalized = rel_path.strip("/")
    for ignore in ignore_paths:
        ignore_norm = ignore.strip("/")
        if normalized == ignore_norm or normalized.startswith(ignore_norm + "/"):
            return True, ignore_norm
    return False, ""


def find_reference_count(rel_path: str, blobs: Iterable[str]) -> int:
    patterns = (rel_path, f"./{rel_path}")
    count = 0
    for text in blobs:
        if any(pattern in text for pattern in patterns):
            count += 1
    return count


def same_basename_paths(root: Path) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = defaultdict(list)
    for path in root.rglob("*.py"):
        if path.is_file() and ".git" not in path.parts:
            mapping[path.name].append(str(path.relative_to(root)))
    return mapping


def count_calls(tree: ast.AST, module: str, name: str) -> int:
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Name) and func.value.id == module and func.attr == name:
                count += 1
        elif isinstance(func, ast.Name) and func.id == name:
            count += 1
    return count


def import_roots(tree: ast.AST) -> list[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return sorted(roots)


def test_class_count(tree: ast.AST) -> int:
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            count += 1
    return count


def test_function_count(tree: ast.AST) -> int:
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
            "test_"
        ):
            count += 1
    return count


def fixture_count(tree: ast.AST) -> int:
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for deco in node.decorator_list:
            if isinstance(deco, ast.Name) and deco.id == "fixture":
                count += 1
            elif isinstance(deco, ast.Attribute):
                if isinstance(deco.value, ast.Name) and deco.value.id == "pytest" and deco.attr == "fixture":
                    count += 1
            elif isinstance(deco, ast.Call):
                func = deco.func
                if isinstance(func, ast.Attribute):
                    if isinstance(func.value, ast.Name) and func.value.id == "pytest" and func.attr == "fixture":
                        count += 1
                elif isinstance(func, ast.Name) and func.id == "fixture":
                    count += 1
    return count


def evidence_snippets(text: str) -> list[str]:
    snippets: list[str] = []
    for line in text.splitlines():
        trimmed = line.strip()
        if not trimmed:
            continue
        if any(pattern.search(trimmed) for pattern in PATTERN_MAP.values()):
            snippets.append(trimmed[:180])
        elif "pytest.skip(" in trimmed or "@pytest.mark.xfail" in trimmed:
            snippets.append(trimmed[:180])
    deduped: list[str] = []
    seen: set[str] = set()
    for snippet in snippets:
        if snippet not in seen:
            deduped.append(snippet)
            seen.add(snippet)
        if len(deduped) == 6:
            break
    return deduped


def analyze_file(
    root: Path,
    path: Path,
    ignore_paths: list[str],
    refs: dict[str, list[str]],
    basename_map: dict[str, list[str]],
) -> InventoryEntry:
    rel_path = str(path.relative_to(root))
    text = path.read_text(encoding="utf-8", errors="replace")
    line_count = text.count("\n") + (0 if text.endswith("\n") or not text else 1)
    tree = ast.parse(text)

    ignored, ignore_pattern = path_is_ignored(rel_path, ignore_paths)
    same_name_paths = sorted(
        candidate for candidate in basename_map[path.name] if candidate != rel_path
    )
    top_level = path.parts[1] if len(path.parts) > 2 else "root"
    manual_dir_gate = rel_path.startswith(str(MANUAL_DIR))
    needs_update = bool(PATTERN_MAP["needs_update"].search(text))
    not_implemented = bool(PATTERN_MAP["not_implemented"].search(text))

    entry = InventoryEntry(
        path=rel_path,
        top_level=top_level,
        line_count=line_count,
        file_bytes=path.stat().st_size,
        test_function_count=test_function_count(tree),
        test_class_count=test_class_count(tree),
        fixture_count=fixture_count(tree),
        import_count=sum(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)),
        skip_count=text.count("pytest.skip("),
        skipif_count=text.count("@pytest.mark.skipif"),
        importorskip_count=text.count("pytest.importorskip("),
        xfail_count=text.count("@pytest.mark.xfail") + text.count("pytest.xfail("),
        todo_count=len(re.findall(r"\b(TODO|FIXME|HACK|XXX)\b", text)),
        print_count=text.count("print("),
        has_shebang=text.startswith("#!"),
        has_main_guard=bool(PATTERN_MAP["main_guard"].search(text)),
        has_localhost=bool(PATTERN_MAP["localhost"].search(text)),
        has_manual_gate=manual_dir_gate or bool(PATTERN_MAP["manual_gate"].search(text)),
        has_not_implemented=not_implemented,
        has_needs_update=needs_update,
        ignored_by_default=ignored,
        ignore_pattern=ignore_pattern,
        workflow_ref_count=find_reference_count(rel_path, refs["workflows"]),
        docs_ref_count=find_reference_count(rel_path, refs["docs"]),
        scripts_ref_count=find_reference_count(rel_path, refs["scripts"]),
        config_ref_count=find_reference_count(rel_path, refs["config"]),
        same_basename_count=len(same_name_paths) + 1,
        same_basename_other_paths=same_name_paths[:8],
        import_roots=import_roots(tree),
        evidence_snippets=evidence_snippets(text),
    )
    classify_entry(entry)
    return entry


def classify_entry(entry: InventoryEntry) -> None:
    reasons: list[str] = []
    score = 0

    if entry.ignored_by_default:
        reasons.append(f"ignored-by-default:{entry.ignore_pattern}")
        score += 4
    if entry.has_manual_gate:
        reasons.append("manual-gate-or-directory")
        score += 6
    if entry.has_localhost:
        reasons.append("localhost-dependent")
        score += 4
    if entry.has_shebang:
        reasons.append("script-shebang")
        score += 1
    if entry.has_main_guard:
        reasons.append("script-main-guard")
        score += 5
    if entry.print_count >= 5:
        reasons.append("print-heavy-manual-harness")
        score += 2
    if entry.same_basename_count > 1:
        outside_tests = [
            other for other in entry.same_basename_other_paths if not other.startswith("tests/")
        ]
        if outside_tests:
            reasons.append("duplicate-basename-outside-tests")
            score += 4
        else:
            reasons.append("duplicate-basename-inside-tests")
            score += 1
    if entry.xfail_count:
        reasons.append("xfail-present")
        score += 2
    if entry.has_needs_update:
        reasons.append("needs-update-language")
        score += 2
    if entry.has_not_implemented:
        reasons.append("not-implemented-language")
        score += 1
    if entry.workflow_ref_count:
        reasons.append("workflow-referenced")
        score -= 4
    if entry.docs_ref_count:
        reasons.append("docs-referenced")
        score -= 1
    if (
        entry.top_level in {"unit", "integration", "optimizer_validation"}
        and not entry.ignored_by_default
    ):
        score -= 2

    rel_path = entry.path
    if entry.has_manual_gate and (entry.has_localhost or entry.print_count >= 3):
        recommendation = "move-out-of-tests"
    elif (
        "duplicate-basename-outside-tests" in reasons
        and entry.ignored_by_default
        and entry.workflow_ref_count == 0
    ):
        recommendation = "merge-or-delete-after-review"
    elif entry.ignored_by_default and any(hint in rel_path for hint in RUNTIME_QUARANTINE_HINTS):
        recommendation = "keep-quarantined-runtime"
    elif entry.xfail_count or entry.has_needs_update or (
        entry.has_not_implemented
        and (entry.skip_count or entry.skipif_count or entry.importorskip_count)
    ):
        recommendation = "keep-track-debt"
    elif entry.ignored_by_default:
        recommendation = "review-quarantine"
    else:
        recommendation = "keep"

    if entry.workflow_ref_count and recommendation == "merge-or-delete-after-review":
        recommendation = "review-quarantine"

    entry.stale_score = max(score, 0)
    entry.recommendation = recommendation
    entry.reasons = reasons


def discover_test_files(root: Path) -> list[Path]:
    return sorted((root / "tests").rglob("test_*.py"))


def batch_candidates(candidates: list[InventoryEntry], batch_size: int) -> list[list[InventoryEntry]]:
    return [candidates[i : i + batch_size] for i in range(0, len(candidates), batch_size)]


def write_csv(entries: list[InventoryEntry], output_path: Path) -> None:
    rows = [asdict(entry) for entry in entries]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(data: object, output_path: Path) -> None:
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    widths = [max(len(row[idx]) for row in rows) for idx in range(len(rows[0]))]
    rendered = []
    for idx, row in enumerate(rows):
        rendered.append(
            "| " + " | ".join(cell.ljust(widths[col]) for col, cell in enumerate(row)) + " |"
        )
        if idx == 0:
            rendered.append("| " + " | ".join("-" * width for width in widths) + " |")
    return "\n".join(rendered)


def write_summary_markdown(
    entries: list[InventoryEntry],
    candidates: list[InventoryEntry],
    missing_ignores: list[str],
    output_path: Path,
    batch_size: int,
) -> None:
    total = len(entries)
    recommendation_counts = Counter(entry.recommendation for entry in entries)
    ignored_count = sum(entry.ignored_by_default for entry in entries)
    skip_files = sum(
        1
        for entry in entries
        if entry.skip_count or entry.skipif_count or entry.importorskip_count
    )
    xfail_files = sum(1 for entry in entries if entry.xfail_count)
    manual_candidates = [
        entry for entry in candidates if entry.recommendation == "move-out-of-tests"
    ]
    duplicate_candidates = [
        entry
        for entry in candidates
        if entry.recommendation == "merge-or-delete-after-review"
    ]
    top_rows = [["Path", "Recommendation", "Score", "Signals", "Workflow Refs"]]
    for entry in candidates[:20]:
        top_rows.append(
            [
                entry.path,
                entry.recommendation,
                str(entry.stale_score),
                ", ".join(entry.reasons[:3]),
                str(entry.workflow_ref_count),
            ]
        )

    lines = [
        "# Stale Test Audit",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}",
        "",
        "## Overview",
        "",
        f"- Test files inventoried: {total}",
        f"- Ignored by default: {ignored_count}",
        f"- Files with skip logic: {skip_files}",
        f"- Files with xfail logic: {xfail_files}",
        f"- Review candidates routed to Sonnet: {len(candidates)}",
        f"- Sonnet batch size: {batch_size}",
        "",
        "## Recommendation Counts",
        "",
    ]
    table = [["Recommendation", "Count"]]
    for recommendation, count in sorted(recommendation_counts.items()):
        table.append([recommendation, str(count)])
    lines.append(markdown_table(table))
    lines.extend(
        [
            "",
            "## Highest-Signal Candidates",
            "",
            markdown_table(top_rows),
            "",
            "## Immediate Human Review Queue",
            "",
            f"- Manual/local validation harnesses to move out of `tests/`: {len(manual_candidates)}",
            f"- Duplicate/merge candidates: {len(duplicate_candidates)}",
            "- Keep-but-track debt items include xfail files and tests documenting not-yet-implemented behavior.",
            "",
            "## Pytest Config Gaps",
            "",
        ]
    )
    if missing_ignores:
        for missing in missing_ignores:
            lines.append(f"- Ignore target does not exist: `{missing}`")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## How To Use With Sonnet",
            "",
            "1. Start with `sonnet_candidates.json` and the `sonnet_batches/` directory.",
            f"2. Review only the generated candidate set, not all {total} test files.",
            "3. Ask Sonnet to classify each candidate as `keep`, `move`, `merge`, `delete-candidate`, or `needs-owner` with evidence.",
            "4. Require Sonnet to justify deletion with at least two independent stale signals and note any workflow/doc references.",
            "5. Apply deletions only in small PRs after human review.",
            "",
            "The full inventory still retains additional high-score `keep` files for manual follow-up, but they are intentionally not sent to Sonnet by default.",
            "",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_output_dir(root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return root / ".release_review" / "runs" / f"stale-test-audit-{timestamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated artifacts. Default: .release_review/runs/stale-test-audit-<timestamp>/",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of candidate files per Sonnet batch.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    output_dir = args.output_dir or default_output_dir(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sonnet_batches").mkdir(exist_ok=True)

    ignore_paths = load_pytest_ignores(root)
    refs = iter_reference_texts(root)
    basename_map = same_basename_paths(root)
    entries = [
        analyze_file(root, path, ignore_paths, refs, basename_map)
        for path in discover_test_files(root)
    ]
    entries.sort(key=lambda entry: (entry.path,))

    candidates = [entry for entry in entries if entry.recommendation != "keep"]
    candidates.sort(
        key=lambda entry: (
            -entry.stale_score,
            entry.recommendation,
            entry.path,
        )
    )

    missing_ignores = [
        ignore for ignore in ignore_paths if not (root / ignore).exists()
    ]

    inventory_json = output_dir / "test_file_inventory.json"
    inventory_csv = output_dir / "test_file_inventory.csv"
    candidates_json = output_dir / "sonnet_candidates.json"
    summary_md = output_dir / "stale_test_candidates.md"
    metadata_json = output_dir / "audit_metadata.json"

    write_json([asdict(entry) for entry in entries], inventory_json)
    write_csv(entries, inventory_csv)
    write_json([asdict(entry) for entry in candidates], candidates_json)
    write_summary_markdown(
        entries=entries,
        candidates=candidates,
        missing_ignores=missing_ignores,
        output_path=summary_md,
        batch_size=args.batch_size,
    )

    batches = batch_candidates(candidates, args.batch_size)
    batch_paths: list[str] = []
    for index, batch in enumerate(batches, start=1):
        batch_path = output_dir / "sonnet_batches" / f"batch_{index:02d}.json"
        write_json([asdict(entry) for entry in batch], batch_path)
        batch_paths.append(str(batch_path.relative_to(root)))

    write_json(
        {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "repo_root": str(root),
            "inventory_count": len(entries),
            "candidate_count": len(candidates),
            "batch_size": args.batch_size,
            "missing_ignore_paths": missing_ignores,
            "artifacts": {
                "inventory_json": str(inventory_json.relative_to(root)),
                "inventory_csv": str(inventory_csv.relative_to(root)),
                "candidates_json": str(candidates_json.relative_to(root)),
                "summary_md": str(summary_md.relative_to(root)),
                "batch_paths": batch_paths,
            },
        },
        metadata_json,
    )

    print(f"Output dir: {output_dir.relative_to(root)}")
    print(f"Inventory entries: {len(entries)}")
    print(f"Candidates: {len(candidates)}")
    print(f"Missing ignore targets: {len(missing_ignores)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
