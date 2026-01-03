"""Module inventory generator."""

from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from traigent.utils.secure_path import (
    PathTraversalError,
    safe_read_text,
    safe_write_text,
    validate_path,
)

try:  # pragma: no cover
    from .analysis_utils import (
        count_sloc,
        detect_language,
        format_timestamp,
        load_coverage_map,
        load_lint_map,
        run_command,
        safe_relpath,
        to_module_name,
        write_csv,
    )
except ImportError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.code_analysis.analysis_utils import (
        count_sloc,
        detect_language,
        format_timestamp,
        load_coverage_map,
        load_lint_map,
        run_command,
        safe_relpath,
        to_module_name,
        write_csv,
    )


def find_codeowners_file(project_root: Path) -> Optional[Path]:
    candidates = [
        project_root / "CODEOWNERS",
        project_root / ".github" / "CODEOWNERS",
        project_root / "docs" / "CODEOWNERS",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_codeowners(project_root: Path) -> List[Tuple[str, List[str]]]:
    codeowners_path = find_codeowners_file(project_root)
    if not codeowners_path:
        return []
    rules: List[Tuple[str, List[str]]] = []
    try:
        safe_path = validate_path(codeowners_path, project_root, must_exist=True)
        content = safe_read_text(safe_path, project_root)
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            pattern = parts[0]
            owners = parts[1:]
            rules.append((pattern, owners))
    except OSError:
        return []
    return rules


def resolve_owners(path: Path, project_root: Path, rules: Sequence[Tuple[str, List[str]]]) -> str:
    rel_path = safe_relpath(path, project_root)
    matched: Optional[List[str]] = None
    matched_len = -1
    for pattern, owners in rules:
        if fnmatch.fnmatch(rel_path, pattern.lstrip("/")):
            if len(pattern) > matched_len:
                matched = owners
                matched_len = len(pattern)
    if matched:
        return " ".join(matched)
    return ""


def gather_inventory(
    project_root: Path,
    source_root: Path,
    coverage_map: Dict[str, float],
    lint_map: Dict[str, int],
    codeowners_rules: Sequence[Tuple[str, List[str]]],
) -> Iterable[Sequence[object]]:
    for path in sorted(source_root.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        module = to_module_name(source_root, path)
        language = detect_language(path)
        rel_path = safe_relpath(path, project_root)
        sloc = count_sloc(path)
        try:
            stat = path.stat()
            size = stat.st_size
            modified = format_timestamp(stat.st_mtime)
        except OSError:
            size = 0
            modified = ""
        owners = resolve_owners(path, project_root, codeowners_rules)
        coverage = coverage_map.get(rel_path, None)
        lint_errors = lint_map.get(rel_path, 0)
        yield (
            module,
            rel_path,
            language,
            sloc,
            size,
            modified,
            owners,
            f"{coverage:.2f}" if coverage is not None else "",
            lint_errors,
        )


def run_lint(project_root: Path, lint_output: Path) -> None:
    lint_output.parent.mkdir(parents=True, exist_ok=True)
    result = run_command(
        ["ruff", "check", "--output-format", "json", "--exit-zero"],
        project_root,
    )
    if result.stdout:
        safe_write_text(lint_output, result.stdout, project_root, encoding="utf-8")
    elif result.returncode != 0:
        safe_write_text(
            lint_output,
            f"{{\n  \"error\": \"ruff command failed with code {result.returncode}\"\n}}\n",
            project_root,
            encoding="utf-8",
        )


def run_coverage(project_root: Path, coverage_output: Path) -> None:
    coverage_output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "pytest",
        "--cov=traigent",
        f"--cov-report=xml:{coverage_output}",
    ]
    result = run_command(cmd, project_root)
    if result.returncode != 0:
        # Persist stdout/stderr for troubleshooting and fall back to empty coverage map.
        failure_log = coverage_output.with_suffix(".log")
        safe_write_text(
            failure_log,
            f"Command: {' '.join(cmd)}\nReturn code: {result.returncode}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n",
            project_root,
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate code inventory metadata.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-root", type=Path, default=Path("traigent"))
    parser.add_argument("--coverage-xml", type=Path, help="Path to coverage XML", required=True)
    parser.add_argument("--lint-json", type=Path, help="Path to lint JSON file", required=True)
    parser.add_argument("--output", type=Path, help="Output CSV path", required=True)
    parser.add_argument("--skip-owners", action="store_true")
    args = parser.parse_args()

    base_dir = Path.cwd()
    try:
        project_root = validate_path(args.project_root, base_dir, must_exist=True)
        source_root = validate_path(
            project_root / args.source_root,
            project_root,
            must_exist=True,
        )
        coverage_xml = validate_path(args.coverage_xml, project_root, must_exist=True)
        lint_json = validate_path(args.lint_json, project_root, must_exist=True)
        output_path = validate_path(args.output, project_root)
    except (PathTraversalError, FileNotFoundError) as exc:
        raise SystemExit(f"Error: {exc}") from exc

    codeowners_rules: Sequence[Tuple[str, List[str]]] = []
    if not args.skip_owners:
        codeowners_rules = load_codeowners(project_root)

    coverage_map = load_coverage_map(coverage_xml, project_root)
    lint_map = load_lint_map(lint_json, project_root)

    rows = gather_inventory(project_root, source_root, coverage_map, lint_map, codeowners_rules)
    header = [
        "module",
        "path",
        "language",
        "sloc",
        "file_size_bytes",
        "last_modified_iso",
        "owners",
        "test_coverage_percent",
        "lint_error_count",
    ]
    write_csv(output_path, header, rows)


if __name__ == "__main__":  # pragma: no cover
    main()
