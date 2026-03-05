#!/usr/bin/env python3
"""Repository-wide what-if scanner for Traigent source.

Scans all Python files under a source directory, records full file/line coverage,
and emits heuristic findings for manual follow-up.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PatternRule:
    code: str
    severity: str
    category: str
    pattern: re.Pattern[str]
    description: str


PATTERN_RULES: tuple[PatternRule, ...] = (
    PatternRule(
        code="PATTERN004",
        severity="medium",
        category="resilience",
        pattern=re.compile(r"\bwhile\s+True\s*:"),
        description="Infinite loop requires explicit exit/cancel safeguards.",
    ),
    PatternRule(
        code="PATTERN005",
        severity="medium",
        category="resilience",
        pattern=re.compile(r"\bexcept\s+Exception\s*:"),
        description="Broad exception catch may hide failure modes.",
    ),
    PatternRule(
        code="PATTERN006",
        severity="high",
        category="security",
        pattern=re.compile(r"\bexcept\s*:\s*$"),
        description="Bare except catches system-exiting exceptions.",
    ),
    PatternRule(
        code="PATTERN007",
        severity="medium",
        category="network",
        pattern=re.compile(r"\bverify\s*=\s*False\b"),
        description="TLS verification disabled in network call.",
    ),
    PatternRule(
        code="PATTERN008",
        severity="medium",
        category="determinism",
        pattern=re.compile(r"\brandom\.(random|uniform|randint|choice|choices)\s*\("),
        description="Random behavior without explicit seed can reduce reproducibility.",
    ),
    PatternRule(
        code="PATTERN009",
        severity="medium",
        category="performance",
        pattern=re.compile(r"\btime\.sleep\s*\("),
        description="Blocking sleep can stall event loops or workers.",
    ),
    PatternRule(
        code="PATTERN010",
        severity="low",
        category="maintainability",
        pattern=re.compile(r"\b(TODO|FIXME|HACK|XXX)\b"),
        description="Marker indicates unfinished or brittle implementation.",
    ),
    PatternRule(
        code="PATTERN011",
        severity="medium",
        category="config",
        pattern=re.compile(r"os\.environ\s*\[\s*[\"'][^\"']+[\"']\s*\]\s*="),
        description="Global env mutation can leak state across calls/tests.",
    ),
)


def discover_py_files(src_dir: Path) -> list[Path]:
    return sorted(path for path in src_dir.rglob("*.py") if path.is_file())


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def module_name(path: Path, src_dir: Path) -> str:
    rel = path.relative_to(src_dir)
    if len(rel.parts) == 1:
        return "root"
    return rel.parts[0]


def package_segment(path: Path, src_dir: Path) -> str:
    rel = path.relative_to(src_dir)
    if len(rel.parts) < 2:
        return "root"
    return f"{rel.parts[0]}/{rel.parts[1]}"


def iter_pattern_findings(path: Path, lines: list[str]) -> Iterable[dict]:
    for line_no, line in enumerate(lines, start=1):
        for rule in PATTERN_RULES:
            if rule.pattern.search(line):
                yield {
                    "source": "pattern",
                    "rule": rule.code,
                    "severity": rule.severity,
                    "category": rule.category,
                    "file": str(path),
                    "line": line_no,
                    "symbol": "",
                    "description": rule.description,
                    "evidence": line.strip()[:400],
                }


def node_length(node: ast.AST) -> int:
    start = getattr(node, "lineno", None)
    end = getattr(node, "end_lineno", None)
    if isinstance(start, int) and isinstance(end, int) and end >= start:
        return end - start + 1
    return 0


def ast_findings(path: Path, text: str) -> list[dict]:
    findings: list[dict] = []
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        findings.append(
            {
                "source": "ast",
                "rule": "AST000",
                "severity": "high",
                "category": "correctness",
                "file": str(path),
                "line": int(exc.lineno or 1),
                "symbol": "",
                "description": "Syntax parsing failed; file may contain invalid Python.",
                "evidence": str(exc).strip()[:400],
            }
        )
        return findings

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # eval / exec usage
            if isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec"}:
                findings.append(
                    {
                        "source": "ast",
                        "rule": "AST004",
                        "severity": "high",
                        "category": "security",
                        "file": str(path),
                        "line": int(getattr(node, "lineno", 1)),
                        "symbol": node.func.id,
                        "description": f"Use of {node.func.id}() can execute dynamic code.",
                        "evidence": f"{node.func.id}(...)",
                    }
                )

            # subprocess shell=True
            for kw in node.keywords or []:
                if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    findings.append(
                        {
                            "source": "ast",
                            "rule": "AST005",
                            "severity": "high",
                            "category": "security",
                            "file": str(path),
                            "line": int(getattr(node, "lineno", 1)),
                            "symbol": "",
                            "description": "subprocess call with shell=True can enable command injection.",
                            "evidence": "shell=True",
                        }
                    )

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
            length = node_length(node)
            if length >= 150:
                findings.append(
                    {
                        "source": "ast",
                        "rule": "AST001",
                        "severity": "medium",
                        "category": "maintainability",
                        "file": str(path),
                        "line": int(getattr(node, "lineno", 1)),
                        "symbol": name,
                        "description": "Very long function; what-if reasoning may miss edge cases.",
                        "evidence": f"function_len={length}",
                    }
                )

            # Mutable defaults
            all_defaults = list(node.args.defaults) + [
                d for d in node.args.kw_defaults if d is not None
            ]
            for default in all_defaults:
                if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    findings.append(
                        {
                            "source": "ast",
                            "rule": "AST002",
                            "severity": "high",
                            "category": "correctness",
                            "file": str(path),
                            "line": int(getattr(default, "lineno", getattr(node, "lineno", 1))),
                            "symbol": name,
                            "description": "Mutable default argument can leak state between calls.",
                            "evidence": ast.unparse(default)[:400]
                            if hasattr(ast, "unparse")
                            else default.__class__.__name__,
                        }
                    )

            # Async function containing time.sleep
            if isinstance(node, ast.AsyncFunctionDef):
                for child in ast.walk(node):
                    if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                        if (
                            isinstance(child.func.value, ast.Name)
                            and child.func.value.id == "time"
                            and child.func.attr == "sleep"
                        ):
                            findings.append(
                                {
                                    "source": "ast",
                                    "rule": "AST003",
                                    "severity": "high",
                                    "category": "performance",
                                    "file": str(path),
                                    "line": int(getattr(child, "lineno", getattr(node, "lineno", 1))),
                                    "symbol": name,
                                    "description": "Blocking sleep inside async function.",
                                    "evidence": "time.sleep(...) in async def",
                                }
                            )

            param_names = [arg.arg for arg in node.args.args]
            critical_params = [p for p in param_names if p in {"max_trials", "timeout", "max_iterations", "subset_size"}]
            if critical_params:
                compare_text = ast.dump(node, include_attributes=False)
                for param in critical_params:
                    # Heuristic: flag when parameter exists but no obvious comparison/guard logic references it.
                    if param not in compare_text:
                        continue
                    has_guard = False
                    for child in ast.walk(node):
                        if isinstance(child, ast.Compare):
                            names = [n.id for n in ast.walk(child) if isinstance(n, ast.Name)]
                            if param in names:
                                has_guard = True
                                break
                    if not has_guard:
                        findings.append(
                            {
                                "source": "ast",
                                "rule": "AST006",
                                "severity": "medium",
                                "category": "api",
                                "file": str(path),
                                "line": int(getattr(node, "lineno", 1)),
                                "symbol": name,
                                "description": f"Parameter '{param}' has no local guard/comparison; validate bounds in call path.",
                                "evidence": f"def {name}(..., {param}, ...)",
                            }
                        )

    return findings


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Traigent what-if scanner")
    parser.add_argument("--src-dir", default="traigent", help="Source directory to scan")
    parser.add_argument(
        "--out-dir",
        default="docs/reviews/what_if_2026-03-04/generated",
        help="Output directory for scanner artifacts",
    )
    args = parser.parse_args()

    src_dir = Path(args.src_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = discover_py_files(src_dir)
    scanned_at = datetime.now(timezone.utc).isoformat()

    coverage_rows: list[dict] = []
    findings: list[dict] = []
    total_lines = 0

    for file_path in files:
        text = safe_read(file_path)
        lines = text.splitlines()
        line_count = len(lines)
        total_lines += line_count
        rel = file_path.relative_to(src_dir.parent)

        coverage_rows.append(
            {
                "file": str(rel),
                "module": module_name(file_path, src_dir),
                "package_segment": package_segment(file_path, src_dir),
                "line_count": line_count,
                "sha256": sha256_hex(text),
                "auto_scan_status": "scanned",
                "manual_review_status": "pending",
                "scanned_at_utc": scanned_at,
            }
        )

        findings.extend(iter_pattern_findings(rel, lines))
        findings.extend(ast_findings(rel, text))

    coverage_columns = [
        "file",
        "module",
        "package_segment",
        "line_count",
        "sha256",
        "auto_scan_status",
        "manual_review_status",
        "scanned_at_utc",
    ]
    findings_columns = [
        "source",
        "rule",
        "severity",
        "category",
        "file",
        "line",
        "symbol",
        "description",
        "evidence",
    ]

    write_csv(out_dir / "file_coverage.csv", coverage_rows, coverage_columns)
    write_csv(out_dir / "auto_findings.csv", findings, findings_columns)

    summary = {
        "scanned_at_utc": scanned_at,
        "source_dir": str(src_dir),
        "files_scanned": len(files),
        "total_lines_scanned": total_lines,
        "finding_count": len(findings),
        "severity_breakdown": {
            sev: sum(1 for item in findings if item["severity"] == sev)
            for sev in ("high", "medium", "low")
        },
        "categories": sorted({row["package_segment"] for row in coverage_rows}),
    }
    (out_dir / "scan_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
