#!/usr/bin/env python3
"""Validate LLM review reports for module coverage and required checks.

Report format (per track):
{
  "module": "traigent/core/cache_policy.py",
  "category": "code_quality|soundness_correctness|performance|security",
  "summary": "...",
  "functions": [ {"name": "foo", "status": "ok|issue|needs_followup", "notes": "..."}, ... ],
  "checks": [ {"name": "docstrings_present", "result": "pass|fail", "evidence": "..."}, ... ]
}

Validation:
- All discovered functions (top-level + class methods) must appear in functions[].
- All required checks for the category must be present.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

from .function_inventory import list_functions

ALLOWED_FUNC_STATUS = {"ok", "issue", "needs_followup"}
ALLOWED_CHECK_RESULT = {"pass", "fail", "needs_followup"}
ALLOWED_ISSUE_SEVERITY = {"low", "medium", "high", "critical"}
ALLOWED_CONFIDENCE = {"low", "medium", "high"}


def load_required_checks(path: str | Path) -> list[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "required_checks" in data:
        return [str(x) for x in data["required_checks"]]
    if isinstance(data, list):
        return [str(x) for x in data]
    raise ValueError("Invalid required_checks file")


def validate_report(
    module_path: str,
    report_path: str,
    category: str,
    required_checks_path: str,
) -> tuple[bool, dict[str, Any]]:
    inv = list_functions(module_path)
    module_file = Path(module_path)
    module_name = module_file.name
    try:
        module_lines = module_file.read_text(encoding="utf-8").splitlines()
    except Exception:
        module_lines = []
    expected_funcs = set(inv.get("top_level", [])) | set(inv.get("class_methods", []))
    expected_classes = set(inv.get("classes", []))

    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    problems: list[str] = []

    # Structure checks
    if not Path(module_path).exists():
        problems.append(f"module path not found: {module_path}")
    for key in ("module", "category", "summary", "classes", "functions", "checks"):
        if key not in report:
            problems.append(f"missing required key: {key}")
    # schema details (recommended unified shape)
    if "issues" not in report or not isinstance(report.get("issues"), list):
        problems.append("missing or invalid 'issues' (must be list)")
    if "recommendations" not in report or not isinstance(
        report.get("recommendations"), list
    ):
        problems.append("missing or invalid 'recommendations' (must be list)")

    # Basic quality of summary
    summary = report.get("summary", "")
    if not isinstance(summary, str) or len(summary.strip()) < 40:
        problems.append("summary too short; provide >= 40 chars of findings")

    # Category match
    if report.get("category") != category:
        problems.append(f"category mismatch: {report.get('category')} != {category}")

    # Functions coverage
    covered = {
        str(item.get("name")) for item in report.get("functions", []) if "name" in item
    }
    missing_funcs = sorted(list(expected_funcs - covered))
    if missing_funcs:
        problems.append(f"missing functions: {', '.join(missing_funcs)}")

    # Classes coverage
    covered_classes = set(report.get("classes", []))
    missing_classes = sorted(list(expected_classes - covered_classes))
    if expected_classes and not report.get("classes"):
        problems.append("missing classes array in report")
    elif missing_classes:
        problems.append(f"missing classes: {', '.join(missing_classes)}")

    # Functions item shape
    for item in report.get("functions", []):
        if not isinstance(item, dict):
            problems.append("functions[] entries must be objects")
            continue
        nm = item.get("name")
        st = item.get("status")
        notes = item.get("notes")
        if not nm or not isinstance(nm, str):
            problems.append("function entry missing 'name'")
        if st not in ALLOWED_FUNC_STATUS:
            problems.append(
                f"function '{nm}' has invalid status '{st}' (allowed: {sorted(ALLOWED_FUNC_STATUS)})"
            )
        if not isinstance(notes, str) or len(notes.strip()) < 20:
            problems.append(
                f"function '{nm}' notes too short; need >= 20 chars with brief evidence"
            )

    # Required checks
    required = set(load_required_checks(required_checks_path))
    present = {
        str(item.get("name")) for item in report.get("checks", []) if "name" in item
    }
    missing_checks = sorted(list(required - present))
    if missing_checks:
        problems.append(f"missing checks: {', '.join(missing_checks)}")

    # Checks shape and evidence
    for chk in report.get("checks", []):
        if not isinstance(chk, dict):
            problems.append("checks[] entries must be objects")
            continue
        nm = chk.get("name")
        res = chk.get("result")
        ev = chk.get("evidence")
        if res not in ALLOWED_CHECK_RESULT:
            problems.append(
                f"check '{nm}' has invalid result '{res}' (allowed: {sorted(ALLOWED_CHECK_RESULT)})"
            )
        if not isinstance(ev, str) or len(ev.strip()) < 30:
            problems.append(
                f"check '{nm}' evidence too short; need >= 30 chars with rationale or path:line refs"
            )
        conf = chk.get("confidence")
        if conf is not None and conf not in ALLOWED_CONFIDENCE:
            problems.append(
                f"check '{nm}' invalid confidence '{conf}' (allowed: {sorted(ALLOWED_CONFIDENCE)})"
            )

        # Verify any evidence path:line anchors are within file bounds for this module
        if isinstance(ev, str) and module_lines:
            # match both relative and fully qualified paths ending with the module file name
            for m in re.finditer(rf"([\w\./\\-]*{re.escape(module_name)}):(\d+)", ev):
                try:
                    line_no = int(m.group(2))
                    if line_no < 1 or line_no > len(module_lines):
                        problems.append(
                            f"check '{nm}' evidence references out-of-range line: {m.group(0)} (max {len(module_lines)})"
                        )
                except ValueError:
                    problems.append(
                        f"check '{nm}' evidence contains invalid line number: {m.group(0)}"
                    )

    # Issues shape
    known_scopes = covered | expected_classes
    for issue in report.get("issues", []):
        if not isinstance(issue, dict):
            problems.append("issues[] entries must be objects")
            continue
        if not issue.get("id") or not issue.get("title"):
            problems.append("issue missing 'id' or 'title'")
        sev = issue.get("severity")
        issue_id = issue.get("id", "<unknown>")
        sev_lower = None
        if not sev:
            problems.append(
                f"issue '{issue_id}' missing required 'severity' (one of {sorted(ALLOWED_ISSUE_SEVERITY)})"
            )
        else:
            sev_lower = str(sev).lower()
            if sev_lower not in ALLOWED_ISSUE_SEVERITY:
                problems.append(
                    f"issue '{issue_id}' invalid severity '{sev}' (allowed: {sorted(ALLOWED_ISSUE_SEVERITY)})"
                )
        conf = issue.get("confidence")
        if conf is not None and conf not in ALLOWED_CONFIDENCE:
            problems.append(
                f"issue '{issue_id}' invalid confidence '{conf}' (allowed: {sorted(ALLOWED_CONFIDENCE)})"
            )
        scope = issue.get("scope", [])
        if scope and not isinstance(scope, list):
            problems.append(f"issue '{issue_id}' scope must be a list")
        else:
            bad = [s for s in scope if s not in known_scopes]
            if bad:
                problems.append(
                    f"issue '{issue_id}' scope contains unknown symbols: {', '.join(bad)}"
                )
        if sev_lower in {"high", "critical"} and not scope:
            problems.append(
                f"issue '{issue_id}' (severity {sev}) must include scope referencing affected functions/classes"
            )

        evidence = issue.get("evidence")
        if not isinstance(evidence, str) or len(evidence.strip()) < 40:
            problems.append(
                f"issue '{issue_id}' missing detailed evidence (>=40 chars with path:line anchors)"
            )
        elif module_lines:
            for m in re.finditer(
                rf"([\w\./\\-]*{re.escape(module_name)}):(\d+)", evidence
            ):
                try:
                    line_no = int(m.group(2))
                    if line_no < 1 or line_no > len(module_lines):
                        problems.append(
                            f"issue '{issue_id}' evidence references out-of-range line: {m.group(0)} (max {len(module_lines)})"
                        )
                except ValueError:
                    problems.append(
                        f"issue '{issue_id}' evidence contains invalid line number: {m.group(0)}"
                    )
        if sev_lower in {"high", "critical"}:
            if evidence is None or ":" not in evidence:
                problems.append(
                    f"issue '{issue_id}' (severity {sev}) evidence must cite specific path:line anchors"
                )
            impact = issue.get("impact")
            if not isinstance(impact, str) or len(impact.strip()) < 30:
                problems.append(
                    f"issue '{issue_id}' (severity {sev}) must include an 'impact' explanation (>=30 chars)"
                )

        # Heuristic: if an issue claims a syntax error but module parses successfully, flag inconsistency
        title = str(issue.get("title", "")).lower()
        if "syntax" in title and module_lines:
            # Our AST parse (via list_functions) succeeded, so a syntax error claim is likely inconsistent.
            problems.append(
                f"issue '{issue.get('id')}' claims a syntax error, but module parsed successfully"
            )

    # Optional metadata/skip_reasons validation
    meta = report.get("metadata")
    if meta is not None and not isinstance(meta, dict):
        problems.append("metadata must be an object when present")
    skips = report.get("skip_reasons")
    if skips is not None:
        if not isinstance(skips, list) or not all(isinstance(s, str) for s in skips):
            problems.append("skip_reasons must be a list of strings when present")

    ok = not problems
    return ok, {
        "module": module_path,
        "report": report_path,
        "category": category,
        "expected_functions": sorted(list(expected_funcs)),
        "covered_functions": sorted(list(covered)),
        "expected_classes": sorted(list(expected_classes)),
        "covered_classes": sorted(list(covered_classes)),
        "required_checks": sorted(list(required)),
        "present_checks": sorted(list(present)),
        "problems": problems,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--required-checks", required=True)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    ok, details = validate_report(
        module_path=args.module,
        report_path=args.report,
        category=args.category,
        required_checks_path=args.required_checks,
    )

    if args.json:
        print(json.dumps({"ok": ok, **details}, indent=2))
    else:
        print(
            ("OK" if ok else "FAIL")
            + f": {details['module']} [{details['category']}]\n"
        )
        if details["problems"]:
            for p in details["problems"]:
                print(f" - {p}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
