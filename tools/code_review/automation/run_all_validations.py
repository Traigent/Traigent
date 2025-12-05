#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import sys as _sys
from pathlib import Path as _P

_ROOT = _P(__file__).resolve()
AUTOMATION_ROOT = _ROOT.parent
REPO_ROOT = _ROOT.parents[3]
REPORTS_ROOT = REPO_ROOT / "reports" / "1_quality" / "automated_reviews"

if str(AUTOMATION_ROOT) not in _sys.path:
    _sys.path.insert(0, str(AUTOMATION_ROOT))

from _shared.validator import validate_report

TRACKS = {
    "code_quality": {
        "required": AUTOMATION_ROOT / "code_quality" / "required_checks.json",
        "out": REPORTS_ROOT / "code_quality",
    },
    "soundness_correctness": {
        "required": AUTOMATION_ROOT / "soundness_correctness" / "required_checks.json",
        "out": REPORTS_ROOT / "soundness_correctness",
    },
    "performance": {
        "required": AUTOMATION_ROOT / "performance" / "required_checks.json",
        "out": REPORTS_ROOT / "performance",
    },
    "security": {
        "required": AUTOMATION_ROOT / "security" / "required_checks.json",
        "out": REPORTS_ROOT / "security",
    },
}


def iter_modules(folder: str) -> list[Path]:
    base = Path(folder)
    return [
        p
        for p in base.rglob("*.py")
        if "tests" not in p.parts and p.name != "__init__.py"
    ]


def validate_one(module: Path) -> bool:
    rel = module.as_posix()
    ok_all = True
    for track, cfg in TRACKS.items():
        cfg["out"].mkdir(parents=True, exist_ok=True)
        report = cfg["out"] / f"{rel}.review.json"
        if not report.exists():
            print(f"[{track}] MISSING REPORT: {report}")
            print(
                "  ↳ Action: ask the reviewer to run the"
                f" {track} review for {rel} and save to this path."
            )
            ok_all = False
            continue
        ok, details = validate_report(
            module_path=rel,
            report_path=str(report),
            category=track,
            required_checks_path=str(cfg["required"]),
        )
        if not ok:
            ok_all = False
            print(f"[{track}] FAIL: {rel}")
            for p in details["problems"]:
                print(f" - {p}")
    return ok_all


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--module")
    group.add_argument("--folder")
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    args = ap.parse_args(argv)

    ok = True
    modules = [Path(args.module)] if args.module else iter_modules(args.folder)  # type: ignore[arg-type]
    for m in modules:
        if not validate_one(m):
            ok = False
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
