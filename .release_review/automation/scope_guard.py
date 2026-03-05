#!/usr/bin/env python3
"""Scope guard for release-review v2."""

from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ScopeConfig:
    include: list[str]
    exclude: list[str]
    shared_files: list[str]
    base_reference: str


class ScopeGuard:
    def __init__(self, repo_path: str | Path | None = None) -> None:
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()

    def _git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=self.repo_path,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def load_scope_config(self, scope_file: str | Path) -> ScopeConfig:
        data = yaml.safe_load(Path(scope_file).read_text())
        if not isinstance(data, dict):
            raise ValueError("scope config must be a mapping")
        return ScopeConfig(
            include=[str(x) for x in data.get("include", [])],
            exclude=[str(x) for x in data.get("exclude", [])],
            shared_files=[str(x) for x in data.get("shared_files", [])],
            base_reference=str(data.get("base_reference", "main")),
        )

    def resolve_base_reference(self, base_reference: str, explicit_base: str | None) -> str:
        if explicit_base:
            return explicit_base
        if base_reference != "previous_release_tag":
            return base_reference
        try:
            return self._git("describe", "--tags", "--abbrev=0")
        except subprocess.CalledProcessError:
            return "main"

    def get_modified_files(self, target_ref: str, base_ref: str) -> list[str]:
        merge_base = self._git("merge-base", base_ref, target_ref)
        output = self._git("diff", "--name-only", f"{merge_base}..{target_ref}")
        return [line for line in output.splitlines() if line.strip()]

    def _matches_any(self, path: str, patterns: list[str]) -> bool:
        if not patterns:
            return True
        return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)

    def is_allowed(self, file_path: str, config: ScopeConfig) -> tuple[bool, str | None]:
        if self._matches_any(file_path, config.exclude):
            return True, "excluded-by-policy"

        if file_path in config.shared_files:
            return True, "shared-file"

        if self._matches_any(file_path, config.include):
            return True, "in-scope"

        return False, "not-included"

    def validate(
        self,
        target_ref: str,
        config: ScopeConfig,
        base_ref: str,
    ) -> dict[str, Any]:
        files = self.get_modified_files(target_ref=target_ref, base_ref=base_ref)

        violations: list[str] = []
        decisions: list[dict[str, str]] = []

        for file_path in files:
            allowed, reason = self.is_allowed(file_path, config)
            decisions.append({"file": file_path, "allowed": str(allowed).lower(), "reason": reason or ""})
            if not allowed:
                violations.append(file_path)

        return {
            "valid": len(violations) == 0,
            "target_ref": target_ref,
            "base_ref": base_ref,
            "total_files": len(files),
            "violations": violations,
            "decisions": decisions,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate scope of modified files")
    parser.add_argument("--target", default="HEAD", help="Target ref to validate")
    parser.add_argument("--base", default=None, help="Base ref override")
    parser.add_argument("--scope-file", default=".release_review/scope.yml")
    parser.add_argument("--repo-path", default=".")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    guard = ScopeGuard(args.repo_path)

    try:
        config = guard.load_scope_config(args.scope_file)
        base_ref = guard.resolve_base_reference(config.base_reference, args.base)
        result = guard.validate(target_ref=args.target, config=config, base_ref=base_ref)
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}")
        return 2

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if result["valid"]:
            print(f"✅ Scope valid: {result['total_files']} files checked")
        else:
            print(f"❌ Scope violations ({len(result['violations'])}):")
            for file_path in result["violations"]:
                print(f"  - {file_path}")

    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
