#!/usr/bin/env python3
"""
Execute extracted inline examples (best-effort) and write/update results.json
in each example folder.

Usage:
  python scripts/run_inline_examples.py examples/gallery/page-inline/configuration-management
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _load_safe_helpers():
    """Load examples/utils/safe_helpers.py without depending on sys.path."""
    import importlib.util

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "examples" / "utils" / "safe_helpers.py"
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location(
                "_traigent_examples_safe_helpers", candidate
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
    raise ImportError("examples/utils/safe_helpers.py not found")


_SAFE_HELPERS = _load_safe_helpers()
resolve_within = _SAFE_HELPERS.resolve_within
UntrustedPathError = _SAFE_HELPERS.UntrustedPathError


def _project_root() -> Path:
    """Find the repo root by walking up to the project metadata."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "examples").is_dir():
            return parent
    return here.parent


_PROJECT_ROOT = _project_root()


def _resolve_base(base: str) -> Path:
    """Resolve the user-supplied base directory strictly under the project root.

    Rejects ``..`` escapes, symlinks pointing outside the tree, and non-
    directory targets. This prevents the runner from executing Python files
    living outside the repository (e.g. ``/tmp/payload``).
    """
    resolved = resolve_within(_PROJECT_ROOT, base, must_exist=True)
    if not resolved.is_dir():
        raise UntrustedPathError(f"{base!r} is not a directory")
    return resolved


def find_example_modules(base: str) -> list[str]:
    base_path = _resolve_base(base)
    modules: list[str] = []
    for root, _dirs, files in os.walk(base_path):
        if "__pycache__" in root:
            continue
        py_files = [f for f in files if f.endswith(".py") and not f.startswith("_")]
        if not py_files:
            continue
        leaf = os.path.basename(root)
        preferred = f"{leaf}.py"
        if preferred in py_files:
            chosen = os.path.join(root, preferred)
        elif len(py_files) == 1:
            chosen = os.path.join(root, py_files[0])
        else:
            # Fallback: pick the first deterministically
            chosen = os.path.join(root, sorted(py_files)[0])
        # Defence in depth: confirm the chosen module still resolves under
        # the validated base path (rejects symlinks escaping the tree).
        try:
            modules.append(str(resolve_within(base_path, chosen)))
        except UntrustedPathError:
            continue
    return sorted(set(modules))


def run_example_module(module_path: str) -> dict:
    module_path = os.path.abspath(module_path)
    folder = os.path.dirname(module_path)
    env = os.environ.copy()
    # Ensure local imports work
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
    try:
        proc = subprocess.run(
            [sys.executable, module_path],
            cwd=folder,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            env=env,
        )
        status = "success" if proc.returncode == 0 else "error"
        result = {
            "status": status,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as e:
        result = {"status": "error", "exception": str(e)}
    return result


def main() -> int:
    if len(sys.argv) < 2:
        print(
            "Usage: python scripts/run_inline_examples.py <base_examples_dir>",
            file=sys.stderr,
        )
        return 2
    base = sys.argv[1]
    try:
        _resolve_base(base)
    except UntrustedPathError as exc:
        print(f"Rejecting base directory {base!r}: {exc}", file=sys.stderr)
        return 1

    modules = find_example_modules(base)
    # Group by example folder (parent directory)
    by_folder: dict[str, list[str]] = {}
    for m in modules:
        by_folder.setdefault(os.path.dirname(m), []).append(m)

    summary: dict[str, dict] = {}
    for folder, mods in by_folder.items():
        results = []
        for m in sorted(mods):
            r = run_example_module(m)
            results.append(
                {
                    "module": os.path.basename(m),
                    "status": r.get("status"),
                    "returncode": r.get("returncode"),
                    "stdout": r.get("stdout"),
                    "stderr": r.get("stderr"),
                }
            )
        # Write single results.json in the folder
        out_path = os.path.join(folder, "results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"results": results}, f, indent=2)
        summary[folder] = {"count": len(results)}

    print(json.dumps({"count": len(modules), "groups": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
