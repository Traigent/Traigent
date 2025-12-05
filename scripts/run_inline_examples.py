#!/usr/bin/env python3
"""
Execute extracted inline examples (best-effort) and write/update results.json
in each example folder.

Usage:
  python scripts/run_inline_examples.py examples/docs/page-inline/configuration-management
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Dict, List


def find_example_modules(base: str) -> List[str]:
    modules: List[str] = []
    for root, dirs, files in os.walk(base):
        # Skip caches
        if "__pycache__" in root:
            continue
        # Prefer a single module in each example folder
        py_files = [f for f in files if f.endswith(".py") and not f.startswith("_")]
        if py_files:
            # If folder contains exactly one .py example file, execute it
            # Otherwise, prefer file named as folder leaf if present
            leaf = os.path.basename(root)
            preferred = f"{leaf}.py"
            if preferred in py_files:
                modules.append(os.path.join(root, preferred))
            elif len(py_files) == 1:
                modules.append(os.path.join(root, py_files[0]))
            else:
                # Fallback: pick the first deterministically
                modules.append(os.path.join(root, sorted(py_files)[0]))
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
    if not os.path.isdir(base):
        print(f"Not a directory: {base}", file=sys.stderr)
        return 1

    modules = find_example_modules(base)
    # Group by example folder (parent directory)
    by_folder: Dict[str, List[str]] = {}
    for m in modules:
        by_folder.setdefault(os.path.dirname(m), []).append(m)

    summary: Dict[str, dict] = {}
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
