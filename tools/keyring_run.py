#!/usr/bin/env python3
"""Run a command with secrets loaded from GNOME Keyring (Secret Service).

This avoids putting secrets in `.env` files or printing them to stdout.

Secrets are looked up using:
  secret-tool lookup project <project> name <ENV_VAR_NAME>

Example:
  python tools/keyring_run.py --project traigent --vars GROQ_API_KEY -- python examples/tvl/reusable_safety/run_demo.py --real-llm
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_SECRET_NAME_PATTERN = re.compile(
    r"(?x)"
    r"(?:"
    r".*_API_KEY$|"
    r".*_API_TOKEN$|"
    r".*_TOKEN$|"
    r".*_SECRET(?:_KEY)?$|"
    r".*_PASSWORD$|"
    r".*_ENCRYPTION_KEY$|"
    r"JWT_SECRET_KEY$|"
    r"TRAIGENT_MASTER_PASSWORD$|"
    r"TRAIGENT_JWT_TOKEN$"
    r")"
)


@dataclass(frozen=True)
class EnvVarName:
    name: str


def _parse_env_names(path: Path) -> list[EnvVarName]:
    names: list[EnvVarName] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        name, _ = line.split("=", 1)
        name = name.strip()
        if name:
            names.append(EnvVarName(name=name))
    return names


def _is_secret_name(name: str) -> bool:
    return DEFAULT_SECRET_NAME_PATTERN.fullmatch(name) is not None


def _secret_tool_lookup(*, project: str, name: str) -> str | None:
    try:
        result = subprocess.run(
            ["secret-tool", "lookup", "project", project, "name", name],
            check=False,
            capture_output=True,
            text=True,
        )
        value = (result.stdout or "").strip()
        return value or None
    except Exception:
        return None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Run a command with env vars loaded from GNOME Keyring."
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("TRAIGENT_KEYRING_PROJECT", "traigent"),
        help="Keyring attribute project name (default: traigent).",
    )
    parser.add_argument(
        "--env-file",
        action="append",
        default=[],
        help="A .env file to read variable NAMES from (repeatable).",
    )
    parser.add_argument(
        "--vars",
        action="append",
        default=[],
        help="Variable name to load (repeatable).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Load all variables from --env-file (not just likely-secrets).",
    )
    parser.add_argument(
        "--require",
        action="store_true",
        help="Fail if any requested variable can't be found (env or keyring).",
    )
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="Command to run (use --)")
    args = parser.parse_args(argv)

    if not args.cmd or args.cmd[0] != "--":
        print("error: expected command after `--`", file=sys.stderr)
        return 2
    cmd = args.cmd[1:]
    if not cmd:
        print("error: empty command", file=sys.stderr)
        return 2

    if shutil.which("secret-tool") is None:
        print("error: secret-tool not found.", file=sys.stderr)
        print("Install on Ubuntu: sudo apt install libsecret-tools", file=sys.stderr)
        return 2

    env_files = [Path(p).expanduser().resolve() for p in args.env_file]
    for p in env_files:
        if not p.exists():
            print(f"error: env file not found: {p}", file=sys.stderr)
            return 2

    requested_names: list[str] = []
    for p in env_files:
        requested_names.extend([x.name for x in _parse_env_names(p)])
    explicit_names = [v for v in args.vars if v.strip()]

    if not requested_names and not explicit_names:
        print(
            "error: no variables specified. Use --vars VAR or --env-file PATH.",
            file=sys.stderr,
        )
        return 2

    selected: set[str] = set(explicit_names)
    if args.all:
        selected.update([n for n in requested_names if n.strip()])
    else:
        selected.update([n for n in requested_names if _is_secret_name(n)])

    if not selected:
        print(
            "error: no variables matched selection. "
            "Use --all or pass explicit --vars VAR_NAME.",
            file=sys.stderr,
        )
        return 2

    child_env = dict(os.environ)
    missing: list[str] = []
    loaded: list[str] = []

    for name in sorted(selected):
        if child_env.get(name):
            loaded.append(name)
            continue
        value = _secret_tool_lookup(project=args.project, name=name)
        if value:
            child_env[name] = value
            loaded.append(name)
        else:
            missing.append(name)

    if missing and args.require:
        print(
            f"error: missing required secrets (env or keyring): {', '.join(sorted(missing))}",
            file=sys.stderr,
        )
        return 1

    # Do not print secret values.
    if loaded:
        print(f"Loaded {len(loaded)} env var(s) from env/keyring.")
    if missing:
        print(
            f"Warning: {len(missing)} env var(s) not found (skipped): {', '.join(sorted(missing))}",
            file=sys.stderr,
        )

    completed = subprocess.run(cmd, env=child_env)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
