#!/usr/bin/env python3
"""Import secrets from a .env file into GNOME Keyring (Secret Service).

This is a dev-convenience tool for Ubuntu 24.04 (and other Linux desktops)
using `secret-tool` (libsecret).

It stores each secret under attributes:
  project=<project>  name=<ENV_VAR_NAME>

Example:
  ./tools/keyring_import_env.py --env-file .env --project traigent
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
class EnvVar:
    name: str
    value: str


def _parse_env_file(path: Path) -> list[EnvVar]:
    """Parse KEY=VALUE pairs from a .env file (best-effort).

    Prefer python-dotenv if available (handles quotes/escapes well); otherwise use a
    minimal parser that supports simple quoted values.
    """
    try:
        from dotenv import dotenv_values  # type: ignore

        values = dotenv_values(path)
        parsed: list[EnvVar] = []
        for key, value in values.items():
            if not key or value is None:
                continue
            parsed.append(EnvVar(name=str(key), value=str(value)))
        return parsed
    except Exception:
        pass

    parsed: list[EnvVar] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export ") :].lstrip()

        if "=" not in line:
            continue

        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name:
            continue

        # Strip simple surrounding quotes.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        parsed.append(EnvVar(name=name, value=value))

    return parsed


def _is_secret_name(name: str, *, include_all: bool) -> bool:
    if include_all:
        return True
    return DEFAULT_SECRET_NAME_PATTERN.fullmatch(name) is not None


def _store_secret(*, project: str, name: str, value: str) -> None:
    # Never pass the secret via argv; stdin is safer (not visible in `ps` output).
    subprocess.run(
        [
            "secret-tool",
            "store",
            "--label",
            f"Traigent ({project}) {name}",
            "project",
            project,
            "name",
            name,
        ],
        input=value,
        text=True,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def _iter_selected_vars(
    env_vars: Iterable[EnvVar],
    *,
    include_all: bool,
    include: set[str],
    exclude: set[str],
) -> list[EnvVar]:
    selected: list[EnvVar] = []
    for item in env_vars:
        if item.name in exclude:
            continue
        if include and item.name not in include:
            continue
        if not _is_secret_name(item.name, include_all=include_all):
            continue
        if item.value.strip() == "":
            continue
        selected.append(item)
    return sorted(selected, key=lambda x: x.name)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Import secrets from a .env file into GNOME Keyring (secret-tool)."
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file (default: .env).",
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("TRAIGENT_KEYRING_PROJECT", "traigent"),
        help="Keyring attribute project name (default: traigent).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Store all non-empty variables (not just likely-secrets).",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Store only these variable names (repeatable).",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Skip these variable names (repeatable).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be stored; do not write to keyring.",
    )
    args = parser.parse_args(argv)

    if shutil.which("secret-tool") is None:
        print("error: secret-tool not found.", file=sys.stderr)
        print("Install on Ubuntu: sudo apt install libsecret-tools", file=sys.stderr)
        return 2

    env_path = Path(args.env_file).expanduser().resolve()
    if not env_path.exists():
        print(f"error: env file not found: {env_path}", file=sys.stderr)
        return 2

    include = set(args.include)
    exclude = set(args.exclude)

    env_vars = _parse_env_file(env_path)
    selected = _iter_selected_vars(
        env_vars,
        include_all=bool(args.all),
        include=include,
        exclude=exclude,
    )

    if not selected:
        print(
            "No variables selected to store. "
            "Use --all or --include VAR_NAME to force.",
            file=sys.stderr,
        )
        return 1

    print(f"Env file: {env_path}")
    print(f"Project: {args.project}")
    print(f"Selected: {len(selected)} variable(s)")

    if args.dry_run:
        for item in selected:
            print(f"- {item.name} (len={len(item.value)})")
        return 0

    failures: list[str] = []
    for item in selected:
        try:
            _store_secret(project=args.project, name=item.name, value=item.value)
            print(f"Stored: {item.name}")
        except subprocess.CalledProcessError as e:
            failures.append(item.name)
            # Do not print secrets; just the stderr from secret-tool.
            stderr = (e.stderr or b"").decode("utf-8", errors="replace").strip()
            print(f"Failed: {item.name}: {stderr}", file=sys.stderr)

    if failures:
        print(
            f"Completed with failures ({len(failures)}): {', '.join(failures)}",
            file=sys.stderr,
        )
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

