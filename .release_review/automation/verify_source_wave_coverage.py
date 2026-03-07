#!/usr/bin/env python3
"""Verify staged source-wave inventories against the current source tree."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable


def _read_inventory(path: Path) -> list[str]:
    entries: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line)
    return entries


def _iter_source_files(repo_root: Path) -> Iterable[str]:
    for root_name in ("traigent", "traigent_validation"):
        root = repo_root / root_name
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts or ".mypy_cache" in path.parts:
                continue
            yield path.relative_to(repo_root).as_posix()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-inventory",
        default=".release_review/inventories/source_files.txt",
        help="Repository-relative path to the canonical source inventory.",
    )
    parser.add_argument(
        "--inventories-dir",
        default=".release_review/inventories",
        help="Repository-relative directory containing priority wave inventories.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    source_inventory_path = repo_root / args.source_inventory
    inventories_dir = repo_root / args.inventories_dir
    wave_paths = sorted(inventories_dir.glob("priority_wave*.txt"))

    errors: list[str] = []
    source_entries: list[str] = []
    wave_entries: list[str] = []
    disk_entries: list[str] = []

    if not source_inventory_path.exists():
        errors.append(f"missing source inventory: {source_inventory_path}")
    else:
        source_entries = _read_inventory(source_inventory_path)
        disk_entries = list(_iter_source_files(repo_root))

    if not wave_paths:
        errors.append(f"no wave inventories found under {inventories_dir}")

    for wave_path in wave_paths:
        wave_entries.extend(_read_inventory(wave_path))

    source_counter = Counter(source_entries)
    wave_counter = Counter(wave_entries)

    duplicate_source = sorted(path for path, count in source_counter.items() if count > 1)
    duplicate_wave = sorted(path for path, count in wave_counter.items() if count > 1)

    if duplicate_source:
        errors.append(f"duplicate entries in source inventory: {len(duplicate_source)}")
    if duplicate_wave:
        errors.append(f"duplicate entries across wave inventories: {len(duplicate_wave)}")

    missing_from_source_inventory = sorted(set(disk_entries) - set(source_entries))
    extra_in_source_inventory = sorted(set(source_entries) - set(disk_entries))
    missing_from_waves = sorted(set(source_entries) - set(wave_entries))
    extra_in_waves = sorted(set(wave_entries) - set(source_entries))

    if missing_from_source_inventory:
        errors.append(
            f"disk source files missing from source inventory: {len(missing_from_source_inventory)}"
        )
    if extra_in_source_inventory:
        errors.append(
            f"source inventory entries not present on disk: {len(extra_in_source_inventory)}"
        )
    if missing_from_waves:
        errors.append(f"source inventory files missing from wave inventories: {len(missing_from_waves)}")
    if extra_in_waves:
        errors.append(f"wave inventory entries not present in source inventory: {len(extra_in_waves)}")

    print(f"disk_source_files={len(disk_entries)}")
    print(f"source_inventory_entries={len(source_entries)}")
    print(f"wave_inventory_entries={len(wave_entries)}")
    print(f"wave_inventory_files={len(wave_paths)}")

    for wave_path in wave_paths:
        count = len(_read_inventory(wave_path))
        print(f"{wave_path.relative_to(repo_root).as_posix()}={count}")

    if not errors:
        print("coverage_status=ok")
        return 0

    print("coverage_status=error")
    for error in errors:
        print(f"error={error}")

    if duplicate_source:
        for path in duplicate_source:
            print(f"duplicate_source={path}")
    if duplicate_wave:
        for path in duplicate_wave:
            print(f"duplicate_wave={path}")
    if missing_from_source_inventory:
        for path in missing_from_source_inventory:
            print(f"missing_from_source_inventory={path}")
    if extra_in_source_inventory:
        for path in extra_in_source_inventory:
            print(f"extra_in_source_inventory={path}")
    if missing_from_waves:
        for path in missing_from_waves:
            print(f"missing_from_waves={path}")
    if extra_in_waves:
        for path in extra_in_waves:
            print(f"extra_in_waves={path}")

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
