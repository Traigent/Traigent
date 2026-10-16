"""The compact content-class table is generated from TraigentSchema and pinned."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
TABLE = ROOT / "traigent" / "labels" / "content_class_table.json"
SCHEMA_WT = Path(
    "/home/nimrod/TraigentProjects/core_project/worktrees/TraigentSchema/content-less"
)
PINNED_SCHEMA_COMMIT = "9af17d84b0f3a202fdc12cc4bde082e6295a71d9"
PINNED_MANIFEST_SHA256 = (
    "197ccd7e7d712938b716f416588338f6ee44ede6cde15a609584683a328a7a71"
)
PINNED_TABLE_SHA256 = "2ec9f9825cbc72d111aa98c491ef150c230cb09c94f12f7dc2524bb56c0a2fa5"


def test_table_sha256_and_provenance_are_pinned() -> None:
    assert hashlib.sha256(TABLE.read_bytes()).hexdigest() == PINNED_TABLE_SHA256
    src = json.loads(TABLE.read_text("utf-8"))["source"]
    assert src == {
        "schema_commit": PINNED_SCHEMA_COMMIT,
        "manifest_sha256": PINNED_MANIFEST_SHA256,
    }
    assert TABLE.stat().st_size < 100_000  # compact: no schema files shipped
    assert not (ROOT / "traigent" / "labels" / "contract_schemas").exists()


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        ["git", "-C", str(SCHEMA_WT), *args],  # noqa: S607
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.mark.skipif(not SCHEMA_WT.exists(), reason="TraigentSchema worktree absent")
def test_manifest_at_the_pinned_schema_commit_matches_the_pin() -> None:
    shown = subprocess.run(  # noqa: S603
        [
            "git",
            "-C",
            str(SCHEMA_WT),
            "show",
            f"{PINNED_SCHEMA_COMMIT}:docs/content-class-manifest.json",
        ],  # noqa: S607
        capture_output=True,
        timeout=30,
        check=False,
    )
    if shown.returncode:
        pytest.skip("pinned Schema commit not present in this checkout")
    assert hashlib.sha256(shown.stdout).hexdigest() == PINNED_MANIFEST_SHA256


@pytest.mark.skipif(not SCHEMA_WT.exists(), reason="TraigentSchema worktree absent")
def test_table_regenerates_from_the_schema_worktree_when_it_is_at_the_pin() -> None:
    manifest = SCHEMA_WT / "docs" / "content-class-manifest.json"
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != PINNED_MANIFEST_SHA256:
        pytest.skip(
            "Schema worktree moved past the pinned manifest; regenerate and re-pin"
        )
    spec = importlib.util.spec_from_file_location(
        "gen_cct", ROOT / "scripts" / "generate_content_class_table.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fresh = mod.build(SCHEMA_WT)
    shipped = json.loads(TABLE.read_text("utf-8"))
    assert fresh["nodes"] == shipped["nodes"] and fresh["routes"] == shipped["routes"]
