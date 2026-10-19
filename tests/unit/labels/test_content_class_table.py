"""The compact content-class table is generated from TraigentSchema and pinned.

CI/local setup for exact regeneration: check out TraigentSchema at the pinned commit
and export its path, e.g.

    TRAIGENT_SCHEMA_DIR=/path/to/TraigentSchema pytest tests/unit/labels/test_content_class_table.py

When the variable is set the tests REQUIRE that the checkout's manifest is the pinned
one and that regenerating gives exactly the shipped table (they fail, never skip).
When it is unset only the sha256/provenance pin runs.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
TABLE = ROOT / "traigent" / "labels" / "content_class_table.json"
SCHEMA_DIR = os.environ.get("TRAIGENT_SCHEMA_DIR")
PINNED_SCHEMA_COMMIT = "31e0ddf818766d4570a58b3601b99b70dcc4c46e"
PINNED_MANIFEST_SHA256 = (
    "f3ed961e9674b660b38a5e1fec706bab1a10cd547addf59c8369ad7081040c1b"
)
PINNED_TABLE_SHA256 = "cd715b40b5262e7c8ad878ccbc0b91d14ff9494e75546e41adaac25758a11ce5"


def test_table_sha256_and_provenance_are_pinned() -> None:
    assert hashlib.sha256(TABLE.read_bytes()).hexdigest() == PINNED_TABLE_SHA256
    src = json.loads(TABLE.read_text("utf-8"))["source"]
    assert src == {
        "schema_commit": PINNED_SCHEMA_COMMIT,
        "manifest_sha256": PINNED_MANIFEST_SHA256,
    }
    assert TABLE.stat().st_size < 100_000  # compact: no schema files shipped
    assert not (ROOT / "traigent" / "labels" / "contract_schemas").exists()


@pytest.mark.skipif(not SCHEMA_DIR, reason="TRAIGENT_SCHEMA_DIR not set")
def test_schema_checkout_manifest_is_the_pinned_one() -> None:
    manifest = Path(SCHEMA_DIR or "") / "docs" / "content-class-manifest.json"
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == PINNED_MANIFEST_SHA256


@pytest.mark.skipif(not SCHEMA_DIR, reason="TRAIGENT_SCHEMA_DIR not set")
def test_table_regenerates_exactly_from_the_schema_checkout() -> None:
    spec = importlib.util.spec_from_file_location(
        "gen_cct", ROOT / "scripts" / "generate_content_class_table.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.render(Path(SCHEMA_DIR or ""), PINNED_SCHEMA_COMMIT) == TABLE.read_text(
        "utf-8"
    )


def _generator():
    spec = importlib.util.spec_from_file_location(
        "gen_cct_merge", ROOT / "scripts" / "generate_content_class_table.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_allof_constraint_merge_keeps_the_stricter_pattern_and_bounds() -> None:
    mod = _generator()
    out = {"pt": "^[A-Za-z0-9_-]+$", "ml": 40, "mn": 1}
    mod._merge_constraints(
        out, {"pattern": "^[0-9a-f]{32}$", "maxLength": 32, "minLength": 32}
    )
    assert out == {"pt": "^[0-9a-f]{32}$", "ml": 32, "mn": 32}


def test_allof_constraint_merge_that_would_lose_a_pattern_fails_generation() -> None:
    mod = _generator()
    with pytest.raises(SystemExit):
        mod._merge_constraints({"pt": "^a+$"}, {"pattern": "^b+$"})
