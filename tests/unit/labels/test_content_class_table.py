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
PINNED_SCHEMA_COMMIT = "f826711832068384858064999eecba0a1600a9cd"
PINNED_MANIFEST_SHA256 = (
    "b0b1f1751449c31c5545f4010ea1200a15774dff66022b91c0d51f0b71521279"
)
PINNED_TABLE_SHA256 = "e1baed0136f09ddb9b135669d1d576196c02b2fa53ad9643dd43bd2fe971a2d4"


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
