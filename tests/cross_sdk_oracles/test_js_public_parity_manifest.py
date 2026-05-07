from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import traigent

PYTHON_DEVELOP_SHA = "f9f1adcb19ea8c3874bb1b7ef21f6d11b1b95a18"  # pragma: allowlist secret


def _enterprise_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_parity_manifest() -> dict:
    manifest_override = os.environ.get("TRAIGENT_PARITY_MANIFEST")
    manifest_path = (
        Path(manifest_override)
        if manifest_override
        else _enterprise_root() / "TraigentSchema" / "parity" / "python-js-sdk.json"
    )
    assert manifest_path.exists(), f"Missing Python/JS parity manifest at {manifest_path}"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _load_js_surface() -> dict:
    snapshot_override = os.environ.get("TRAIGENT_JS_API_SURFACE_SNAPSHOT")
    snapshot_path = (
        Path(snapshot_override)
        if snapshot_override
        else _enterprise_root()
        / "traigent-js"
        / "tests"
        / "integration"
        / "fixtures"
        / "api-surface.snapshot.json"
    )
    assert snapshot_path.exists(), f"Missing JS API surface snapshot at {snapshot_path}"
    return json.loads(snapshot_path.read_text(encoding="utf-8"))


def test_python_public_clients_are_represented_in_js_api_surface() -> None:
    manifest = _load_parity_manifest()
    js_surface = _load_js_surface()
    js_root = set(js_surface["root"])

    assert manifest["python"]["targetSha"] == PYTHON_DEVELOP_SHA
    required = set(manifest["javascript"]["requiredRootExports"])
    missing = sorted(required - js_root)

    assert missing == [], (
        "JS root exports do not satisfy the schema-owned Python/JS parity manifest: "
        f"{missing}"
    )


def test_python_public_root_symbols_are_classified_in_parity_manifest() -> None:
    manifest = _load_parity_manifest()
    classified = {
        symbol
        for symbols in manifest["classifications"].values()
        for symbol in symbols
    }
    missing = sorted(set(traigent.__all__) - classified)
    if missing and os.environ.get("CI") != "true":
        pytest.skip(
            "Local Python checkout exports symbols outside the pinned parity "
            f"manifest target: {missing}"
        )

    assert missing == [], (
        "Python exported root symbols are missing from the schema-owned "
        f"Python/JS parity manifest: {missing}"
    )
