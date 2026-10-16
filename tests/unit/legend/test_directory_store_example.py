"""The documented example adapter passes the SDK's own conformance self-test (no network)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from traigent.legend import (
    CaptureIndex,
    ClientStore,
    ConfigurationSource,
    capture,
    conformance_check,
    restore,
)

_EXAMPLE = (
    Path(__file__).resolve().parents[3]
    / "examples/advanced/legend-client-store/directory_store.py"
)


@pytest.fixture
def store(tmp_path):
    spec = importlib.util.spec_from_file_location("legend_directory_store", _EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DirectoryStore(tmp_path / "store")


def test_example_store_satisfies_protocol_and_conformance(store):
    assert isinstance(store, ClientStore)
    outcome = conformance_check(store)
    assert outcome.passed, outcome


def test_example_store_round_trips_to_exact(store, tmp_path):
    idx = CaptureIndex()
    rec = capture(
        "versionref_" + "2" * 32,
        ConfigurationSource({"a": 1}),
        store=store,
        index=idx,
    )
    assert rec.recovery_state == "round_trip_verified"
    res = restore(rec, tmp_path / "out", store=store, index=idx)
    assert res.grade == "exact"


def test_example_store_never_overwrites_a_different_capture(store):
    import io

    key = "rmf0:sha256:" + "ab" * 32
    assert store.put("configuration", key, io.BytesIO(b"one")) == "stored_unverified"
    assert store.put("configuration", key, io.BytesIO(b"one")) == "stored_unverified"
    assert store.put("configuration", key, io.BytesIO(b"two")) == "store_key_conflict"
