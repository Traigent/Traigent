"""Cross-SDK conformance: grid enumeration via the public ``traigent.optimize``.

The fixture vendored under ``tests/parity/fixtures`` is the answer key (it comes
from TraigentSchema at the ref pinned in ``fixtures.lock.json``). This test
drives the public ``@traigent.optimize(...)`` decorator with ``algorithm="grid"``
fully offline and compares the configurations the user function was actually
invoked with against the fixture. It never imports the grid optimizer directly.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import traigent
from traigent.evaluators.base import Dataset, EvaluationExample

PARITY_DIR = Path(__file__).parent
LOCK_PATH = PARITY_DIR / "fixtures.lock.json"
FIXTURES_DIR = PARITY_DIR / "fixtures"
FIXTURE_ID = "optimizer.grid-enumeration.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lock_entry(lock: dict[str, Any], fixture_id: str) -> dict[str, str]:
    entry = lock["fixtures"][fixture_id]
    return {"path": entry["path"], "sha256": entry["sha256"]}


def _verified_fixture_path(
    lock_path: Path, fixtures_dir: Path, fixture_id: str
) -> Path:
    """Return the fixture path, failing (never skipping) on any lock violation."""
    assert lock_path.is_file(), f"fixture lock missing: {lock_path}"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    entry = _lock_entry(lock, fixture_id)
    path = fixtures_dir / entry["path"]
    assert path.is_file(), f"locked fixture missing: {path}"
    actual = _sha256(path)
    assert actual == entry["sha256"], (
        f"fixture {fixture_id} sha256 {actual} != locked {entry['sha256']}"
    )
    return path


def _canonical(config: dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


@pytest.fixture(scope="module")
def fixture() -> dict[str, Any]:
    path = _verified_fixture_path(LOCK_PATH, FIXTURES_DIR, FIXTURE_ID)
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    assert data["fixtureId"] == FIXTURE_ID
    return data


def test_lock_pins_schema_ref_and_repo() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    assert lock["schemaRepo"] == "Traigent/TraigentSchema"
    assert len(lock["schemaRef"]) == 40
    assert FIXTURE_ID in lock["fixtures"]


def test_missing_fixture_fails_not_skips(tmp_path: Path) -> None:
    lock = {
        "schemaRepo": "Traigent/TraigentSchema",
        "schemaRef": "0" * 40,
        "fixtures": {FIXTURE_ID: {"path": "absent.json", "sha256": "0" * 64}},
    }
    lock_path = tmp_path / "fixtures.lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(AssertionError, match="locked fixture missing"):
        _verified_fixture_path(lock_path, tmp_path, FIXTURE_ID)
    with pytest.raises(AssertionError, match="fixture lock missing"):
        _verified_fixture_path(tmp_path / "nope.json", tmp_path, FIXTURE_ID)


def test_sha_mismatch_fails(tmp_path: Path) -> None:
    (tmp_path / "f.json").write_text("{}", encoding="utf-8")
    lock = {"fixtures": {FIXTURE_ID: {"path": "f.json", "sha256": "0" * 64}}}
    lock_path = tmp_path / "fixtures.lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(AssertionError, match="sha256"):
        _verified_fixture_path(lock_path, tmp_path, FIXTURE_ID)


def test_grid_enumeration_via_public_optimize(
    fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))

    space = fixture["input"]["configurationSpace"]
    max_trials = fixture["input"]["maxTrials"]
    expected = fixture["expected"]["configurations"]
    assert fixture["expected"]["count"] == len(expected)

    invoked: list[dict[str, Any]] = []
    dataset = Dataset([EvaluationExample({"text": "a"}, "ok")], name="parity_grid")

    @traigent.optimize(
        eval_dataset=dataset,
        objectives=["accuracy"],
        configuration_space={k: list(v) for k, v in space.items()},
        injection_mode="parameter",
        offline=True,
        max_trials=max_trials,
        algorithm="grid",
    )
    def stub(text: str, config: dict[str, Any]) -> str:
        invoked.append({k: config[k] for k in space})
        return "ok"

    result = asyncio.run(stub.optimize())

    raw = [_canonical(c) for c in invoked]
    assert len(raw) == fixture["expected"]["count"], raw
    assert len(set(raw)) == len(raw), "duplicate configurations enumerated"
    assert set(raw) == {_canonical(c) for c in expected}
    assert len(raw) <= max_trials
    assert len(result.trials) == len(raw)


def test_grid_max_trials_truncates_via_public_optimize(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))

    space = {"model": ["model-a", "model-b"], "top_k": [1, 3, 5]}
    full_product = {
        _canonical({"model": m, "top_k": k})
        for m in space["model"]
        for k in space["top_k"]
    }
    assert len(full_product) == 6
    max_trials = 4

    invoked: list[dict[str, Any]] = []
    dataset = Dataset([EvaluationExample({"text": "a"}, "ok")], name="parity_grid_cap")

    @traigent.optimize(
        eval_dataset=dataset,
        objectives=["accuracy"],
        configuration_space={k: list(v) for k, v in space.items()},
        injection_mode="parameter",
        offline=True,
        max_trials=max_trials,
        algorithm="grid",
    )
    def stub(text: str, config: dict[str, Any]) -> str:
        invoked.append({k: config[k] for k in space})
        return "ok"

    result = asyncio.run(stub.optimize())

    raw = [_canonical(c) for c in invoked]
    assert len(raw) == max_trials, raw
    assert len(set(raw)) == len(raw), "duplicate configurations enumerated"
    assert set(raw) <= full_product
    assert len(result.trials) == max_trials
