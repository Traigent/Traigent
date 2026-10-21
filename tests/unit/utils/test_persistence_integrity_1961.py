"""#1961: sha256 manifest verified at load; best_config checked against space."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest

from traigent.api.types import OptimizationResult, OptimizationStatus
from traigent.utils.persistence import PersistenceManager

MANIFEST = "integrity.sha256.json"


def _result(best_config=None, space=None) -> OptimizationResult:
    return OptimizationResult(
        trials=[],
        best_config=best_config if best_config is not None else {"model": "cheap"},
        best_score=0.5,
        optimization_id="opt-1961",
        duration=1.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="grid",
        timestamp=datetime.now(UTC),
        metadata={
            "function_name": "fn",
            "configuration_space": (
                space if space is not None else {"model": ["cheap", "smart"]}
            ),
        },
    )


@pytest.fixture
def mgr(tmp_path: Path) -> PersistenceManager:
    return PersistenceManager(str(tmp_path))


def test_round_trip_writes_manifest_and_loads(mgr, tmp_path):
    mgr.save_result(_result(), name="r")
    assert (tmp_path / "r" / MANIFEST).exists()
    assert mgr.load_result("r").best_config == {"model": "cheap"}


def test_flipped_byte_in_trials_raises(mgr, tmp_path):
    mgr.save_result(_result(), name="r")
    f = tmp_path / "r" / "trials.json.gz"
    data = bytearray(f.read_bytes())
    data[-5] ^= 0xFF
    f.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="integrity") as exc:
        mgr.load_result("r")
    assert str(tmp_path) not in str(exc.value)


def test_edited_metadata_raises(mgr, tmp_path):
    mgr.save_result(_result(), name="r")
    f = tmp_path / "r" / "metadata.json"
    meta = json.loads(f.read_text())
    meta["best_score"] = 0.99
    f.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="integrity"):
        mgr.load_result("r")


def test_unreadable_manifest_raises(mgr, tmp_path):
    mgr.save_result(_result(), name="r")
    (tmp_path / "r" / MANIFEST).write_text("{not json")
    with pytest.raises(ValueError, match="unreadable"):
        mgr.load_result("r")


def test_listed_file_missing_raises(mgr, tmp_path):
    mgr.save_result(_result(), name="r")
    (tmp_path / "r" / "trials.json.gz").unlink()
    with pytest.raises(ValueError, match="missing"):
        mgr.load_result("r")


def test_manifest_not_covering_files_raises(mgr, tmp_path):
    mgr.save_result(_result(), name="r")
    (tmp_path / "r" / MANIFEST).write_text(json.dumps({"sha256": {}}))
    with pytest.raises(ValueError, match="cover"):
        mgr.load_result("r")


def test_missing_manifest_is_legacy_warns_once_and_loads(mgr, tmp_path, caplog):
    mgr.save_result(_result(), name="r")
    (tmp_path / "r" / MANIFEST).unlink()
    with caplog.at_level(logging.WARNING):
        loaded = mgr.load_result("r")
    assert loaded.best_config == {"model": "cheap"}
    warnings = [r for r in caplog.records if "integrity manifest" in r.getMessage()]
    assert len(warnings) == 1


def test_best_config_outside_space_raises_value_free(mgr, tmp_path):
    mgr.save_result(_result(best_config={"model": "SECRET-VALUE"}), name="r")
    with pytest.raises(ValueError, match="outside the saved configuration space") as e:
        mgr.load_result("r")
    assert "SECRET-VALUE" not in str(e.value)


def test_best_config_range_checks(mgr):
    mgr.save_result(
        _result(best_config={"t": 0.5}, space={"t": {"low": 0.0, "high": 1.0}}),
        name="ok",
    )
    assert mgr.load_result("ok").best_config == {"t": 0.5}
    mgr.save_result(
        _result(best_config={"t": 7}, space={"t": {"low": 0.0, "high": 1.0}}),
        name="bad",
    )
    with pytest.raises(ValueError, match="outside"):
        mgr.load_result("bad")
