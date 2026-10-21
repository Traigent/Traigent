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


def _tamper(tmp_path, name, inner):
    """Edit the inner best_config in metadata.json and drop the manifest."""
    f = tmp_path / name / "metadata.json"
    meta = json.loads(f.read_text())
    meta["result_fields"]["best_config"] = inner
    f.write_text(json.dumps(meta))
    (tmp_path / name / MANIFEST).unlink()


def test_invalid_inner_best_config_rejected_even_if_outer_valid(mgr, tmp_path):
    mgr.save_result(_result(), name="r")
    _tamper(tmp_path, "r", {"model": "SECRET-INNER"})
    with pytest.raises(ValueError) as e:
        mgr.load_result("r")
    assert "SECRET-INNER" not in str(e.value)


def test_outer_inner_best_config_disagree_rejected_value_free(mgr, tmp_path):
    mgr.save_result(_result(), name="r")
    _tamper(tmp_path, "r", {"model": "smart"})  # valid but differs
    with pytest.raises(ValueError, match="disagree") as e:
        mgr.load_result("r")
    assert "smart" not in str(e.value)


def test_numeric_list_is_choices_not_interval(mgr):
    mgr.save_result(_result(best_config={"n": 5}, space={"n": [1, 10]}), name="r")
    with pytest.raises(ValueError, match="outside"):
        mgr.load_result("r")
    mgr.save_result(_result(best_config={"n": 10}, space={"n": [1, 10]}), name="ok")
    assert mgr.load_result("ok").best_config == {"n": 10}


def test_tuple_range_survives_save_as_range(mgr):
    mgr.save_result(_result(best_config={"t": 0.5}, space={"t": (0.0, 1.0)}), name="r")
    assert mgr.load_result("r").best_config == {"t": 0.5}
    mgr.save_result(_result(best_config={"t": 2.0}, space={"t": (0.0, 1.0)}), name="b")
    with pytest.raises(ValueError, match="outside"):
        mgr.load_result("b")


def test_string_against_numeric_bounds_rejected(mgr):
    mgr.save_result(
        _result(best_config={"t": "0.5"}, space={"t": {"low": 0.0, "high": 1.0}}),
        name="r",
    )
    with pytest.raises(ValueError, match="outside"):
        mgr.load_result("r")


def test_unrecognised_space_shape_skips_and_logs(mgr, caplog):
    mgr.save_result(_result(best_config={"t": 1}, space={"t": {"weird": 1}}), name="r")
    with caplog.at_level(logging.WARNING):
        assert mgr.load_result("r").best_config == {"t": 1}
    assert any("check skipped" in r.getMessage() for r in caplog.records)


def test_legacy_save_list_range_never_rejected(mgr, tmp_path):
    # Pre-change saves stored a tuple range (0.0, 1.0) as the list [0.0, 1.0].
    mgr.save_result(_result(best_config={"t": 0.5}, space={"t": [0.0, 1.0]}), name="r")
    (tmp_path / "r" / MANIFEST).unlink()  # legacy: no manifest
    assert mgr.load_result("r").best_config == {"t": 0.5}
    mgr.save_result(_result(best_config={"t": 7.0}, space={"t": [0.0, 1.0]}), name="o")
    (tmp_path / "o" / MANIFEST).unlink()
    with pytest.raises(ValueError, match="outside"):
        mgr.load_result("o")


def test_tuple_range_survives_two_save_load_rounds(mgr):
    mgr.save_result(_result(best_config={"t": 0.5}, space={"t": (0, 1)}), name="a")
    first = mgr.load_result("a")
    assert first.metadata["configuration_space"]["t"] == (0, 1)
    mgr.save_result(first, name="b")
    second = mgr.load_result("b")
    assert second.best_config == {"t": 0.5}
    assert second.metadata["configuration_space"]["t"] == (0, 1)


def test_choice_list_still_rejects_between_values_across_rounds(mgr):
    mgr.save_result(_result(best_config={"t": 0.5}, space={"t": [0, 1]}), name="a")
    with pytest.raises(ValueError, match="outside"):
        mgr.load_result("a")


@pytest.mark.parametrize(
    "bad", ["../x", "/etc/passwd", "sub/x", "..\\x", "unexpected.json"]
)
def test_manifest_entry_outside_allowlist_rejected_without_echo(mgr, tmp_path, bad):
    mgr.save_result(_result(), name="r")
    mpath = tmp_path / "r" / MANIFEST
    payload = json.loads(mpath.read_text())
    payload["sha256"][bad] = "0" * 64
    mpath.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="not allowed") as exc:
        mgr.load_result("r")
    assert bad not in str(exc.value)
    assert str(tmp_path) not in str(exc.value)


def test_write_manifest_rejects_non_allowlisted_name(tmp_path):
    from traigent.utils.file_versioning import write_integrity_manifest

    (tmp_path / "other.txt").write_text("x")
    with pytest.raises(ValueError, match="not allowed"):
        write_integrity_manifest(tmp_path, ["other.txt"])
    with pytest.raises(ValueError, match="not allowed"):
        write_integrity_manifest(tmp_path, ["../other.txt"])
