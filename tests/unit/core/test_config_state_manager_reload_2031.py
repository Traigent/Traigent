"""Issue #2031: `ConfigStateManager.load_optimization_results` must restore the run.

The writer already had full fidelity — it dumps ``asdict(result)``, so every
field is on disk. The loader was the lossy half: it rebuilt the dataclass with a
constructor call naming 11 of the 27 fields, so a saved run came back with
``total_cost=None``, no ``stop_reason``, no ``warnings``, no cloud references and
``source == "backend"`` even when the JSON right there said ``"local"``.

These tests drive their assertions off the manifest in
``traigent/utils/optimization_result_persistence.py`` rather than a hand-written
field list, so they cannot drift from it. Fields are compared one at a time,
never ``saved == loaded``: ``_experiment_stats`` participates in the dataclass's
``__eq__`` and is deliberately not restored.

Cost safety: no ``OptimizedFunction``, no ``optimize()``, no evaluator — pure
save/load over ``tmp_path``. No LLM call, no network, no spend.
"""
# Traceability: CONC-Layer-Core CONC-Quality-Reliability FUNC-ORCH-LIFECYCLE REQ-ORCH-003

from __future__ import annotations

import copy
import dataclasses
import json
import logging

import pytest

from traigent.api.types import OptimizationResult, OptimizationStatus
from traigent.core.config_state_manager import ConfigStateManager, OptimizationState
from traigent.utils.optimization_result_persistence import (
    RESULT_RESET,
    RESULT_RESTORE,
    RESULT_SCHEMA_VERSION,
    SCHEMA_VERSION_KEY,
    _SENTINELS,
)

_LOGGER_NAME = "traigent.utils.optimization_result_persistence"


def _manager(tmp_path) -> ConfigStateManager:
    def _fn(x):
        return x

    return ConfigStateManager(
        func=_fn,
        default_config={"model": "cheap"},
        local_storage_path=str(tmp_path / "store"),
        configuration_space={"model": ["cheap", "smart"]},
        auto_load_best=False,
        load_from=None,
        setup_wrapper_callback=lambda: None,
    )


def _sentinel_result() -> OptimizationResult:
    return OptimizationResult(**copy.deepcopy(_SENTINELS))


def _save(tmp_path, result: OptimizationResult, name: str = "results.json"):
    path = tmp_path / name
    saver = _manager(tmp_path)
    saver._optimization_results = result
    saver.save_optimization_results(str(path))
    return path


def _load(tmp_path, path) -> OptimizationResult:
    loader = _manager(tmp_path)
    loader.load_optimization_results(str(path))
    loaded = loader._optimization_results
    assert loaded is not None
    return loaded


@pytest.fixture
def round_trip(tmp_path):
    """Save a fully-populated result and load it back through a fresh manager."""
    path = _save(tmp_path, _sentinel_result())
    return _load(tmp_path, path)


def test_every_restored_field_survives_the_round_trip(round_trip) -> None:
    """The #2031 fix: 25 restorable fields, not 11.

    Before the fix this failed on 16 of them at once — ``source`` came back
    ``"backend"`` for a run saved as ``"local"``, and ``total_cost``,
    ``total_tokens``, ``metrics``, ``stop_reason``, ``reason_code``,
    ``experiment_id``, ``experiment_run_id``, ``cloud_url``, ``run_label``,
    ``warnings``, ``warning_codes``, ``best_config_margin`` and
    ``preset_selection`` all came back at their defaults.
    """
    for name in sorted(RESULT_RESTORE - {"trials"}):
        assert getattr(round_trip, name) == _SENTINELS[name], (
            f"{name} was not restored by ConfigStateManager.load_optimization_results"
        )

    assert [trial.trial_id for trial in round_trip.trials] == ["trial-sentinel-0"]


def test_every_reset_field_comes_back_at_its_default(round_trip) -> None:
    """Non-restoration is the other half of the manifest, and is also asserted."""
    defaults = {
        field.name: (
            field.default_factory()
            if field.default_factory is not dataclasses.MISSING
            else field.default
        )
        for field in dataclasses.fields(OptimizationResult)
    }

    for name in sorted(RESULT_RESET):
        assert getattr(round_trip, name) == defaults[name]


def test_restoring_does_not_inject_anything_into_the_warning_channels(
    round_trip,
) -> None:
    """``warnings`` / ``warning_codes`` are the user's, not the loader's.

    They round-trip verbatim: no "this artifact was lossy" entry is added, which
    would both corrupt the #2026 verbatim round-trip pin and surface a storage
    detail as a money-correctness warning (#1407).
    """
    assert round_trip.warnings == _SENTINELS["warnings"]
    assert round_trip.warning_codes == _SENTINELS["warning_codes"]


def test_warning_codes_clamp_survives_the_round_trip(round_trip) -> None:
    """``OBJECTIVE_UNMATCHED`` must keep forcing ``success_rate`` to 0.0.

    ``success_rate`` reads ``warning_codes`` (types.py). Dropping the codes on
    load silently disarmed the clamp, so a reloaded run whose objective never
    matched reported a 100% success rate. The sentinel run has one COMPLETED
    trial, so an unclamped value would be 1.0.
    """
    assert "OBJECTIVE_UNMATCHED" in round_trip.warning_codes
    assert round_trip.success_rate == 0.0


def test_experiment_stats_cache_is_reset_and_recomputes_equal(tmp_path) -> None:
    """The memo cache is not restored — it is recomputed from ``trials``."""
    result = _sentinel_result()
    original_stats = result.experiment_stats  # populate the cache before asdict()
    assert original_stats is not None

    path = _save(tmp_path, result)
    assert json.loads(path.read_text(encoding="utf-8"))["_experiment_stats"] is not None

    loaded = _load(tmp_path, path)

    assert loaded._experiment_stats is None
    assert loaded.experiment_stats == original_stats


def test_saved_artifact_declares_the_schema_version(tmp_path) -> None:
    """The version is what lets the loader call a missing field corruption.

    Read off the manifest, not hardcoded: a hardcoded number here fails on every
    bump — including a correct one — and teaches whoever is adding a field that
    the way to get back to green is to edit the expected number, which is the
    one step that does nothing for the artifacts already on disk.
    """
    path = _save(tmp_path, _sentinel_result())

    saved = json.loads(path.read_text(encoding="utf-8"))[SCHEMA_VERSION_KEY]
    assert saved == RESULT_SCHEMA_VERSION


def test_versioned_artifact_missing_a_field_raises_rather_than_defaulting(
    tmp_path,
) -> None:
    """A truncated post-#2031 artifact is corruption, not an older format."""
    path = _save(tmp_path, _sentinel_result())
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["total_cost"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(Exception, match="total_cost"):
        _load(tmp_path, path)


def test_an_explicit_null_schema_version_is_rejected_not_read_as_legacy(
    tmp_path,
) -> None:
    """``"_schema_version": null`` is an invalid declaration, not an absent one.

    Reading it as "unversioned" routes a damaged versioned artifact down the
    legacy path, where a missing field is expected and quietly defaulted — so
    one hand-edited value turns the strict reader back into the blanket
    ``.get(name, default)`` that #2031 removed. Here ``stop_reason`` is missing
    too: under the legacy reading it would come back ``None`` and the load would
    succeed.
    """
    path = _save(tmp_path, _sentinel_result())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[SCHEMA_VERSION_KEY] = None
    del payload["stop_reason"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(Exception, match=SCHEMA_VERSION_KEY):
        _load(tmp_path, path)


def test_legacy_artifact_loads_with_defaults_and_logs_what_was_lost(
    tmp_path, caplog
) -> None:
    """A pre-#2031 artifact must still load — and must say what it could not give.

    The assertion under test is the *presence of the signal*: before the fix such
    a file loaded silently and the caller had no way to tell a real ``None`` from
    a dropped field.
    """
    path = _save(tmp_path, _sentinel_result())
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload[SCHEMA_VERSION_KEY]
    for name in ("total_cost", "total_tokens", "warnings", "warning_codes", "source"):
        del payload[name]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        loaded = _load(tmp_path, path)

    assert loaded.optimization_id == _SENTINELS["optimization_id"]
    assert loaded.total_cost is None
    assert loaded.warnings == []
    # Captain decision 1: never "backend" for an artifact that never said so.
    assert loaded.source == "unknown"
    for name in ("total_cost", "total_tokens", "warnings", "warning_codes"):
        assert name in caplog.text


@pytest.mark.parametrize("bad_timestamp", [None, "not-a-date", 1773584422])
def test_the_writer_refuses_what_its_own_loader_refuses(
    tmp_path, bad_timestamp
) -> None:
    """Stamping the schema version is a promise the writer has to keep.

    ``SCHEMA_VERSION_KEY`` tells the reader "every restorable field is present
    and readable, so a missing one is corruption". This writer dumps the whole
    dataclass, and before the fix it stamped that promise without running the
    encoder: ``asdict`` + ``json.dump(default=str)`` wrote ``timestamp: None``
    happily, and ``load_optimization_results`` then refused the file it had
    just written. The failure surfaced at read time, on data already on disk,
    arbitrarily far from the caller that caused it — and the artifact was a
    total loss, because nothing had validated it while the caller was still
    around to be told.

    ``PersistenceManager.save_result`` already refused these (#2031); routing
    this writer through the same encoder is what makes that true of *both*
    writers rather than one.
    """
    result = _sentinel_result()
    result.timestamp = bad_timestamp  # type: ignore[assignment]

    path = tmp_path / "unwritable.json"
    saver = _manager(tmp_path)
    saver._optimization_results = result

    with pytest.raises(ValueError, match="timestamp"):
        saver.save_optimization_results(str(path))

    # And no artifact claiming a schema version it cannot honour is left behind.
    assert not path.exists()


def test_the_writer_refuses_a_status_its_loader_would_have_degraded(tmp_path) -> None:
    """Same asymmetry for ``status``, caught on the write instead of the read.

    The decoder is deliberately lenient here — an unrecognized member restores
    as ``UNKNOWN`` (#1302 AC3) rather than raising — so an unvalidated write
    did not fail loudly on load either: it silently downgraded the run's
    outcome. Refusing the write is what keeps a caller bug from becoming a
    quietly wrong artifact.
    """
    result = _sentinel_result()
    result.status = "bogus"  # type: ignore[assignment]

    path = tmp_path / "unwritable-status.json"
    saver = _manager(tmp_path)
    saver._optimization_results = result

    with pytest.raises(ValueError, match="bogus"):
        saver.save_optimization_results(str(path))

    assert not path.exists()


def test_everything_this_writer_stamps_is_readable_by_its_own_loader(
    tmp_path, round_trip
) -> None:
    """The invariant behind the two refusals above, stated positively."""
    path = _save(tmp_path, _sentinel_result(), name="readable.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload[SCHEMA_VERSION_KEY] == RESULT_SCHEMA_VERSION
    # The encoder normalizes the timestamp; `default=str` used to write the
    # space-separated `str(datetime)` form. Both decode to the same instant,
    # and the round_trip fixture proves this one does.
    assert payload["timestamp"] == _SENTINELS["timestamp"].isoformat()
    assert payload["status"] == "cancelled"
    assert round_trip.timestamp == _SENTINELS["timestamp"]


def test_the_writer_still_dumps_the_whole_dataclass(tmp_path) -> None:
    """Routing through the encoder must not narrow this format to the payload.

    ``ConfigStateManager``'s artifact has always been a whole-dataclass dump —
    trials inline, RESET fields included (the *loader* is what drops them,
    #2020) — and readers of the raw JSON exist. The encoder re-encodes three
    fields; it does not re-shape the file.
    """
    path = _save(tmp_path, _sentinel_result(), name="shape.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["trials"][0]["trial_id"] == "trial-sentinel-0"
    assert "sync_session_id" in payload
    assert "_experiment_stats" in payload
    assert payload["metadata"] == _SENTINELS["metadata"]
    # Nested dataclasses still arrive as dicts, not as `str(obj)`.
    assert payload["preset_selection"]["selected_configs"] == [
        {"model": "cheap"},
        {"model": "smart"},
    ]


def test_restored_status_still_drives_the_manager_state(tmp_path) -> None:
    """A non-COMPLETED status is now really restored, so check the state mapping.

    The loader previously read ``status`` already; #2031 does not change that,
    but it does mean CANCELLED reaches the state machine from more artifacts.
    ``OPTIMIZED`` is the documented fallback branch for any non-FAILED status.
    """
    result = _sentinel_result()
    assert result.status is OptimizationStatus.CANCELLED

    path = _save(tmp_path, result)
    loader = _manager(tmp_path)
    loader.load_optimization_results(str(path))

    assert loader._optimization_results is not None
    assert loader._optimization_results.status is OptimizationStatus.CANCELLED
    assert loader._state is OptimizationState.OPTIMIZED
