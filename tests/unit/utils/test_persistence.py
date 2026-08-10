"""Tests for persistence utilities."""

from __future__ import annotations

import builtins
import copy
import gzip
import hashlib
import io
import json
import logging
import os
import pickle
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from traigent.api.types import (
    OptimizationResult,
    OptimizationStatus,
    TrialError,
    TrialResult,
    TrialStatus,
)
from traigent.utils import optimization_result_persistence as manifest
from traigent.utils.optimization_result_persistence import (
    _SENTINELS,
    RESULT_RESET,
    RESULT_RESTORE,
    RESULT_SCHEMA_VERSION,
    SCHEMA_VERSION_KEY,
)
from traigent.utils.persistence import (
    PersistenceManager,
    RestrictedUnpickler,
    ResumableOptimization,
)


def _make_optimization_result() -> OptimizationResult:
    """Helper to build a minimal optimization result."""
    timestamp = datetime.now(UTC)
    return OptimizationResult(
        trials=[],
        best_config={"param": 1},
        best_score=0.0,
        optimization_id="opt-123",
        duration=3.0,
        convergence_info={"status": "stable"},
        status=OptimizationStatus.COMPLETED,
        objectives=["objective"],
        algorithm="grid_search",
        timestamp=timestamp,
        metadata={
            "function_name": "demo_function",
            "function_slug": "demo-function",
            "configuration_space": {"param": [0, 1]},
        },
    )


def _load_restricted(payload: bytes) -> object:
    return RestrictedUnpickler(io.BytesIO(payload)).load()


def test_get_result_hash_uses_metadata(tmp_path) -> None:
    """get_result_hash should use metadata-backed attributes when direct ones are absent."""
    persistence = PersistenceManager(base_dir=tmp_path)
    result = _make_optimization_result()

    generated_hash = persistence.get_result_hash(result)

    expected_payload = {
        "function_name": "demo_function",
        "algorithm": "grid_search",
        "objectives": ["objective"],
        "configuration_space": {"param": [0, 1]},
        "trial_count": 0,
    }
    expected_hash = hashlib.sha256(
        json.dumps(expected_payload, sort_keys=True).encode()
    ).hexdigest()[:12]

    assert generated_hash == expected_hash


def test_resumable_load_checkpoint_roundtrip(tmp_path) -> None:
    """ResumableOptimization.load_checkpoint should delegate to load_result."""
    persistence = PersistenceManager(base_dir=tmp_path)
    resumable = ResumableOptimization(persistence)
    result = _make_optimization_result()

    checkpoint_path = resumable.save_checkpoint(result, "initial")
    assert "checkpoint_initial" in checkpoint_path

    loaded = resumable.load_checkpoint("initial")
    assert isinstance(loaded, OptimizationResult)
    assert loaded.best_config == result.best_config
    assert loaded.best_score == result.best_score
    # #2031: metadata is now restored verbatim from the result's own record
    # instead of being rebuilt from the curated metadata.json keys, so
    # function_slug (and anything else the run put there) survives. The two
    # curated keys are still present — CLI apply / auto-load paths read them.
    assert loaded.metadata == {
        "function_name": "demo_function",
        "function_slug": "demo-function",
        "configuration_space": {"param": [0, 1]},
    }


def test_resumable_can_resume_finds_checkpoint(tmp_path) -> None:
    """can_resume should surface matching checkpoints based on metadata."""
    persistence = PersistenceManager(base_dir=tmp_path)
    resumable = ResumableOptimization(persistence)
    result = _make_optimization_result()

    resumable.save_checkpoint(result, "latest")

    resume_token = resumable.can_resume(
        function_name="demo_function", configuration_space={"param": [0, 1]}
    )
    assert resume_token == "checkpoint_latest"

    assert (
        resumable.can_resume(
            function_name="demo_function", configuration_space={"param": [1, 2]}
        )
        is None
    )


def test_restricted_unpickler_rejects_issue_eval_exploit(tmp_path) -> None:
    """The issue #1634 eval/REDUCE payload must not execute."""

    sentinel = tmp_path / "eval_executed"

    class Exploit:
        def __reduce__(self):
            code = f"__import__('pathlib').Path({str(sentinel)!r}).write_text('owned')"
            return eval, (code,)

    payload = pickle.dumps(Exploit(), protocol=2)

    with pytest.raises(pickle.UnpicklingError):
        _load_restricted(payload)

    assert not sentinel.exists()


@pytest.mark.parametrize(
    ("callable_name", "args"),
    [
        ("eval", ("1 + 1",)),
        ("exec", ("value = 1",)),
        ("__import__", ("os",)),
        ("open", ("unused", "w")),
        ("getattr", ("text", "__class__")),
        ("setattr", ("text", "x", 1)),
        ("compile", ("1 + 1", "<payload>", "eval")),
        ("globals", ()),
    ],
)
def test_restricted_unpickler_rejects_dangerous_builtins_by_name(
    callable_name: str, args: tuple[object, ...]
) -> None:
    class DangerousBuiltin:
        def __reduce__(self):
            return getattr(builtins, callable_name), args

    payload = pickle.dumps(DangerousBuiltin(), protocol=2)

    with pytest.raises(pickle.UnpicklingError):
        _load_restricted(payload)


def test_restricted_unpickler_rejects_builtin_import_gadget() -> None:
    class ImportGadget:
        def __reduce__(self):
            return __import__, ("os",)

    payload = pickle.dumps(ImportGadget(), protocol=2)

    with pytest.raises(pickle.UnpicklingError):
        _load_restricted(payload)


def test_restricted_unpickler_rejects_os_system_gadget(tmp_path) -> None:
    sentinel = tmp_path / "os_system_executed"

    class OsSystemGadget:
        def __reduce__(self):
            command = (
                f"{sys.executable} -c "
                f'"from pathlib import Path; '
                f"Path({str(sentinel)!r}).write_text('owned')\""
            )
            return os.system, (command,)

    payload = pickle.dumps(OsSystemGadget(), protocol=2)

    with pytest.raises(pickle.UnpicklingError):
        _load_restricted(payload)

    assert not sentinel.exists()


def test_restricted_unpickler_rejects_subprocess_gadget(tmp_path) -> None:
    sentinel = tmp_path / "subprocess_executed"

    class SubprocessGadget:
        def __reduce__(self):
            return subprocess.check_call, (
                [
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; "
                        f"Path({str(sentinel)!r}).write_text('owned')"
                    ),
                ],
            )

    payload = pickle.dumps(SubprocessGadget(), protocol=2)

    with pytest.raises(pickle.UnpicklingError):
        _load_restricted(payload)

    assert not sentinel.exists()


def test_legacy_pickle_fallback_roundtrips_real_trial_result(tmp_path) -> None:
    persistence = PersistenceManager(base_dir=tmp_path)
    timestamp = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    result = _make_optimization_result()
    result.trials = [
        TrialResult(
            trial_id="trial-1",
            config={"temperature": 0.2, "tags": ("baseline", "secure")},
            metrics={"accuracy": 0.91},
            status=TrialStatus.COMPLETED,
            duration=1.25,
            timestamp=timestamp,
            metadata={
                "attempts": {1, 2},
                "modes": frozenset({"json", "pickle"}),
                "payload": b"abc",
                "complexity": complex(1, 2),
            },
            error=TrialError(
                message="handled",
                error_type="ValueError",
                traceback="traceback text",
                timestamp=timestamp,
                config={"temperature": 0.2},
            ),
        )
    ]

    result_dir = Path(persistence.save_result(result, "legacy-only"))
    (result_dir / "trials.json.gz").unlink()

    loaded = persistence.load_result("legacy-only")

    assert len(loaded.trials) == 1
    loaded_trial = loaded.trials[0]
    assert isinstance(loaded_trial, TrialResult)
    assert loaded_trial.status is TrialStatus.COMPLETED
    assert loaded_trial.timestamp == timestamp
    assert loaded_trial.metadata["attempts"] == {1, 2}
    assert loaded_trial.metadata["modes"] == frozenset({"json", "pickle"})
    assert loaded_trial.metadata["payload"] == b"abc"
    assert loaded_trial.metadata["complexity"] == complex(1, 2)
    assert loaded_trial.error is not None
    assert loaded_trial.error.error_type == "ValueError"


def test_load_result_does_not_restore_sync_session_id(tmp_path) -> None:
    """load_result must NOT resurrect sync_session_id (#2020, Divergence 7).

    The id is a live, machine-local handle into the local session store. A
    reloaded historical result may have had its record removed by
    ``traigent sync --clean``, may live under a different
    TRAIGENT_RESULTS_FOLDER, or may come from another machine — so a restored
    value would be a stale id that ``traigent sync`` rejects with
    "Session ... not found", which is the exact failure #2020 removes. ``None``
    correctly means "not available here". This test pins that deliberate
    non-restoration: do not "fix" it into a round-trip.

    Pins the LOAD side specifically. ``save_result`` writes a fixed metadata
    whitelist that has never carried ``sync_session_id``, so a save→load
    assertion alone passes even if round-tripping is added to both sides; so the
    id is injected into the on-disk ``metadata.json`` here and ``load_result``
    must still drop it.
    """
    persistence = PersistenceManager(base_dir=tmp_path)
    result = _make_optimization_result()
    result.sync_session_id = "local-abc"

    persistence.save_result(result, "sync-handle")

    # The whitelist must not have picked it up on the way out …
    metadata_path = tmp_path / "sync-handle" / "metadata.json"
    saved_metadata = json.loads(metadata_path.read_text())
    assert "sync_session_id" not in saved_metadata

    # … and a metadata.json that DOES carry one (hand-edited, or written by a
    # future/forked SDK) must not resurrect it on the way in.
    saved_metadata["sync_session_id"] = "local-abc"
    metadata_path.write_text(json.dumps(saved_metadata))

    loaded = persistence.load_result("sync-handle")

    assert loaded.sync_session_id is None
    assert "sync_session_id" not in loaded.metadata


def test_restricted_unpickler_loads_protocol_2_real_payload() -> None:
    trial = TrialResult(
        trial_id="trial-1",
        config={"temperature": 0.2},
        metrics={"accuracy": 0.91},
        status=TrialStatus.COMPLETED,
        duration=1.25,
        timestamp=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        metadata={"payload": b"abc"},
    )
    payload = pickle.dumps([trial], protocol=2)

    loaded = _load_restricted(payload)

    assert isinstance(loaded, list)
    assert loaded[0] == trial


def test_restricted_unpickler_loads_gzipped_real_results_payload(tmp_path) -> None:
    payload_path = tmp_path / "trials.pkl.gz"
    trial = TrialResult(
        trial_id="trial-1",
        config={"temperature": 0.2},
        metrics={"accuracy": 0.91},
        status=TrialStatus.COMPLETED,
        duration=1.25,
        timestamp=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
    )
    with gzip.open(payload_path, "wb") as fp:
        pickle.dump([trial], fp, protocol=pickle.HIGHEST_PROTOCOL)

    with gzip.open(payload_path, "rb") as fp:
        loaded = RestrictedUnpickler(fp).load()

    assert loaded == [trial]


# --- #2031: PersistenceManager result-reload fidelity ------------------------
#
# Unlike the ConfigStateManager side, BOTH halves were lossy here: save_result
# writes a curated ~17-key metadata.json that never carried optimization_id,
# status, source, stop_reason, reason_code, total_cost, total_tokens, metrics,
# warnings, warning_codes, experiment_id, experiment_run_id, cloud_url,
# run_label, best_config_margin or the result's own timestamp — so load_result
# had nothing to restore them from and fabricated three of them instead. The
# curated keys are unchanged (list_results sorts on created_at, can_resume
# matches on function_name + configuration_space); the full field set is written
# alongside them under "result_fields".


def _sentinel_result() -> OptimizationResult:
    """A result whose every restorable field holds a non-default value."""
    return OptimizationResult(**copy.deepcopy(_SENTINELS))


def test_save_load_restores_every_field_in_the_manifest(tmp_path) -> None:
    """#2031: 25 restorable fields survive save -> load, not 11.

    Driven off the manifest rather than a hand-written list, so it cannot drift
    from ``traigent/utils/optimization_result_persistence.py``. Compared per
    field, never ``saved == loaded``: ``_experiment_stats`` is in ``__eq__`` and
    is deliberately not restored.
    """
    persistence = PersistenceManager(base_dir=tmp_path)
    persistence.save_result(_sentinel_result(), "full-fidelity")

    loaded = persistence.load_result("full-fidelity")

    for name in sorted(RESULT_RESTORE - {"trials"}):
        restore_message = f"{name} was not restored by PersistenceManager.load_result"
        assert getattr(loaded, name) == _SENTINELS[name], restore_message
    assert [trial.trial_id for trial in loaded.trials] == ["trial-sentinel-0"]


def test_save_load_resets_every_field_in_the_reset_manifest(tmp_path) -> None:
    """The other half of the manifest: what must NOT come back."""
    persistence = PersistenceManager(base_dir=tmp_path)
    result = _sentinel_result()
    result.sync_session_id = "20260727_005247_744918_answer_b1960eba"
    assert result.experiment_stats is not None  # populate the memo cache

    persistence.save_result(result, "reset-fields")
    loaded = persistence.load_result("reset-fields")

    assert RESULT_RESET == frozenset({"sync_session_id", "_experiment_stats"})
    assert loaded.sync_session_id is None
    assert loaded._experiment_stats is None
    # … and the cache recomputes to the same value from the restored trials.
    assert loaded.experiment_stats == result.experiment_stats


def test_load_result_ignores_a_sync_session_id_planted_in_result_fields(
    tmp_path,
) -> None:
    """#2020 / #2026 pin, extended to the new payload (#2031).

    ``test_load_result_does_not_restore_sync_session_id`` plants the id in the
    top-level metadata.json. This plants it where the restored fields actually
    come from, so the guarantee is pinned against the path that could plausibly
    round-trip it.
    """
    persistence = PersistenceManager(base_dir=tmp_path)
    persistence.save_result(_sentinel_result(), "planted")

    metadata_path = tmp_path / "planted" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    assert "sync_session_id" not in metadata["result_fields"]
    metadata["result_fields"]["sync_session_id"] = "local-abc"
    metadata_path.write_text(json.dumps(metadata))

    assert persistence.load_result("planted").sync_session_id is None


def test_warning_codes_clamp_survives_the_round_trip(tmp_path) -> None:
    """``OBJECTIVE_UNMATCHED`` must keep forcing ``success_rate`` to 0.0.

    ``success_rate`` reads ``warning_codes``; dropping them on load silently
    disarmed the clamp, so a reloaded run whose objective never matched reported
    a 100% success rate. The sentinel run has one COMPLETED trial, so an
    unclamped value would be 1.0.
    """
    persistence = PersistenceManager(base_dir=tmp_path)
    persistence.save_result(_sentinel_result(), "clamped")

    loaded = persistence.load_result("clamped")

    assert "OBJECTIVE_UNMATCHED" in loaded.warning_codes
    assert loaded.success_rate == 0.0


def test_curated_metadata_keeps_its_15_keys_and_no_longer_writes_strategy_preset(
    tmp_path,
) -> None:
    """The human/CLI-readable summary keys keep their names and meaning — minus one.

    ``list_results`` sorts on ``created_at`` and ``can_resume`` matches on
    ``function_name`` + ``configuration_space``; ``created_at`` stays *save*
    time. The result's own ``timestamp`` now lives in ``result_fields``.

    The presence loop below is deliberately not the whole test. A loop over
    expected keys can only catch a *disappearance*; deleting ``strategy_preset``
    from its tuple is invisible to it, so on its own it left the write-side
    removal unpinned — restoring the ``"strategy_preset": ...`` line in
    ``PersistenceManager.save_result`` kept the suite green. The absence
    assertions are what pin the removal, so the curated key set is checked in
    both directions: exactly these 15 keys are written, and ``strategy_preset``
    is not one of them.
    """
    persistence = PersistenceManager(base_dir=tmp_path)
    persistence.save_result(_sentinel_result(), "curated")

    metadata = json.loads((tmp_path / "curated" / "metadata.json").read_text())

    curated_keys = (
        "function_identifier",
        "function_name",
        "algorithm",
        "objectives",
        "configuration_space",
        "best_score",
        "best_config",
        "preset_selection",
        "success_rate",
        "duration",
        "convergence_info",
        "created_at",
        "total_trials",
        "successful_trials",
        "session_summary",
    )
    for key in curated_keys:
        assert key in metadata, f"curated key '{key}' disappeared"
    assert metadata["created_at"] != metadata["result_fields"]["timestamp"]

    # #2100/#2101: nothing populates ``metadata["strategy_preset"]`` any more,
    # so the curated summary must stop writing the key — not write it as null.
    # ``preset_selection`` above is the one that stays, for old artifacts.
    assert "strategy_preset" not in metadata
    assert "sync_session_id" not in metadata
    assert "_experiment_stats" not in metadata

    # Both directions: an EXTRA curated key is as much a regression as a
    # missing one, and this is what a re-added write actually trips.
    assert set(metadata) - {"result_fields", SCHEMA_VERSION_KEY} == set(curated_keys)


def test_versioned_metadata_round_trips_verbatim_including_an_empty_dict(
    tmp_path,
) -> None:
    """On the versioned path the decoded ``metadata`` is authoritative.

    The curated keys written alongside it are a *derived* summary for
    ``list_results`` / ``can_resume``, with save-time defaults substituted
    (``function_name`` becomes ``"unknown"``, ``configuration_space`` becomes
    ``{}``). Merging them back onto the restored result would let those
    placeholders override the field the run actually recorded — here turning a
    genuinely empty ``metadata`` into ``{"function_name": ..., ...}`` — and
    would break the verbatim round trip pinned by #2026.
    """
    persistence = PersistenceManager(base_dir=tmp_path)
    result = _sentinel_result()
    result.metadata = {}
    persistence.save_result(result, "empty-metadata")

    # The outer summary really does carry a curated value to be tempted by.
    metadata = json.loads((tmp_path / "empty-metadata" / "metadata.json").read_text())
    assert metadata["function_name"] == "unknown"
    assert metadata["configuration_space"] == {}
    assert metadata["result_fields"]["metadata"] == {}

    assert persistence.load_result("empty-metadata").metadata == {}


def test_versioned_metadata_is_not_overwritten_by_the_outer_summary(
    tmp_path,
) -> None:
    """A curated key must never shadow the authoritative persisted field.

    Same guarantee as above for a non-empty metadata: the outer summary is a
    duplicate written for a different purpose, so where the two disagree the
    result's own record wins.
    """
    persistence = PersistenceManager(base_dir=tmp_path)
    result = _sentinel_result()
    result.metadata = {"function_name": "answer_question", "run_note": "kept"}
    persistence.save_result(result, "authoritative")

    metadata_path = tmp_path / "authoritative" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["function_name"] = "a_different_name_from_the_summary"
    metadata["configuration_space"] = {"model": ["cheap", "smart"]}
    metadata_path.write_text(json.dumps(metadata))

    loaded = persistence.load_result("authoritative")

    assert loaded.metadata == {"function_name": "answer_question", "run_note": "kept"}
    assert "configuration_space" not in loaded.metadata


def _write_legacy_result(tmp_path, name: str) -> Path:
    """Write a pre-#2031 artifact: the 15 curated keys and nothing else."""
    result_dir = tmp_path / name
    result_dir.mkdir(parents=True)
    (result_dir / "metadata.json").write_text(
        json.dumps(
            {
                "function_identifier": "answer_question",
                "function_name": "answer_question",
                "algorithm": "grid_search",
                "objectives": ["accuracy"],
                "configuration_space": {"model": ["cheap", "smart"]},
                "best_score": 0.9,
                "best_config": {"model": "cheap"},
                "preset_selection": None,
                "success_rate": 1.0,
                "duration": 12.5,
                "convergence_info": {"converged": True},
                "created_at": "2026-01-02T03:04:05+00:00",
                "total_trials": 1,
                "successful_trials": 1,
                "session_summary": {"winning_trial_ids": ["t0"]},
            }
        )
    )
    with gzip.open(result_dir / "trials.json.gz", "wt") as handle:
        json.dump(
            [
                {
                    "trial_id": "t0",
                    "config": {"model": "cheap"},
                    "metrics": {"accuracy": 0.9},
                    "duration": 1.0,
                    "status": "completed",
                    "timestamp": "2026-01-02T03:04:00+00:00",
                    "metadata": {},
                }
            ],
            handle,
        )
    return result_dir


def test_legacy_artifact_loads_without_fabricating_what_it_never_stored(
    tmp_path, caplog
) -> None:
    """#2031 AC4: an old result still loads, and stops lying about itself.

    Before the fix it came back with a fabricated ``loaded_<name>``
    ``optimization_id``, a hardcoded ``COMPLETED`` status it never recorded,
    ``source == "backend"`` for what may well have been a local run, and no
    signal at all that anything had been dropped.
    """
    _write_legacy_result(tmp_path, "legacy-run")
    persistence = PersistenceManager(base_dir=tmp_path)

    with caplog.at_level(
        logging.WARNING, logger="traigent.utils.optimization_result_persistence"
    ):
        loaded = persistence.load_result("legacy-run")

    assert loaded.optimization_id == "unrestored-legacy:legacy-run"
    assert loaded.status is OptimizationStatus.UNKNOWN
    assert loaded.source == "unknown"
    # created_at is save time; the legacy format never stored run-completion time.
    assert loaded.timestamp == datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    assert loaded.best_config == {"model": "cheap"}
    assert [trial.trial_id for trial in loaded.trials] == ["t0"]
    # #1854 still restored from the curated keys.
    assert loaded.metadata["session_summary"] == {"winning_trial_ids": ["t0"]}
    # The fidelity signal is a log line naming the dropped fields …
    for name in ("total_cost", "total_tokens", "warnings", "warning_codes"):
        assert name in caplog.text
    # … and nothing was injected into the user-facing channels for it.
    assert loaded.warnings == []
    assert loaded.warning_codes == []


def test_legacy_artifact_still_rebuilds_metadata_from_the_curated_keys(
    tmp_path,
) -> None:
    """The other half of the #2026 split: the legacy path *must* reconstruct.

    A pre-#2031 artifact has no inner ``metadata`` record to be authoritative,
    so the curated summary keys are all there ever was — CLI apply / auto-load
    read ``function_name`` and ``configuration_space``, and #1854 reads
    ``session_summary``. Removing the merge from the versioned path must not
    take this with it.
    """
    _write_legacy_result(tmp_path, "legacy-metadata")
    persistence = PersistenceManager(base_dir=tmp_path)

    loaded = persistence.load_result("legacy-metadata")

    assert loaded.metadata == {
        "function_name": "answer_question",
        "configuration_space": {"model": ["cheap", "smart"]},
        "session_summary": {"winning_trial_ids": ["t0"]},
    }


def test_legacy_artifact_carrying_strategy_preset_still_loads(tmp_path) -> None:
    """A real pre-#2031 artifact does carry ``strategy_preset``; it must load.

    The fixture above no longer writes the key, because nothing produces it any
    more — but artifacts already on disk do. Dropping the key from the legacy
    read allowlist is only safe if such a file still opens, so this pins the
    read path: the artifact loads, and the retired key is simply not
    reconstructed rather than raising or taking the rest of the metadata down
    with it.
    """
    result_dir = _write_legacy_result(tmp_path, "legacy-preset")
    metadata_path = result_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["strategy_preset"] = {
        "preset_name": "balanced",
        "params": {"epsilon": 0.02},
        "selection_grade": "advisory",
    }
    metadata_path.write_text(json.dumps(metadata))

    persistence = PersistenceManager(base_dir=tmp_path)

    loaded = persistence.load_result("legacy-preset")

    assert loaded.best_config == {"model": "cheap"}
    assert [trial.trial_id for trial in loaded.trials] == ["t0"]
    assert loaded.metadata == {
        "function_name": "answer_question",
        "configuration_space": {"model": ["cheap", "smart"]},
        "session_summary": {"winning_trial_ids": ["t0"]},
    }


def test_an_artifact_written_by_an_older_schema_version_still_loads(
    tmp_path, monkeypatch
) -> None:
    """#2031 must not make every already-saved result unreadable.

    Saved artifacts outlive the build that wrote them. When a restorable field
    is added and the schema version bumped, records already on disk carry the
    older version and cannot possibly hold the new key; the loader decodes it as
    the field's declared dataclass default — what such a record genuinely held —
    rather than rejecting the file. Simulated with an existing field re-declared
    as "introduced in version 2", so the real save -> load path is exercised.
    """
    persistence = PersistenceManager(base_dir=tmp_path)
    persistence.save_result(_sentinel_result(), "written-last-release")

    metadata_path = tmp_path / "written-last-release" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    assert metadata[SCHEMA_VERSION_KEY] == RESULT_SCHEMA_VERSION
    assert metadata["result_fields"][SCHEMA_VERSION_KEY] == RESULT_SCHEMA_VERSION
    # … a field the release that wrote this artifact did not have yet.
    del metadata["result_fields"]["best_config_margin"]
    metadata_path.write_text(json.dumps(metadata))

    # The next release: version bumped, the new field recorded at that bumped
    # version, every other field left at its real introduction version (the
    # table is exact-coverage, so a partial one is not a valid build).
    next_version = RESULT_SCHEMA_VERSION + 1
    monkeypatch.setattr(manifest, "RESULT_SCHEMA_VERSION", next_version)
    monkeypatch.setattr(
        manifest,
        "FIELD_INTRODUCED_IN",
        {**manifest.FIELD_INTRODUCED_IN, "best_config_margin": next_version},
    )

    loaded = persistence.load_result("written-last-release")

    assert loaded.best_config_margin is None
    for name in sorted(RESULT_RESTORE - {"trials", "best_config_margin"}):
        assert getattr(loaded, name) == _SENTINELS[name], f"{name} was not restored"


def test_legacy_artifact_defaults_are_fresh_per_load(tmp_path) -> None:
    """Two loads of one legacy artifact must not share a mutable default."""
    _write_legacy_result(tmp_path, "legacy-shared")
    persistence = PersistenceManager(base_dir=tmp_path)

    first = persistence.load_result("legacy-shared")
    first.warnings.append("mutated by the caller")
    first.metrics["injected"] = True
    second = persistence.load_result("legacy-shared")

    assert second.warnings == []
    assert second.metrics == {}


def test_truncated_versioned_artifact_is_corruption_not_a_legacy_fallback(
    tmp_path,
) -> None:
    """A post-#2031 artifact promises every field, so a missing one must raise.

    Silently defaulting here would re-arm the exact defect #2031 fixes for the
    next field somebody adds.
    """
    persistence = PersistenceManager(base_dir=tmp_path)
    persistence.save_result(_sentinel_result(), "truncated")

    metadata_path = tmp_path / "truncated" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    del metadata["result_fields"]["stop_reason"]
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="stop_reason"):
        persistence.load_result("truncated")


def test_unsupported_result_schema_version_fails_explicitly(tmp_path) -> None:
    """An artifact from a newer SDK must not be silently mis-decoded."""
    persistence = PersistenceManager(base_dir=tmp_path)
    persistence.save_result(_sentinel_result(), "from-the-future")

    metadata_path = tmp_path / "from-the-future" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    # Both stamps, as a genuine future writer would have written them.
    metadata[SCHEMA_VERSION_KEY] = 999
    metadata["result_fields"][SCHEMA_VERSION_KEY] = 999
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="999"):
        persistence.load_result("from-the-future")


def test_a_future_version_envelope_cannot_be_decoded_via_an_older_payload(
    tmp_path,
) -> None:
    """The future-version refusal must not be trickable with a valid file.

    ``metadata.json`` stamps the schema version twice — on the envelope and on
    the ``result_fields`` payload — and ``load_result`` decodes the payload's
    copy. So an envelope declaring a version this build cannot read, wrapped
    around a payload that claims the version it *can*, would be decoded anyway:
    the refusal that exists to stop this reader dropping fields it does not
    understand would never run, on a structurally valid file.
    """
    persistence = PersistenceManager(base_dir=tmp_path)
    persistence.save_result(_sentinel_result(), "two-faced")

    metadata_path = tmp_path / "two-faced" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata[SCHEMA_VERSION_KEY] = 999
    metadata["result_fields"]["billing_currency"] = "EUR"  # a v999 field
    assert metadata["result_fields"][SCHEMA_VERSION_KEY] == RESULT_SCHEMA_VERSION
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError) as excinfo:
        persistence.load_result("two-faced")

    message = str(excinfo.value)
    assert "999" in message
    assert str(RESULT_SCHEMA_VERSION) in message


def test_a_version_stamped_on_only_one_half_of_the_artifact_is_rejected(
    tmp_path,
) -> None:
    """Both stamps are written together, so one alone means the file was edited."""
    persistence = PersistenceManager(base_dir=tmp_path)
    persistence.save_result(_sentinel_result(), "half-stamped")

    metadata_path = tmp_path / "half-stamped" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    del metadata[SCHEMA_VERSION_KEY]
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="envelope"):
        persistence.load_result("half-stamped")


def test_list_results_does_not_carry_the_full_result_payload(tmp_path) -> None:
    """#2031: the listing stays a summary.

    ``result_fields`` exists for ``load_result``; leaving it in every listing
    entry would hand callers (including ``list_optimization_results`` over MCP)
    a second full copy of each result's metadata, metrics and best_config.
    """
    persistence = PersistenceManager(base_dir=tmp_path)
    persistence.save_result(_sentinel_result(), "listed")

    (entry,) = persistence.list_results()

    assert "result_fields" not in entry
    # … and neither does the version stamp that describes it: a listing entry is
    # rendered as a set of result attributes, and `_schema_version` is not one.
    assert SCHEMA_VERSION_KEY not in entry
    assert entry["name"] == "listed"
    assert entry["best_score"] == _SENTINELS["best_score"]


def test_every_trial_field_survives_the_round_trip(tmp_path) -> None:
    """Trial fidelity, not just ``trial_id``.

    The result-level assertions above compare trials by id alone, so the trial
    writer could drop everything else and stay green. It did drop one thing:
    ``error_message`` was never written, though ``load_result`` has always read
    it — so a failed trial reloaded with its FAILED status intact and no record
    of *why* it failed.
    """
    failed = TrialResult(
        trial_id="trial-failed-0",
        config={"model": "smart"},
        metrics={},
        status=TrialStatus.FAILED,
        duration=2.25,
        timestamp=datetime(2026, 3, 15, 14, 31, 5, tzinfo=UTC),
        error_message="rate limited by the provider after 3 retries",
        metadata={"replicate": 2},
    )
    result = _sentinel_result()
    result.trials = [*result.trials, failed]

    persistence = PersistenceManager(base_dir=tmp_path)
    persistence.save_result(result, "trial-fidelity")
    loaded = persistence.load_result("trial-fidelity")

    reloaded = {trial.trial_id: trial for trial in loaded.trials}
    assert set(reloaded) == {"trial-sentinel-0", "trial-failed-0"}
    for original in result.trials:
        restored = reloaded[original.trial_id]
        assert restored.config == original.config
        assert restored.metrics == original.metrics
        assert restored.status is original.status
        assert restored.duration == original.duration
        assert restored.timestamp == original.timestamp
        assert restored.metadata == original.metadata
        assert restored.error_message == original.error_message

    assert reloaded["trial-failed-0"].error_message == (
        "rate limited by the provider after 3 retries"
    )


class _DuckTypedResult:
    """A result-like object that carries every field the encoder needs.

    ``save_result`` has always accepted any object shaped like a result, and
    ``tests/unit/cli`` exercises the CLI through exactly such stubs.
    """

    def __init__(self) -> None:
        self.trials: list[TrialResult] = []
        self.successful_trials: list[TrialResult] = []
        self.status = OptimizationStatus.COMPLETED
        self.best_config = {"model": "cheap"}
        self.best_score = 0.75
        self.metadata = {"function_name": "answer_question"}
        self.algorithm = "grid"
        self.objectives = ["accuracy"]
        self.preset_selection = None
        self.success_rate = 1.0
        self.duration = 0.5
        self.convergence_info: dict = {}
        self.optimization_id = "opt-duck"
        self.timestamp = datetime(2026, 3, 15, 14, 30, 22, tzinfo=UTC)
        self.source = "local"
        self.total_cost = 12.5
        self.stop_reason = "early_stopping"
        self.warnings = ["pricing incomplete"]
        self.warning_codes = ["PRICING_INCOMPLETE"]


def test_a_duck_typed_result_is_recorded_in_full_when_it_can_be(tmp_path) -> None:
    """Not being an ``OptimizationResult`` is not a reason to discard fields.

    The gate used to fall back to the curated pre-#2031 artifact for *any*
    non-dataclass result, and warn that values such as ``source`` and
    ``total_cost`` "cannot be recorded" — while holding them. They can be: the
    encoder is what decides, and it only fails on a field with no value to
    record.
    """
    persistence = PersistenceManager(base_dir=tmp_path)
    persistence.save_result(_DuckTypedResult(), "duck-full")

    loaded = persistence.load_result("duck-full")

    assert loaded.optimization_id == "opt-duck"
    assert loaded.status is OptimizationStatus.COMPLETED
    assert loaded.source == "local"
    assert loaded.total_cost == 12.5
    assert loaded.stop_reason == "early_stopping"
    assert loaded.warnings == ["pricing incomplete"]
    assert loaded.warning_codes == ["PRICING_INCOMPLETE"]
    assert loaded.timestamp == datetime(2026, 3, 15, 14, 30, 22, tzinfo=UTC)


def test_a_result_that_cannot_be_encoded_still_saves_as_the_legacy_artifact(
    tmp_path, caplog
) -> None:
    """The regression the gate was added for: a partial stub must still save.

    ``optimization_id`` and ``timestamp`` are required dataclass fields with no
    declared default, so an object lacking them has no honest full record to
    write. It gets the artifact it has always got, and the warning says which
    field stopped the full one rather than asserting that nothing could be kept.
    """
    partial = _DuckTypedResult()
    del partial.optimization_id

    persistence = PersistenceManager(base_dir=tmp_path)
    with caplog.at_level(logging.WARNING, logger="traigent.utils.persistence"):
        persistence.save_result(partial, "duck-partial")

    metadata = json.loads((tmp_path / "duck-partial" / "metadata.json").read_text())
    assert "result_fields" not in metadata
    assert SCHEMA_VERSION_KEY not in metadata
    assert "optimization_id" in caplog.text
    assert "_DuckTypedResult" in caplog.text

    loaded = persistence.load_result("duck-partial")
    assert loaded.optimization_id == "unrestored-legacy:duck-partial"


def test_a_type_violating_real_result_fails_the_save_instead_of_downgrading(
    tmp_path,
) -> None:
    """A genuine ``OptimizationResult`` gets no quiet fallback.

    A dataclass always carries every attribute, so an encoding failure there is a
    type-violating value — here a ``status`` that is not an
    ``OptimizationStatus`` — and writing a lossy artifact for it would hide a
    caller bug behind a log line.
    """
    result = _sentinel_result()
    result.status = "definitely_not_a_status"  # type: ignore[assignment]

    persistence = PersistenceManager(base_dir=tmp_path)
    with pytest.raises(ValueError, match="definitely_not_a_status"):
        persistence.save_result(result, "bogus-status")


class _UnserializableMetadataValue:
    """A metadata value whose ``to_dict()`` raises, as a plugin's might.

    ``_safe_json_value`` calls ``to_dict()`` on anything that has one, and that
    branch is unguarded — so an arbitrary caller exception reaches the save.
    """

    def to_dict(self) -> dict:
        raise ValueError("boom: this object refuses to serialize")


def test_unserializable_metadata_degrades_the_artifact_but_does_not_destroy_it(
    tmp_path, caplog
) -> None:
    """A sanitization failure must cost fidelity, not the whole run.

    ``encode_result_fields`` writes ``result.metadata`` in full, where the
    pre-#2031 artifact only ever carried a curated four keys of it. That
    widened what the sanitizer walks to *everything a user or plugin put in
    metadata* — and ``metadata`` is declared ``dict[str, Any]``, so an object
    in there is not the type violation the strict re-raise exists for. Letting
    it out meant the entire artifact was lost, where the previous release
    wrote the curated one.

    So this degrades to exactly that curated artifact, loudly, and the run
    stays on disk.
    """
    result = _sentinel_result()
    result.metadata = {
        "function_name": "answer_question",
        "configuration_space": {"model": ["cheap", "smart"]},
        "plugin_payload": _UnserializableMetadataValue(),
    }

    persistence = PersistenceManager(base_dir=tmp_path)
    with caplog.at_level(logging.WARNING, logger="traigent.utils.persistence"):
        persistence.save_result(result, "unserializable-metadata")

    metadata = json.loads(
        (tmp_path / "unserializable-metadata" / "metadata.json").read_text()
    )
    # Degraded: no full record, and the log says why rather than swallowing it.
    assert "result_fields" not in metadata
    assert SCHEMA_VERSION_KEY not in metadata
    assert "boom" in caplog.text
    assert "metadata" in caplog.text

    # But the run is persisted, and the curated summary is intact and readable.
    assert metadata["function_name"] == "answer_question"
    assert metadata["best_score"] == _SENTINELS["best_score"]
    loaded = persistence.load_result("unserializable-metadata")
    assert loaded.best_config == _SENTINELS["best_config"]


def test_the_strict_re_raise_still_covers_a_declared_field_type_violation(
    tmp_path,
) -> None:
    """The degrade above must not become a blanket "never fail the save".

    The two failure classes are different: an unserializable object under a
    ``dict[str, Any]`` key is within that field's declared type, while a
    ``timestamp`` that is not a timestamp is a caller bug that would otherwise
    reach disk as an artifact ``load_result`` refuses. Only the first degrades.
    """
    result = _sentinel_result()
    result.timestamp = "not-a-date"  # type: ignore[assignment]
    result.metadata = {"plugin_payload": _UnserializableMetadataValue()}

    persistence = PersistenceManager(base_dir=tmp_path)
    with pytest.raises(ValueError, match="not-a-date"):
        persistence.save_result(result, "violating-timestamp")


def test_a_duck_typed_result_with_unserializable_metadata_also_degrades(
    tmp_path, caplog
) -> None:
    """The duck-typed door gets the same treatment, for the same reason."""
    duck = _DuckTypedResult()
    duck.metadata = {
        "function_name": "answer_question",
        "x": _UnserializableMetadataValue(),
    }

    persistence = PersistenceManager(base_dir=tmp_path)
    with caplog.at_level(logging.WARNING, logger="traigent.utils.persistence"):
        persistence.save_result(duck, "duck-unserializable")

    metadata = json.loads(
        (tmp_path / "duck-unserializable" / "metadata.json").read_text()
    )
    assert "result_fields" not in metadata
    assert persistence.load_result("duck-unserializable").best_config == {
        "model": "cheap"
    }
