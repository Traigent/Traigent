"""Tests for persistence utilities."""

from __future__ import annotations

import builtins
import gzip
import hashlib
import io
import json
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
    assert loaded.metadata == {
        "function_name": "demo_function",
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
