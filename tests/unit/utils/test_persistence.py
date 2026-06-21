"""Tests for persistence utilities."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

from traigent.api.types import OptimizationResult, OptimizationStatus
from traigent.storage.local_storage import LocalStorageManager
from traigent.utils.persistence import PersistenceManager, ResumableOptimization


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


def test_list_results_includes_current_local_storage_sessions(tmp_path) -> None:
    """results list should discover current .traigent/local_storage sessions."""
    traigent_dir = tmp_path / ".traigent"
    storage = LocalStorageManager(str(traigent_dir / "local_storage"))
    session_id = storage.create_session(
        "demo_function",
        optimization_config={
            "algorithm": "grid_search",
            "objectives": ["accuracy"],
            "search_space": {"param": [0, 1]},
        },
    )
    storage.add_trial_result(session_id, {"param": 1}, 0.91)
    storage.finalize_session(session_id, "completed")

    persistence = PersistenceManager(base_dir=traigent_dir)
    results = persistence.list_results()

    result = next(item for item in results if item["name"] == session_id)
    assert result["source"] == "local_storage"
    assert result["function_name"] == "demo_function"
    assert result["algorithm"] == "grid_search"
    assert result["best_score"] == 0.91
    assert result["total_trials"] == 1
    assert result["success_rate"] == 1.0


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
