"""Tests for persistence utilities."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from traigent.api.types import OptimizationResult, OptimizationStatus
from traigent.utils.persistence import PersistenceManager, ResumableOptimization


def _make_optimization_result() -> OptimizationResult:
    """Helper to build a minimal optimization result."""
    timestamp = datetime.now(timezone.utc)
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
