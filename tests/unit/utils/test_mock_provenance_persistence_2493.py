"""#2493: ``mock_llm_run`` / ``mock_llm_mixed`` survive save and load.

A result saved before these keys existed must load without them, and every
reader must report that as unknown (never as a real run).
"""

# Traceability: CONC-Layer-Data CONC-Quality-Reliability FUNC-STORAGE

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from traigent.api.types import OptimizationResult, OptimizationStatus
from traigent.utils.persistence import PersistenceManager
from traigent.utils.results_table import _mock_run_notice


def _result(metadata: dict) -> OptimizationResult:
    return OptimizationResult(
        trials=[],
        best_config={"p": 1},
        best_score=0.5,
        optimization_id="opt-2493",
        duration=1.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["objective"],
        algorithm="grid_search",
        timestamp=datetime.now(UTC),
        metadata=metadata,
    )


@pytest.fixture(autouse=True)
def _mock_mode_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "false")


@pytest.mark.parametrize(
    "recorded",
    [
        {"mock_llm_run": True},
        {"mock_llm_run": True, "mock_llm_mixed": True},
        {"mock_llm_run": False},
    ],
)
def test_provenance_keys_survive_save_and_load(tmp_path, recorded) -> None:
    persistence = PersistenceManager(base_dir=tmp_path)
    persistence.save_result(_result(dict(recorded)), "run")

    loaded = persistence.load_result("run")

    for key, value in recorded.items():
        assert loaded.metadata[key] is value
    if "mock_llm_mixed" not in recorded:
        assert "mock_llm_mixed" not in loaded.metadata


def test_legacy_result_lacks_keys_and_is_unknown_not_false(tmp_path) -> None:
    persistence = PersistenceManager(base_dir=tmp_path)
    persistence.save_result(_result({}), "legacy")

    loaded = persistence.load_result("legacy")

    assert "mock_llm_run" not in loaded.metadata
    assert loaded.metadata.get("mock_llm_run") is None  # unknown, not False
    # Readers do not assert "real": no notice when mock mode is off ...
    assert _mock_run_notice(loaded) is None


def test_legacy_result_is_hedged_when_mock_mode_on(tmp_path, monkeypatch) -> None:
    persistence = PersistenceManager(base_dir=tmp_path)
    persistence.save_result(_result({}), "legacy")
    loaded = persistence.load_result("legacy")

    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")

    notice = _mock_run_notice(loaded)
    assert notice is not None and "does not record" in notice
    # An explicit recorded False stays authoritative (a real run).
    assert _mock_run_notice(_result({"mock_llm_run": False})) is None
