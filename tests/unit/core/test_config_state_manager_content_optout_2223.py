"""Issue #2223: `save_optimization_results` (the `save_to=` path) must honor
``TRAIGENT_LOG_EXAMPLE_CONTENT`` the same way the optimization logger's trial
jsonl already does (#1069).

Before this fix, ``ConfigStateManager.save_optimization_results`` dumped the
whole ``OptimizationResult`` dataclass — including every
``trial.metadata["example_results"]`` entry's raw ``input_data`` /
``expected_output`` / ``actual_output`` — with no awareness of the env switch
at all. A customer who set ``TRAIGENT_LOG_EXAMPLE_CONTENT=false`` expecting no
raw example content to reach disk anywhere still got it via ``save_to=``.

Cost safety: no ``OptimizedFunction``, no ``optimize()``, no evaluator — pure
save over ``tmp_path``. No LLM call, no network, no spend.
"""
# Traceability: CONC-Layer-Core CONC-Quality-Reliability FUNC-ORCH-LIFECYCLE REQ-ORCH-003

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from traigent.api.types import (
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)
from traigent.core.config_state_manager import ConfigStateManager
from traigent.utils.optimization_logger import ENV_LOG_EXAMPLE_CONTENT

SENTINEL_INPUT = "SENTINEL_RAW_QUESTION_2223"
SENTINEL_EXPECTED = "SENTINEL_RAW_EXPECTED_2223"
SENTINEL_OUTPUT = "SENTINEL_RAW_RESPONSE_2223"


def _fn(x: int) -> int:
    return x


def _manager(tmp_path: Path) -> ConfigStateManager:
    return ConfigStateManager(
        func=_fn,
        default_config={"model": "cheap"},
        local_storage_path=str(tmp_path / "store"),
        configuration_space={"model": ["cheap", "smart"]},
        auto_load_best=False,
        load_from=None,
        setup_wrapper_callback=lambda: None,
    )


def _example_with_sentinels() -> dict:
    return {
        "example_id": "ex-1",
        "input_data": {"question": SENTINEL_INPUT},
        "expected_output": SENTINEL_EXPECTED,
        "actual_output": {"raw_text": SENTINEL_OUTPUT},
        "metrics": {"accuracy": 1.0},
        "execution_time": 0.01,
        "success": True,
        "error_message": None,
        "metadata": {},
    }


def _result_with_example() -> OptimizationResult:
    trial = TrialResult(
        trial_id="trial-1",
        config={"model": "cheap"},
        metrics={"accuracy": 1.0},
        status=TrialStatus.COMPLETED,
        duration=0.1,
        timestamp=datetime(2026, 6, 2, tzinfo=UTC),
        metadata={"example_results": [_example_with_sentinels()]},
    )
    return OptimizationResult(
        trials=[trial],
        best_config={"model": "cheap"},
        best_score=1.0,
        optimization_id="opt-1",
        duration=1.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="grid",
        timestamp=datetime(2026, 6, 2, tzinfo=UTC),
    )


def _save(tmp_path: Path) -> str:
    path = tmp_path / "results.json"
    saver = _manager(tmp_path)
    saver._optimization_results = _result_with_example()
    saver.save_optimization_results(str(path))
    return path.read_text(encoding="utf-8")


# --- the canary -------------------------------------------------------


def test_save_to_omits_raw_example_content_when_opted_out(tmp_path, monkeypatch):
    """The regression: opting out must keep sentinel content off disk.

    This is the canary this workspace requires for any export of
    user-controlled values (see #2223): a sentinel goes in, opt-out is set,
    and the sentinel must be provably absent from the written artifact — not
    merely "the code looks right".
    """
    monkeypatch.setenv(ENV_LOG_EXAMPLE_CONTENT, "false")

    raw = _save(tmp_path)

    assert SENTINEL_INPUT not in raw
    assert SENTINEL_EXPECTED not in raw
    assert SENTINEL_OUTPUT not in raw

    # ids/metrics/success survive — only content is redacted.
    written = json.loads(raw)
    example = written["trials"][0]["metadata"]["example_results"][0]
    assert example["example_id"] == "ex-1"
    assert example["metrics"] == {"accuracy": 1.0}
    assert example["success"] is True
    assert example["input_data"] is None
    assert example["expected_output"] is None
    assert example["actual_output"] is None


def test_save_to_includes_raw_example_content_by_default(tmp_path, monkeypatch):
    """Default (unset) must match the sibling opt-out's own default: on.

    Documents the existing, permissive default (unset = log raw content) —
    this test pins that behavior rather than silently changing it; see the
    PR body for the separate observation that the default itself is
    permissive.
    """
    monkeypatch.delenv(ENV_LOG_EXAMPLE_CONTENT, raising=False)

    raw = _save(tmp_path)

    assert SENTINEL_INPUT in raw
    assert SENTINEL_EXPECTED in raw
    assert SENTINEL_OUTPUT in raw


@pytest.mark.parametrize("value", ["0", "no", "off", "FALSE", " Off "])
def test_save_to_recognizes_every_falsey_spelling(tmp_path, monkeypatch, value):
    """Same falsey-value set the sibling opt-out recognizes (#1069) — a
    second, narrower reading here would silently under-redact for a customer
    relying on one of these spellings.
    """
    monkeypatch.setenv(ENV_LOG_EXAMPLE_CONTENT, value)

    raw = _save(tmp_path)

    assert SENTINEL_INPUT not in raw
    assert SENTINEL_EXPECTED not in raw
    assert SENTINEL_OUTPUT not in raw
