"""Issue #2234: `PersistenceManager.save_result` honors TRAIGENT_LOG_EXAMPLE_CONTENT.

The sibling writer `ConfigStateManager.save_optimization_results` was fixed in
#2223. This writer persisted raw per-example input/expected/actual content into
`trials.json.gz` and the `trials.pkl.gz` pickle regardless of the opt-out.

Canary: a sentinel placed in every content field must be absent from every file
the method writes (gzip decompressed, pickle loaded) when the opt-out is set.
The documented default (`_should_log_example_content`: unset means content IS
written) is kept, so the sentinel is present when the variable is unset.
"""

from __future__ import annotations

import dataclasses
import gzip
import pickle
from datetime import UTC, datetime
from pathlib import Path

import pytest

from traigent.api.types import (
    ExampleResult,
    OptimizationResult,
    TrialResult,
    TrialStatus,
)
from traigent.utils.persistence import PersistenceManager

SENTINEL = "CANARY-2234-EXAMPLE-CONTENT"
ENV = "TRAIGENT_LOG_EXAMPLE_CONTENT"


def _example_object() -> ExampleResult:
    return ExampleResult(
        example_id="ex-obj",
        input_data={"question": f"{SENTINEL}-input"},
        expected_output=f"{SENTINEL}-expected",
        actual_output=f"{SENTINEL}-actual",
        metrics={"accuracy": 1.0},
        execution_time=0.1,
        success=True,
    )


def _result() -> OptimizationResult:
    example_dict = _example_object().to_dict()
    example_dict["example_id"] = "ex-dict"
    trial = TrialResult(
        trial_id="trial-1",
        config={"model": "m"},
        metrics={"accuracy": 1.0},
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime.now(UTC),
        # Both shapes reach trial metadata in practice: the trial factory
        # stores redacted dicts, while hand-built results carry the objects.
        metadata={"example_results": [_example_object(), example_dict]},
    )
    return OptimizationResult(
        trials=[trial],
        best_config={"model": "m"},
        best_score=1.0,
        optimization_id="opt-2234",
        duration=1.0,
        convergence_info={},
        status="completed",
        objectives=["accuracy"],
        algorithm="grid",
        timestamp=datetime.now(UTC),
        metadata={"function_name": "f"},
    )


def _strings_in(value: object) -> list[str]:
    """Every string reachable from an unpickled object graph.

    Walks fields directly: ``TrialResult.__repr__`` redacts its own fields, so
    ``repr()`` would hide content that is really in the pickle.
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for k, v in value.items() for s in _strings_in(k) + _strings_in(v)]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [s for item in value for s in _strings_in(item)]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return [
            s
            for f in dataclasses.fields(value)
            for s in _strings_in(getattr(value, f.name))
        ]
    return []


def _artifact_texts(result_dir: Path) -> dict[str, str]:
    """Every written file as searchable text: gzip decompressed, pickles loaded."""
    texts: dict[str, str] = {}
    for path in sorted(result_dir.iterdir()):
        raw = path.read_bytes()
        if path.name.endswith(".gz"):
            raw = gzip.decompress(raw)
        if path.name.endswith(".pkl.gz"):
            # Trusted: this test just wrote the file.
            loaded = pickle.loads(raw)  # noqa: S301
            texts[path.name] = "\n".join(_strings_in(loaded))
        texts[path.name + ":bytes"] = raw.decode("utf-8", errors="replace")
    return texts


@pytest.mark.parametrize("value", ["0", "false", "off"])
def test_opt_out_strips_example_content_from_every_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(ENV, value)
    result = _result()

    result_dir = Path(PersistenceManager(tmp_path).save_result(result, name="run"))

    texts = _artifact_texts(result_dir)
    assert {"metadata.json:bytes", "trials.json.gz:bytes", "trials.pkl.gz"} <= set(
        texts
    )
    leaking = sorted(name for name, text in texts.items() if SENTINEL in text)
    assert leaking == []
    # ids and metrics are kept; only content is dropped.
    assert "ex-obj" in texts["trials.pkl.gz"]
    assert "ex-dict" in texts["trials.json.gz:bytes"]


def test_opt_out_does_not_mutate_the_in_memory_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV, "0")
    result = _result()

    PersistenceManager(tmp_path).save_result(result, name="run")

    examples = result.trials[0].metadata["example_results"]
    assert examples[0].actual_output == f"{SENTINEL}-actual"
    assert examples[1]["input_data"] == {"question": f"{SENTINEL}-input"}


def test_opt_out_result_still_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV, "0")
    manager = PersistenceManager(tmp_path)
    manager.save_result(_result(), name="run")

    loaded = manager.load_result("run")

    assert [t.trial_id for t in loaded.trials] == ["trial-1"]


def test_default_keeps_example_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unset keeps today's documented default: content is written."""
    monkeypatch.delenv(ENV, raising=False)

    result_dir = Path(PersistenceManager(tmp_path).save_result(_result(), name="run"))

    texts = _artifact_texts(result_dir)
    assert SENTINEL in texts["trials.json.gz:bytes"]
    assert SENTINEL in texts["trials.pkl.gz"]
