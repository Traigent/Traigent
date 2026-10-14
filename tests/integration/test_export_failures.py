"""Acceptance test for Traigent#2565: export a run's failing examples.

The issue's acceptance criterion, end to end and fully offline:

1. run a small deterministic grid optimization through the REAL local
   evaluator, on a JSONL dataset containing one known-wrong example;
2. export the failures with ``traigent results export-failures``;
3. the output contains exactly that example with its original input and
   expected_output, and round-trips through ``Dataset.from_jsonl``;
4. an unresolvable id raises (API) / exits non-zero without writing (CLI).

``block_network`` proves no socket or DNS traffic happens during the run or
the export: dataset content never leaves the machine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from traigent.cli.main import cli
from traigent.config.types import TraigentConfig
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.evaluators.base import Dataset
from traigent.evaluators.local import LocalEvaluator
from traigent.optimizers.grid import GridSearchOptimizer
from traigent.utils.persistence import PersistenceManager
from traigent.utils.regression_dataset import (
    UnresolvedExampleIdError,
    select_examples_by_id,
)

#: The agent upper-cases its input. Row 2's expected output is deliberately
#: wrong, so every configuration gets exactly that row wrong.
ROWS: list[dict[str, Any]] = [
    {"input": {"text": "alpha"}, "output": "ALPHA"},
    {"input": {"text": "beta"}, "output": "BETA"},
    {"input": {"text": "gamma"}, "output": "NOT-GAMMA"},
    {"input": {"text": "delta"}, "output": "DELTA"},
]
WRONG_INDEX = 2


def upper_agent(text: str) -> str:
    return text.upper()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.fixture
def offline_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    monkeypatch.delenv("TRAIGENT_API_KEY", raising=False)
    monkeypatch.delenv("TRAIGENT_BACKEND_URL", raising=False)
    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "local-store"))
    monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")
    monkeypatch.setenv("TRAIGENT_RUN_COST_LIMIT", "100.0")
    monkeypatch.chdir(tmp_path)
    return tmp_path


async def _run_optimization(dataset: Dataset):
    # The agent ignores ``alpha``, so both configs get the same rows right and
    # wrong; the run still has two columns to test the all_configs rule against.
    optimizer = GridSearchOptimizer(
        config_space={"alpha": [0, 1]}, objectives=["accuracy"]
    )
    orchestrator = OptimizationOrchestrator(
        optimizer=optimizer,
        evaluator=LocalEvaluator(metrics=["accuracy"], detailed=True),
        max_trials=2,
        config=TraigentConfig(offline=True, algorithm="grid"),
    )
    return await orchestrator.optimize(upper_agent, dataset)


@pytest.mark.asyncio
async def test_export_failures_acceptance(
    offline_env: Path, block_network: None
) -> None:
    workspace = offline_env
    dataset_path = workspace / "data.jsonl"
    _write_jsonl(dataset_path, ROWS)
    dataset = Dataset.from_jsonl(str(dataset_path))

    result = await _run_optimization(dataset)

    # 1. The real local evaluator emitted positional ids for every row, in
    #    every trial, and only the known-wrong row scored 0.
    assert len(result.trials) == 2
    expected_ids = [f"example_{i}" for i in range(len(ROWS))]
    for trial in result.trials:
        records = trial.metadata["example_results"]
        ids = [
            r["example_id"] if isinstance(r, dict) else r.example_id for r in records
        ]
        assert ids == expected_ids
    wrong_id = f"example_{WRONG_INDEX}"
    assert result.failing_examples() == [wrong_id]
    assert result.failing_examples(mode="best_config") == [wrong_id]

    # 2. Persist, then export through the CLI.
    storage = workspace / "results"
    PersistenceManager(str(storage)).save_result(result, name="acceptance")
    out = workspace / "failures.jsonl"

    cli_result = CliRunner().invoke(
        cli,
        [
            "results",
            "export-failures",
            "acceptance",
            "--dataset",
            str(dataset_path),
            "--out",
            str(out),
            "--storage-dir",
            str(storage),
        ],
    )
    assert cli_result.exit_code == 0, cli_result.output

    # 3. Exactly the wrong example, with its ORIGINAL input/expected output.
    lines = [line for line in out.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["input"] == ROWS[WRONG_INDEX]["input"]
    assert row["expected_output"] == ROWS[WRONG_INDEX]["output"]

    reloaded = Dataset.from_jsonl(str(out))
    assert len(reloaded) == 1
    source_example = dataset.examples[WRONG_INDEX]
    assert reloaded.examples[0].input_data == source_example.input_data
    assert reloaded.examples[0].expected_output == source_example.expected_output
    assert reloaded.examples[0].metadata == {
        **source_example.metadata,
        "example_id": wrong_id,
    }

    # 4a. API: the in-memory result exports the same row; an id the run never
    #     recorded raises.
    in_memory = select_examples_by_id(dataset, result.failing_examples(), result)
    assert [ex.input_data for ex in in_memory] == [source_example.input_data]
    with pytest.raises(UnresolvedExampleIdError):
        select_examples_by_id(dataset, ["example_99"], result)

    # 4b. CLI: against a dataset missing the failing row, the id cannot be
    #     resolved -> non-zero exit, and an existing --out is not replaced.
    truncated_path = workspace / "truncated.jsonl"
    _write_jsonl(truncated_path, ROWS[:WRONG_INDEX])
    previous = out.read_bytes()

    bad = CliRunner().invoke(
        cli,
        [
            "results",
            "export-failures",
            "acceptance",
            "--dataset",
            str(truncated_path),
            "--out",
            str(out),
            "--storage-dir",
            str(storage),
        ],
    )
    assert bad.exit_code != 0
    assert wrong_id in bad.output
    assert out.read_bytes() == previous
