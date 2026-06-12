from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest

from traigent.api.decorators import optimize
from traigent.api.parameter_ranges import Choices, TextDocument
from traigent.api.types import OptimizationResult, TrialResult, TrialStatus
from traigent.core.optimized_function import OptimizedFunction
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.generation import SkillTrainOptions
from traigent.generation.skill_train.edits import EditOp
from traigent.generation.skill_train.trainer import RolloutRecord, SkillTrainer

SENTINEL = "SENTINEL_PRIVATE_INPUT_42"


def _dataset(size: int = 30, *, sentinel: bool = False) -> Dataset:
    examples = []
    for i in range(size):
        value = SENTINEL if sentinel and i == 0 else f"q{i}"
        examples.append(
            EvaluationExample(input_data={"q": value}, expected_output=f"a{i}")
        )
    return Dataset(examples=examples, name="skillset")


def _rollouts(dataset: Dataset, *, success: bool = False) -> list[RolloutRecord]:
    return [
        RolloutRecord(
            example_id=f"ex{i}",
            input_data=example.input_data,
            expected=example.expected_output,
            actual="actual",
            metrics={"accuracy": 1.0 if success else 0.0},
            success=success,
            is_failure=not success,
        )
        for i, example in enumerate(dataset.examples)
    ]


class _ScriptedEvaluate:
    def __init__(self, scores: dict[str, float], *, empty: bool = False) -> None:
        self.scores = scores
        self.empty = empty
        self.calls: list[tuple[str, str]] = []

    def __call__(
        self, document_text: str, dataset: Dataset
    ) -> tuple[float, list[RolloutRecord]]:
        self.calls.append((document_text, dataset.name))
        score = self.scores.get(document_text, 0.0)
        if self.empty:
            return score, []
        return score, _rollouts(dataset)


class _Reflector:
    def __init__(self, edits: list[EditOp]) -> None:
        self.edits = edits

    def analyze(
        self,
        document: Any,
        rollouts: list[RolloutRecord],
        polarity: str,
        max_edits: int,
    ) -> list[EditOp]:
        return self.edits[:max_edits]

    def merge(
        self,
        failure_edits: list[EditOp],
        success_edits: list[EditOp],
        max_edits: int,
    ) -> list[EditOp]:
        return self.edits[:max_edits]


def _options(**overrides: Any) -> SkillTrainOptions:
    values = {
        "epochs": 1,
        "rollout_batch": 4,
        "steps_per_epoch": 1,
        "reflection_minibatch": 2,
        "edit_budget": 1,
        "selection_split": 0.2,
        "test_split": 0.0,
    }
    values.update(overrides)
    return SkillTrainOptions(**values)


def _trainer(
    *,
    scores: dict[str, float],
    edit: EditOp,
    options: SkillTrainOptions | None = None,
    dataset: Dataset | None = None,
) -> tuple[SkillTrainer, _ScriptedEvaluate]:
    evaluate = _ScriptedEvaluate(scores)
    trainer = SkillTrainer(
        dataset=dataset or _dataset(),
        evaluate_fn=evaluate,
        reflector=_Reflector([edit]),
        options=options or _options(),
    )
    return trainer, evaluate


def test_accepts_strictly_better_selection_score() -> None:
    trainer, _ = _trainer(
        scores={"base": 0.5, "better": 0.7},
        edit=EditOp("replace", "base", "better", "improves"),
    )

    result = trainer.run("base")

    assert result.best_document == "better"
    assert result.baseline_selection_score == 0.5
    assert result.best_selection_score == 0.7
    assert len(result.accepted_edits) == 1
    assert result.evaluation_basis == "selection_only"


def test_equal_score_candidate_is_rejected() -> None:
    trainer, _ = _trainer(
        scores={"base": 0.5, "better": 0.5},
        edit=EditOp("replace", "base", "better", "tie"),
    )

    result = trainer.run("base")

    assert result.best_document == "base"
    assert result.accepted_edits == []
    assert result.epoch_summaries[0]["rejected"] == 1


def test_score_drop_rejects_and_reverts() -> None:
    trainer, _ = _trainer(
        scores={"base": 0.5, "worse": 0.4},
        edit=EditOp("replace", "base", "worse", "bad"),
    )

    result = trainer.run("base")

    assert result.best_document == "base"
    assert result.best_selection_score == 0.5


def test_lower_score_accepted_when_higher_is_better_false() -> None:
    trainer, _ = _trainer(
        scores={"base": 10.0, "lower": 8.0},
        edit=EditOp("replace", "base", "lower", "lower is better"),
        options=_options(higher_is_better=False),
    )

    result = trainer.run("base")

    assert result.best_document == "lower"
    assert result.best_selection_score == 8.0


def test_empty_example_results_raise() -> None:
    evaluate = _ScriptedEvaluate({"base": 0.5}, empty=True)
    trainer = SkillTrainer(
        dataset=_dataset(),
        evaluate_fn=evaluate,
        reflector=_Reflector([EditOp("append", None, "x", "x")]),
        options=_options(),
    )

    with pytest.raises(RuntimeError, match="zero RolloutRecords"):
        trainer.run("base")


def test_max_gate_evaluations_stop_honored() -> None:
    trainer, _ = _trainer(
        scores={"base": 0.5, "better": 0.7},
        edit=EditOp("replace", "base", "better", "improves"),
        options=_options(max_gate_evaluations=0),
    )

    result = trainer.run("base")

    assert result.best_document == "base"
    assert result.epoch_summaries[0]["stop_reason"] == "max_gate_evaluations"


def test_score_cache_avoids_same_doc_selection_reevaluation() -> None:
    trainer, evaluate = _trainer(
        scores={"base": 0.5},
        edit=EditOp("replace", "base", "base", "same hash"),
    )

    trainer.run("base")

    selection_calls = [
        call for call in evaluate.calls if call == ("base", "skillset__selection")
    ]
    assert len(selection_calls) == 1


def test_artifacts_written_and_training_log_omits_private_content(tmp_path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    trainer, _ = _trainer(
        scores={"base": 0.5, "better": 0.7},
        edit=EditOp("replace", "base", "better", "improves"),
        options=_options(artifacts_dir=str(artifacts_dir)),
        dataset=_dataset(sentinel=True),
    )

    result = trainer.run("base")

    assert result.artifacts_dir == str(artifacts_dir)
    report = json.loads((artifacts_dir / "edit_apply_report.json").read_text())
    assert report[0]["status"] == "applied"
    training_log = (artifacts_dir / "training_log.jsonl").read_text()
    assert SENTINEL not in training_log
    assert "better" not in training_log
    assert (artifacts_dir / "best_skill.md").read_text() == "better"


def test_test_split_scored_once_after_loop() -> None:
    trainer, evaluate = _trainer(
        scores={"base": 0.5, "better": 0.7},
        edit=EditOp("replace", "base", "better", "improves"),
        options=_options(test_split=0.1),
    )

    result = trainer.run("base")

    assert result.evaluation_basis == "held_out_test"
    assert result.test_score == 0.7
    counts = Counter(name for _, name in evaluate.calls)
    assert counts["skillset__test"] == 1


@dataclass
class _FakeExampleResult:
    example_id: str
    input_data: dict[str, Any]
    expected_output: str
    actual_output: str
    metrics: dict[str, float]
    success: bool


def test_train_skill_preflight_rejects_unpinned_non_doc_param() -> None:
    def fn(**kwargs: Any) -> str:
        return "ok"

    opt = OptimizedFunction(
        func=fn,
        eval_dataset=_dataset(),
        objectives=["accuracy"],
        configuration_space={
            "doc": Choices(["base"]),
            "temperature": Choices([0.1, 0.2]),
        },
    )

    with pytest.raises(ValueError, match="temperature"):
        opt.train_skill(
            document="base",
            doc_param="doc",
            optimizer_llm=lambda prompt: "{}",
        )


def test_train_skill_bridge_extracts_example_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fn(**kwargs: Any) -> str:
        return "ok"

    opt = OptimizedFunction(
        func=fn,
        eval_dataset=_dataset(),
        objectives=["accuracy"],
        configuration_space={"doc": Choices(["base"])},
    )

    trial = TrialResult(
        trial_id="t1",
        config={"doc": "base"},
        metrics={"accuracy": 0.5},
        status=TrialStatus.COMPLETED,
        duration=0.0,
        timestamp=datetime.now(UTC),
        metadata={
            "example_results": [
                _FakeExampleResult(
                    "ex1", {"q": "x"}, "y", "z", {"accuracy": 0.0}, False
                )
            ]
        },
    )
    opt_result = OptimizationResult(
        trials=[trial],
        best_config={"doc": "base"},
        best_score=0.5,
        optimization_id="o1",
        duration=0.0,
        convergence_info={},
        status=TrialStatus.COMPLETED,  # type: ignore[arg-type]
        objectives=["accuracy"],
        algorithm="fake",
        timestamp=datetime.now(UTC),
    )

    def fake_optimize_sync(**kwargs: Any) -> OptimizationResult:
        return opt_result

    monkeypatch.setattr(opt, "optimize_sync", fake_optimize_sync)
    result = opt.train_skill(
        document="base",
        doc_param="doc",
        optimizer_llm=lambda prompt: json.dumps({"edits": []}),
        skill_train={"epochs": 1, "rollout_batch": 4, "selection_split": 0.2},
    )

    assert result.baseline_selection_score == 0.5


def test_train_skill_auto_discovers_single_text_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fn(**kwargs: Any) -> str:
        return "ok"

    opt = OptimizedFunction(
        func=fn,
        eval_dataset=_dataset(),
        objectives=["accuracy"],
        configuration_space={"doc": TextDocument("base")},
    )

    trial = TrialResult(
        trial_id="t1",
        config={"doc": "base"},
        metrics={"accuracy": 0.5},
        status=TrialStatus.COMPLETED,
        duration=0.0,
        timestamp=datetime.now(UTC),
        metadata={
            "example_results": [
                _FakeExampleResult(
                    "ex1", {"q": "x"}, "y", "z", {"accuracy": 0.0}, False
                )
            ]
        },
    )
    opt_result = OptimizationResult(
        trials=[trial],
        best_config={"doc": "base"},
        best_score=0.5,
        optimization_id="o1",
        duration=0.0,
        convergence_info={},
        status=TrialStatus.COMPLETED,  # type: ignore[arg-type]
        objectives=["accuracy"],
        algorithm="fake",
        timestamp=datetime.now(UTC),
    )

    def fake_optimize_sync(**kwargs: Any) -> OptimizationResult:
        return opt_result

    monkeypatch.setattr(opt, "optimize_sync", fake_optimize_sync)

    result = opt.train_skill(
        document="base",
        optimizer_llm=lambda prompt: json.dumps({"edits": []}),
        skill_train={
            "epochs": 1,
            "rollout_batch": 4,
            "selection_split": 0.2,
            "meta_skill": False,
        },
    )

    assert result.summary["doc_param"] == "doc"


def test_train_skill_text_document_auto_discovery_ambiguity_error() -> None:
    def fn(**kwargs: Any) -> str:
        return "ok"

    opt = OptimizedFunction(
        func=fn,
        eval_dataset=_dataset(),
        objectives=["accuracy"],
        configuration_space={
            "first": TextDocument("base"),
            "second": TextDocument("other"),
        },
    )

    with pytest.raises(ValueError, match="multiple TextDocument"):
        opt.train_skill(
            document="base",
            optimizer_llm=lambda prompt: json.dumps({"edits": []}),
        )


def test_optimize_decorator_preserves_text_document_marker() -> None:
    @optimize(
        eval_dataset=_dataset(),
        objectives=["accuracy"],
        configuration_space={"doc": TextDocument("base")},
    )
    def fn(**kwargs: Any) -> str:
        return "ok"

    assert isinstance(fn.configuration_space["doc"], TextDocument)
