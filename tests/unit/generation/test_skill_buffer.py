from __future__ import annotations

from typing import Any

from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.generation import SkillTrainOptions
from traigent.generation.skill_train.buffer import RejectedEditBuffer
from traigent.generation.skill_train.edits import EditOp
from traigent.generation.skill_train.trainer import RolloutRecord, SkillTrainer


def _dataset(size: int = 30) -> Dataset:
    return Dataset(
        examples=[
            EvaluationExample(input_data={"i": i}, expected_output=i)
            for i in range(size)
        ],
        name="buffer",
    )


def _rollouts(dataset: Dataset) -> list[RolloutRecord]:
    return [
        RolloutRecord(
            example_id=str(example.input_data["i"]),
            input_data=example.input_data,
            expected=example.expected_output,
            actual="wrong",
            metrics={"accuracy": 0.0},
            success=False,
            is_failure=True,
        )
        for example in dataset.examples
    ]


class _Evaluate:
    def __call__(
        self, document_text: str, dataset: Dataset
    ) -> tuple[float, list[RolloutRecord]]:
        score = 0.4 if document_text == "bad" else 0.5
        return score, _rollouts(dataset)


class _DigestReflector:
    def __init__(self) -> None:
        self.digests: list[str | None] = []

    def analyze(
        self,
        document: Any,
        rollouts: list[RolloutRecord],
        polarity: str,
        max_edits: int,
        rejected_digest: str | None = None,
        meta_skill: str | None = None,
    ) -> list[EditOp]:
        self.digests.append(rejected_digest)
        return [EditOp("replace", "base", "bad", "try bad")]

    def merge(
        self,
        failure_edits: list[EditOp],
        success_edits: list[EditOp],
        max_edits: int,
        rejected_digest: str | None = None,
        meta_skill: str | None = None,
    ) -> list[EditOp]:
        return failure_edits[:max_edits]


def test_rejected_edit_buffer_digest_and_epoch_clear() -> None:
    buffer = RejectedEditBuffer(2)
    buffer.record(
        [EditOp("replace", "alpha", "beta", "bad")],
        selection_delta=-0.25,
        epoch=0,
        step=1,
    )

    digest = buffer.digest()

    assert "previously tried and REJECTED" in digest
    assert "replace on 'alpha'" in digest
    assert "Δ-0.2500" in digest
    buffer.clear_epoch()
    assert buffer.digest() == ""


def test_rejected_edit_buffer_can_persist_across_epochs() -> None:
    buffer = RejectedEditBuffer(2, persist_across_epochs=True)
    buffer.record(
        [EditOp("append", None, "content", "bad")],
        selection_delta=-0.1,
        epoch=0,
        step=0,
    )

    buffer.clear_epoch()

    assert "append" in buffer.digest()


def test_gate_rejection_digest_is_injected_into_later_analyst_prompt() -> None:
    reflector = _DigestReflector()
    trainer = SkillTrainer(
        dataset=_dataset(),
        evaluate_fn=_Evaluate(),
        reflector=reflector,
        options=SkillTrainOptions(
            epochs=1,
            steps_per_epoch=2,
            rollout_batch=4,
            reflection_minibatch=4,
            edit_budget=1,
            edit_budget_schedule="constant",
            selection_split=0.2,
            test_split=0.0,
            slow_update=False,
            meta_skill=False,
        ),
    )

    trainer.run("base")

    assert any(
        digest and "previously tried and REJECTED" in digest
        for digest in reflector.digests
    )
