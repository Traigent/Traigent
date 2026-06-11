"""Strictly gated skill-document training loop."""

from __future__ import annotations

import hashlib
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol

from traigent.evaluators.base import Dataset
from traigent.generation.options import SkillTrainOptions
from traigent.utils.logging import get_logger

from .artifacts import write_artifacts
from .document import SkillDocument
from .edits import EditApplyRecord, EditOp, apply_edits
from .splits import split_dataset
from .trainer_result import SkillTrainResult


@dataclass(slots=True)
class RolloutRecord:
    example_id: str
    input_data: Any
    expected: Any
    actual: Any
    metrics: dict[str, float]
    success: bool
    is_failure: bool


logger = get_logger(__name__)

EvaluateFn = Callable[[str, Dataset], tuple[float, list[RolloutRecord]]]


class ReflectorLike(Protocol):
    def analyze(
        self,
        document: SkillDocument,
        rollouts: list[RolloutRecord],
        polarity: Literal["failure", "success"],
        max_edits: int,
    ) -> list[EditOp]: ...

    def merge(
        self,
        failure_edits: list[EditOp],
        success_edits: list[EditOp],
        max_edits: int,
    ) -> list[EditOp]: ...


class SkillTrainer:
    """Minimal M1 SkillOpt-style loop with a strict held-out selection gate."""

    def __init__(
        self,
        *,
        dataset: Dataset,
        evaluate_fn: EvaluateFn,
        reflector: ReflectorLike,
        options: SkillTrainOptions | None = None,
    ) -> None:
        self._dataset = dataset
        self._evaluate_fn = evaluate_fn
        self._reflector = reflector
        self._options = options or SkillTrainOptions()
        self._score_cache: dict[tuple[str, str], tuple[float, list[RolloutRecord]]] = {}
        self._optimizer_calls = 0
        self._gate_evaluations = 0
        self._stop_reason: str | None = None

    def run(self, initial_document: str | SkillDocument) -> SkillTrainResult:
        options = self._options
        train, selection, test = split_dataset(
            self._dataset,
            options.selection_split,
            options.test_split,
            options.split_seed,
        )
        current = (
            initial_document
            if isinstance(initial_document, SkillDocument)
            else SkillDocument(initial_document)
        )
        baseline_selection_score, _ = self._evaluate_cached(current, selection)
        current_selection_score = baseline_selection_score

        accepted_edits: list[EditApplyRecord] = []
        all_edit_records: list[EditApplyRecord] = []
        epoch_summaries: list[dict[str, Any]] = []
        history: list[dict[str, Any]] = [
            {
                "doc_hash": current.doc_hash,
                "split": selection.name,
                "score": baseline_selection_score,
                "decision": "baseline",
            }
        ]

        for epoch in range(options.epochs):
            rng = random.Random(options.split_seed + epoch)
            epoch_summary: dict[str, Any] = {
                "epoch": epoch,
                "accepted": 0,
                "rejected": 0,
                "applied_records": 0,
            }
            for step in range(options.steps_per_epoch):
                if self._stop_reason is not None:
                    break

                batch = self._sample_batch(train, rng, epoch, step)
                _, rollouts = self._evaluate_cached(current, batch)
                failures, successes = self._partition_rollouts(rollouts)

                failure_edits = self._analyze_batches(
                    current, failures, "failure", options.edit_budget
                )
                success_edits = self._analyze_batches(
                    current, successes, "success", options.edit_budget
                )
                if self._stop_reason is not None:
                    break

                deduped = self._dedupe_edits([*failure_edits, *success_edits])
                merged = self._merge_edits(
                    failure_edits=deduped["failure"],
                    success_edits=deduped["success"],
                    max_edits=options.edit_budget,
                )
                if self._stop_reason is not None:
                    break

                ranked = sorted(
                    self._dedupe_edits(merged)["all"],
                    key=lambda op: (
                        0 if op.source_type == "failure" else 1,
                        -op.support_count,
                        op.edit_id,
                    ),
                )[: options.edit_budget]
                candidate_text, records = apply_edits(current.text, ranked)
                for record in records:
                    record.epoch = epoch
                    record.step = step
                    record.selection_score_before = current_selection_score
                all_edit_records.extend(records)
                applied = [record for record in records if record.status == "applied"]
                epoch_summary["applied_records"] += len(applied)
                if not applied:
                    continue

                if self._gate_limit_reached(epoch_summary):
                    break

                candidate = SkillDocument(
                    candidate_text,
                    version=current.version + 1,
                    parent_hash=current.doc_hash,
                )
                candidate_score, _ = self._evaluate_cached(candidate, selection)
                self._gate_evaluations += 1
                for record in records:
                    record.selection_score_after = candidate_score

                accepted = self._is_improvement(
                    candidate_score, current_selection_score
                )
                decision = "accepted" if accepted else "rejected"
                history.append(
                    {
                        "doc_hash": candidate.doc_hash,
                        "split": selection.name,
                        "score": candidate_score,
                        "decision": decision,
                    }
                )
                logger.info(
                    "Skill train gate %s: candidate=%s current=%s",
                    decision,
                    candidate_score,
                    current_selection_score,
                )
                if accepted:
                    current = candidate
                    current_selection_score = candidate_score
                    accepted_edits.extend(applied)
                    epoch_summary["accepted"] += 1
                else:
                    epoch_summary["rejected"] += 1

            if self._stop_reason is not None:
                epoch_summary["stop_reason"] = self._stop_reason
            logger.info("Skill train epoch summary: %s", epoch_summary)
            epoch_summaries.append(epoch_summary)
            if self._stop_reason is not None:
                break

        test_score: float | None = None
        evaluation_basis: Literal["selection_only", "held_out_test"] = "selection_only"
        if test is not None:
            test_score, _ = self._evaluate_cached(current, test)
            evaluation_basis = "held_out_test"

        result = SkillTrainResult(
            best_document=current.text,
            best_selection_score=current_selection_score,
            baseline_selection_score=baseline_selection_score,
            test_score=test_score,
            evaluation_basis=evaluation_basis,
            accepted_edits=accepted_edits,
            all_edit_records=all_edit_records,
            epoch_summaries=epoch_summaries,
            artifacts_dir=None,
        )

        initial_doc = (
            initial_document
            if isinstance(initial_document, SkillDocument)
            else SkillDocument(initial_document)
        )
        artifacts_dir = self._artifact_dir(initial_doc, options)
        if artifacts_dir is not None:
            result.artifacts_dir = write_artifacts(artifacts_dir, result, history)
        return result

    def _evaluate_cached(
        self, document: SkillDocument, dataset: Dataset
    ) -> tuple[float, list[RolloutRecord]]:
        key = (document.doc_hash, dataset.name)
        cached = self._score_cache.get(key)
        if cached is not None:
            return cached

        score, rollouts = self._evaluate_fn(document.text, dataset)
        if not rollouts:
            raise RuntimeError(
                "Skill training evaluator returned zero RolloutRecords for "
                f"dataset path/name {dataset.name!r}; cannot train on empty example_results."
            )
        value = (float(score), rollouts)
        self._score_cache[key] = value
        return value

    def _is_improvement(self, candidate: float, current: float) -> bool:
        if self._options.higher_is_better:
            return candidate > current
        return candidate < current

    def _sample_batch(
        self, train: Dataset, rng: random.Random, epoch: int, step: int
    ) -> Dataset:
        count = min(self._options.rollout_batch, len(train.examples))
        indices = rng.sample(range(len(train.examples)), count)
        return Dataset(
            examples=[train.examples[i] for i in indices],
            name=f"{train.name}__epoch{epoch}__step{step}",
            description=train.description,
            metadata=dict(train.metadata or {}),
        )

    def _partition_rollouts(
        self, rollouts: Sequence[RolloutRecord]
    ) -> tuple[list[RolloutRecord], list[RolloutRecord]]:
        failures: list[RolloutRecord] = []
        successes: list[RolloutRecord] = []
        for rollout in rollouts:
            marked = replace(
                rollout,
                is_failure=self._is_rollout_failure(rollout),
            )
            if marked.is_failure:
                failures.append(marked)
            else:
                successes.append(marked)
        return failures, successes

    def _is_rollout_failure(self, rollout: RolloutRecord) -> bool:
        if not rollout.success:
            return True
        metric = self._options.score_metric
        if metric is None:
            metric = _resolve_metric_name(rollout.metrics)
        value = rollout.metrics.get(metric) if metric else None
        if isinstance(value, (int, float)):
            return bool(float(value) < float(self._options.failure_threshold))
        return False

    def _analyze_batches(
        self,
        document: SkillDocument,
        rollouts: list[RolloutRecord],
        polarity: Literal["failure", "success"],
        max_edits: int,
    ) -> list[EditOp]:
        edits: list[EditOp] = []
        size = self._options.reflection_minibatch
        for start in range(0, len(rollouts), size):
            if self._optimizer_limit_reached():
                return edits
            batch = rollouts[start : start + size]
            if not batch:
                continue
            self._optimizer_calls += 1
            edits.extend(self._reflector.analyze(document, batch, polarity, max_edits))
        return edits

    def _merge_edits(
        self,
        *,
        failure_edits: list[EditOp],
        success_edits: list[EditOp],
        max_edits: int,
    ) -> list[EditOp]:
        if not failure_edits and not success_edits:
            return []
        if self._optimizer_limit_reached():
            return []
        self._optimizer_calls += 1
        return self._reflector.merge(failure_edits, success_edits, max_edits)

    def _optimizer_limit_reached(self) -> bool:
        limit = self._options.max_optimizer_calls
        if limit is not None and self._optimizer_calls >= limit:
            self._stop_reason = "max_optimizer_calls"
            return True
        return False

    def _gate_limit_reached(self, epoch_summary: dict[str, Any]) -> bool:
        limit = self._options.max_gate_evaluations
        if limit is not None and self._gate_evaluations >= limit:
            self._stop_reason = "max_gate_evaluations"
            epoch_summary["stop_reason"] = self._stop_reason
            return True
        return False

    @staticmethod
    def _dedupe_edits(ops: Sequence[EditOp]) -> dict[str, list[EditOp]]:
        seen: set[str] = set()
        failure: list[EditOp] = []
        success: list[EditOp] = []
        all_ops: list[EditOp] = []
        for op in ops:
            if op.edit_id in seen:
                continue
            seen.add(op.edit_id)
            all_ops.append(op)
            if op.source_type == "success":
                success.append(op)
            else:
                failure.append(op)
        return {"failure": failure, "success": success, "all": all_ops}

    @staticmethod
    def _artifact_dir(
        initial_document: SkillDocument, options: SkillTrainOptions
    ) -> str | None:
        if options.artifacts_dir:
            return str(options.artifacts_dir)
        run_basis = f"{initial_document.doc_hash}:{options.split_seed}:{time.time()}"
        run_id = hashlib.sha256(run_basis.encode("utf-8")).hexdigest()[:12]
        return str(Path(".traigent") / "skill_train" / run_id)


def _resolve_metric_name(metrics: dict[str, float]) -> str | None:
    for preferred in ("accuracy", "score", "primary"):
        if preferred in metrics:
            return preferred
    if metrics:
        return sorted(metrics)[0]
    return None


__all__ = ["RolloutRecord", "SkillTrainer", "SkillTrainResult"]
