"""Strictly gated skill-document training loop."""

from __future__ import annotations

import hashlib
import inspect
import json
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
from .buffer import RejectedEditBuffer
from .document import SkillDocument
from .edits import EditApplyRecord, EditOp, apply_edits
from .schedule import edit_budget_for_step
from .slow_update import (
    apply_slow_update,
    build_slow_update_probe,
    categorize_slow_update_rollouts,
    extract_slow_update_content,
)
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
        rejected_digest: str | None = None,
        meta_skill: str | None = None,
    ) -> list[EditOp]: ...

    def merge(
        self,
        failure_edits: list[EditOp],
        success_edits: list[EditOp],
        max_edits: int,
        rejected_digest: str | None = None,
        meta_skill: str | None = None,
    ) -> list[EditOp]: ...

    def slow_update(
        self,
        prev_doc: SkillDocument,
        cur_doc: SkillDocument,
        categorized: object,
        prior_guidance: str,
    ) -> str: ...

    def meta_skill(
        self,
        accept_history: list[dict[str, object]],
        reject_history: list[dict[str, object]],
        prior_meta: str,
    ) -> str: ...


class SkillTrainer:
    """Minimal M1 SkillOpt-style loop with a strict held-out selection gate."""

    def __init__(
        self,
        *,
        dataset: Dataset,
        evaluate_fn: EvaluateFn,
        reflector: ReflectorLike,
        options: SkillTrainOptions | None = None,
        selection_dataset: Dataset | None = None,
        test_dataset: Dataset | None = None,
        artifacts_root: str | Path | None = None,
    ) -> None:
        self._dataset = dataset
        self._evaluate_fn = evaluate_fn
        self._reflector = reflector
        self._options = options or SkillTrainOptions()
        self._selection_dataset = selection_dataset
        self._test_dataset = test_dataset
        self._artifacts_root = Path(artifacts_root) if artifacts_root else None
        self._score_cache: dict[tuple[str, str], tuple[float, list[RolloutRecord]]] = {}
        self._optimizer_calls = 0
        self._gate_evaluations = 0
        self._stop_reason: str | None = None
        self._rejected_buffer = (
            RejectedEditBuffer(
                self._options.rejected_buffer_max,
                persist_across_epochs=False,
            )
            if self._options.rejected_buffer
            else None
        )
        self._meta_skill = ""
        self._accept_history: list[dict[str, object]] = []
        self._reject_history: list[dict[str, object]] = []

    def run(self, initial_document: str | SkillDocument) -> SkillTrainResult:
        options = self._options
        train, selection, test = self._resolve_datasets()
        slow_probe = (
            build_slow_update_probe(
                train, options.slow_update_probe_size, options.split_seed
            )
            if options.slow_update
            else None
        )
        total_steps = options.epochs * options.steps_per_epoch
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
        previous_epoch_best: SkillDocument | None = None

        for epoch in range(options.epochs):
            if self._rejected_buffer is not None:
                self._rejected_buffer.clear_epoch()
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

                budget = edit_budget_for_step(
                    edit_budget=options.edit_budget,
                    edit_budget_floor=options.edit_budget_floor,
                    schedule=options.edit_budget_schedule,
                    step_index=epoch * options.steps_per_epoch + step,
                    total_steps=total_steps,
                )
                batch = self._sample_batch(train, rng, epoch, step)
                _, rollouts = self._evaluate_cached(current, batch)
                failures, successes = self._partition_rollouts(rollouts)

                failure_edits = self._analyze_batches(
                    current, failures, "failure", budget
                )
                success_edits = self._analyze_batches(
                    current, successes, "success", budget
                )
                if self._stop_reason is not None:
                    break

                deduped = self._dedupe_edits([*failure_edits, *success_edits])
                merged = self._merge_edits(
                    failure_edits=deduped["failure"],
                    success_edits=deduped["success"],
                    max_edits=budget,
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
                )[:budget]
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

                selection_delta = self._selection_delta(
                    candidate_score, current_selection_score
                )
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
                    self._record_history(
                        self._accept_history, applied, selection_delta, epoch, step
                    )
                else:
                    for record in applied:
                        record.status = "rejected_gate"
                    epoch_summary["rejected"] += 1
                    self._record_gate_rejection(applied, selection_delta, epoch, step)

            if (
                self._stop_reason is None
                and previous_epoch_best is not None
                and slow_probe is not None
                and epoch >= 1
            ):
                current, current_selection_score = self._maybe_apply_slow_update(
                    previous_epoch_best=previous_epoch_best,
                    current=current,
                    current_selection_score=current_selection_score,
                    slow_probe=slow_probe,
                    selection=selection,
                    epoch=epoch,
                    accepted_edits=accepted_edits,
                    all_edit_records=all_edit_records,
                    history=history,
                    epoch_summary=epoch_summary,
                )
            if self._stop_reason is None:
                self._update_meta_skill(epoch_summary)
            if self._stop_reason is not None:
                epoch_summary["stop_reason"] = self._stop_reason
            logger.info("Skill train epoch summary: %s", epoch_summary)
            epoch_summaries.append(epoch_summary)
            previous_epoch_best = current
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
            summary={
                "meta_skill": self._meta_skill,
                "accept_history": self._accept_history,
                "reject_history": self._reject_history,
            },
        )

        initial_doc = (
            initial_document
            if isinstance(initial_document, SkillDocument)
            else SkillDocument(initial_document)
        )
        artifacts_dir = self._artifact_dir(
            initial_doc,
            options,
            artifacts_root=self._artifacts_root,
        )
        if options.write_artifacts and artifacts_dir is not None:
            logger.info(
                "Skill train artifacts will be written to %s; artifacts may contain "
                "training-data-derived content.",
                artifacts_dir,
            )
            result.artifacts_dir = write_artifacts(artifacts_dir, result, history)
        return result

    def _resolve_datasets(self) -> tuple[Dataset, Dataset, Dataset | None]:
        options = self._options
        if self._selection_dataset is None and self._test_dataset is None:
            return split_dataset(
                self._dataset,
                options.selection_split,
                options.test_split,
                options.split_seed,
            )

        if self._selection_dataset is None:
            held_out_ids = set(_dataset_example_ids(self._test_dataset))
            split_source = _dataset_without_ids(
                self._dataset,
                held_out_ids,
                f"{self._dataset.name or 'dataset'}__train_selection_source",
            )
            train, selection, _ = split_dataset(
                split_source,
                options.selection_split,
                0.0,
                options.split_seed,
            )
            return train, selection, self._test_dataset

        selection = self._selection_dataset
        held_out_ids = {
            *_dataset_example_ids(selection),
            *_dataset_example_ids(self._test_dataset),
        }
        train = _dataset_without_ids(
            self._dataset,
            held_out_ids,
            f"{self._dataset.name or 'dataset'}__train",
        )
        if not train.examples:
            raise ValueError(
                "explicit skill training splits leave no training examples"
            )
        return train, selection, self._test_dataset

    def _evaluate_cached(
        self, document: SkillDocument, dataset: Dataset
    ) -> tuple[float, list[RolloutRecord]]:
        key = (document.doc_hash, _dataset_fingerprint(dataset))
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

    def _selection_delta(self, candidate: float, current: float) -> float:
        if self._options.higher_is_better:
            return candidate - current
        return current - candidate

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
            if self._options.higher_is_better:
                return bool(float(value) < float(self._options.failure_threshold))
            return bool(float(value) > float(self._options.failure_threshold))
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
            edits.extend(
                self._call_analyze(
                    document,
                    batch,
                    polarity,
                    max_edits,
                )
            )
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
        return self._call_merge(failure_edits, success_edits, max_edits)

    def _maybe_apply_slow_update(
        self,
        *,
        previous_epoch_best: SkillDocument,
        current: SkillDocument,
        current_selection_score: float,
        slow_probe: Dataset,
        selection: Dataset,
        epoch: int,
        accepted_edits: list[EditApplyRecord],
        all_edit_records: list[EditApplyRecord],
        history: list[dict[str, Any]],
        epoch_summary: dict[str, Any],
    ) -> tuple[SkillDocument, float]:
        if self._gate_limit_reached(epoch_summary):
            epoch_summary["slow_update"] = "skipped_gate_limit"
            return current, current_selection_score
        if self._optimizer_limit_reached():
            epoch_summary["slow_update"] = "skipped_optimizer_limit"
            return current, current_selection_score

        _, previous_rollouts = self._evaluate_cached(previous_epoch_best, slow_probe)
        _, current_rollouts = self._evaluate_cached(current, slow_probe)
        categorized = categorize_slow_update_rollouts(
            self._mark_rollout_failures(previous_rollouts),
            self._mark_rollout_failures(current_rollouts),
        )
        prior_guidance = extract_slow_update_content(current.text)

        slow_update_method = getattr(self._reflector, "slow_update", None)
        if not callable(slow_update_method):
            epoch_summary["slow_update"] = "skipped_reflector_unsupported"
            return current, current_selection_score

        self._optimizer_calls += 1
        guidance = slow_update_method(
            previous_epoch_best,
            current,
            categorized,
            prior_guidance,
        )
        if not isinstance(guidance, str) or not guidance.strip():
            epoch_summary["slow_update"] = "skipped_empty"
            return current, current_selection_score

        candidate_text, record = apply_slow_update(
            current.text,
            guidance,
            epoch=epoch,
            step=self._options.steps_per_epoch,
        )
        record.selection_score_before = current_selection_score
        all_edit_records.append(record)
        if record.status != "applied":
            epoch_summary["slow_update"] = record.status
            return current, current_selection_score

        candidate = SkillDocument(
            candidate_text,
            version=current.version + 1,
            parent_hash=current.doc_hash,
        )
        candidate_score, _ = self._evaluate_cached(candidate, selection)
        self._gate_evaluations += 1
        record.selection_score_after = candidate_score
        selection_delta = self._selection_delta(
            candidate_score, current_selection_score
        )
        accepted = self._is_improvement(candidate_score, current_selection_score)
        decision = "accepted" if accepted else "rejected"
        history.append(
            {
                "doc_hash": candidate.doc_hash,
                "split": selection.name,
                "score": candidate_score,
                "decision": f"slow_update_{decision}",
            }
        )
        if accepted:
            accepted_edits.append(record)
            epoch_summary["slow_update"] = "accepted"
            self._record_history(
                self._accept_history, [record], selection_delta, epoch, record.step
            )
            return candidate, candidate_score

        record.status = "rejected_gate"
        epoch_summary["slow_update"] = "rejected_gate"
        self._record_gate_rejection(
            [record],
            selection_delta,
            epoch,
            record.step,
        )
        return current, current_selection_score

    def _update_meta_skill(self, epoch_summary: dict[str, Any]) -> None:
        if not self._options.meta_skill:
            return
        meta_method = getattr(self._reflector, "meta_skill", None)
        if not callable(meta_method):
            epoch_summary["meta_skill"] = "skipped_reflector_unsupported"
            return
        if self._optimizer_limit_reached():
            epoch_summary["meta_skill"] = "skipped_optimizer_limit"
            return

        self._optimizer_calls += 1
        content = meta_method(
            self._accept_history,
            self._reject_history,
            self._meta_skill,
        )
        if isinstance(content, str) and content.strip():
            self._meta_skill = content.strip()
            epoch_summary["meta_skill"] = "updated"
        else:
            epoch_summary["meta_skill"] = "skipped_empty"

    def _mark_rollout_failures(
        self, rollouts: Sequence[RolloutRecord]
    ) -> list[RolloutRecord]:
        return [
            replace(rollout, is_failure=self._is_rollout_failure(rollout))
            for rollout in rollouts
        ]

    def _record_gate_rejection(
        self,
        records: list[EditApplyRecord],
        selection_delta: float,
        epoch: int,
        step: int,
    ) -> None:
        edits = [record.edit for record in records]
        if self._rejected_buffer is not None:
            self._rejected_buffer.record(
                edits,
                selection_delta=selection_delta,
                epoch=epoch,
                step=step,
            )
        self._record_history(
            self._reject_history, records, selection_delta, epoch, step
        )

    @staticmethod
    def _record_history(
        target: list[dict[str, object]],
        records: list[EditApplyRecord],
        selection_delta: float,
        epoch: int,
        step: int,
    ) -> None:
        for record in records:
            target.append(
                {
                    "edit_id": record.edit.edit_id,
                    "op": record.edit.op,
                    "source_type": record.edit.source_type,
                    "rationale": record.edit.rationale,
                    "selection_delta": selection_delta,
                    "epoch": epoch,
                    "step": step,
                    "status": record.status,
                }
            )

    def _rejected_digest(self) -> str | None:
        if self._rejected_buffer is None:
            return None
        digest = self._rejected_buffer.digest()
        return digest or None

    def _call_analyze(
        self,
        document: SkillDocument,
        batch: list[RolloutRecord],
        polarity: Literal["failure", "success"],
        max_edits: int,
    ) -> list[EditOp]:
        kwargs = self._prompt_context_kwargs(self._reflector.analyze)
        return self._reflector.analyze(document, batch, polarity, max_edits, **kwargs)

    def _call_merge(
        self,
        failure_edits: list[EditOp],
        success_edits: list[EditOp],
        max_edits: int,
    ) -> list[EditOp]:
        kwargs = self._prompt_context_kwargs(self._reflector.merge)
        return self._reflector.merge(failure_edits, success_edits, max_edits, **kwargs)

    def _prompt_context_kwargs(self, method: Callable[..., Any]) -> dict[str, str]:
        kwargs: dict[str, str] = {}
        if _method_accepts_kwarg(method, "rejected_digest"):
            digest = self._rejected_digest()
            if digest:
                kwargs["rejected_digest"] = digest
        if _method_accepts_kwarg(method, "meta_skill") and self._meta_skill:
            kwargs["meta_skill"] = self._meta_skill
        return kwargs

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
        initial_document: SkillDocument,
        options: SkillTrainOptions,
        *,
        artifacts_root: Path | None = None,
    ) -> str | None:
        if options.artifacts_dir:
            return str(options.artifacts_dir)
        run_basis = f"{initial_document.doc_hash}:{options.split_seed}:{time.time()}"
        run_id = hashlib.sha256(run_basis.encode("utf-8")).hexdigest()[:12]
        root = artifacts_root if artifacts_root is not None else Path(".traigent")
        return str(root / "skill_train" / run_id)


def _resolve_metric_name(metrics: dict[str, float]) -> str | None:
    for preferred in ("accuracy", "score", "primary"):
        if preferred in metrics:
            return preferred
    if metrics:
        return sorted(metrics)[0]
    return None


def _dataset_fingerprint(dataset: Dataset) -> str:
    payload = json.dumps(
        _dataset_example_ids(dataset),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _dataset_example_ids(dataset: Dataset | None) -> list[str]:
    if dataset is None:
        return []
    return [_dataset_example_id(example) for example in dataset.examples]


def _dataset_example_id(example: Any) -> str:
    metadata = dict(getattr(example, "metadata", {}) or {})
    explicit = metadata.get("example_id")
    if explicit not in (None, ""):
        return str(explicit)

    metadata.pop("example_id", None)
    payload = json.dumps(
        {
            "input_data": getattr(example, "input_data", None),
            "expected_output": getattr(example, "expected_output", None),
            "metadata": metadata,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _dataset_without_ids(
    dataset: Dataset, excluded_ids: set[str], name: str
) -> Dataset:
    return Dataset(
        examples=[
            example
            for example in dataset.examples
            if _dataset_example_id(example) not in excluded_ids
        ],
        name=name,
        description=dataset.description,
        metadata=dict(dataset.metadata or {}),
    )


def _method_accepts_kwarg(method: Callable[..., Any], name: str) -> bool:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return True
    return name in signature.parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


__all__ = ["RolloutRecord", "SkillTrainer", "SkillTrainResult"]
