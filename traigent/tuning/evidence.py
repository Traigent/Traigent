"""Local promotion evidence: a frozen candidate measured on a customer holdout.

The customer supplies ``checkers`` (local functions ``(example, output) -> label``) and a
``holdout`` (a JSONL file of ``{"input": {...}, "output": <expected>}`` examples; every
row must carry its expected output).  This module runs the agent function with the
candidate AND the incumbent configuration on every holdout example (interleaved per
row, so provider drift hits both alike), applies each checker, and builds the counts ``{n, k, u, pl, complete}`` that TVL's promotion
monitors consume (RFC 0003 section 6.11, ``monitor_outcomes``).  Verdicts and the
legacy promotion decision are computed by TVL (``tvl.promotion_monitor``,
``tvl.promotion``); nothing is decided here.  The controller records and returns only
counts, verdicts and a digest of the holdout file: it never stores or returns holdout
content, checker labels or agent outputs.  The agent function and the customer's
checkers run as configured and keep their own network behaviour (an agent may call a
hosted model with each row's input); this module does not isolate them.  Anything that cannot be computed honestly raises
:class:`EvidenceUnavailable`, and the candidate stays ``NoDecision``.

Soundness rules: a call that raised is *unresolved* (never a score of 0.0) and counts
toward ``max_unresolved_fraction``; a checker answer outside the requirement's declared
label set is unresolved and reported; a holdout equal to the optimization's own
evaluation set is refused (it cannot be held aside).  Reusing one holdout across many
decisions selects against it: see the guide ("holdout reuse").
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

E_NO_PROMOTION_EVIDENCE = "E_NO_PROMOTION_EVIDENCE"
MEASURED_OBJECTIVE = "accuracy"  # exact match against the holdout's expected output
DEFAULT_MAX_UNRESOLVED_FRACTION = 0.05
E_CHECKER_LABEL_OUT_OF_SET = "E_CHECKER_LABEL_OUT_OF_SET"
COST_UNKNOWN = "COST_UNKNOWN_UNPRICED_MODEL"


class EvidenceUnavailable(Exception):  # noqa: N818 - a closed reason, not an error type
    """Promotion evidence could not be computed; ``reason`` is a closed identifier."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class PromotionEvidence:
    """What the controller records for one candidate (closed values only)."""

    legacy_decision: str
    monitor_verdicts: dict[str, str]
    holdout_digest: str
    counts: dict[str, dict[str, Any]]
    n: int = 0  # holdout rows
    unresolved_rows: int = 0  # rows left out of the paired comparison
    errors: tuple[str, ...] = ()  # closed codes the approver must see


def _read_bytes(path: Path | str) -> bytes:
    try:
        return Path(path).read_bytes()
    except OSError:
        raise EvidenceUnavailable("holdout_unreadable") from None


def _digest_of(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _parse_holdout(data: bytes) -> list[dict[str, Any]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise EvidenceUnavailable("holdout_invalid") from None
    examples: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except ValueError:
            raise EvidenceUnavailable("holdout_invalid") from None
        if not isinstance(item, dict) or not isinstance(item.get("input"), dict):
            raise EvidenceUnavailable("holdout_invalid")
        if item.get("output") is None:  # exact-match scoring needs an expected output
            raise EvidenceUnavailable("holdout_missing_expected_output")
        examples.append(item)
    if not examples:
        raise EvidenceUnavailable("holdout_empty")
    return examples


def read_holdout(path: Path | str) -> tuple[str, list[dict[str, Any]]]:
    """``(digest, examples)`` from ONE read: the digest covers the parsed bytes."""
    data = _read_bytes(path)
    return _digest_of(data), _parse_holdout(data)


def holdout_digest(path: Path | str) -> str:
    """``sha256:`` of the holdout file's bytes."""
    return _digest_of(_read_bytes(path))


def load_holdout(path: Path | str) -> list[dict[str, Any]]:
    """The holdout examples; each is a mapping with an ``input`` mapping."""
    return _parse_holdout(_read_bytes(path))


def _call(fn: Callable[..., Any], arguments: Mapping[str, Any]) -> Any:
    result = fn(**arguments)
    if inspect.isawaitable(result):
        return asyncio.run(_await(result))
    return result


async def _await(awaitable: Any) -> Any:
    return await awaitable


def _run_one(
    fn: Callable[..., Any], config: Mapping[str, Any], example: Mapping[str, Any]
) -> tuple[bool, Any]:
    from traigent.config.context import ConfigurationContext

    # every invocation gets its own deep copies; the originals stay immutable
    try:
        with ConfigurationContext(copy.deepcopy(dict(config))):
            return True, _call(fn, copy.deepcopy(example["input"]))
    except Exception:  # noqa: BLE001 - a failed call is unresolved, content never echoed
        return False, None


def run_holdout(
    fn: Callable[..., Any],
    config: Mapping[str, Any],
    examples: Sequence[Mapping[str, Any]],
) -> list[tuple[bool, Any]]:
    """Run ``fn`` under ``config`` on each example: ``(ok, output)`` per example.

    The configuration is applied through the SDK's call-scoped config context, so
    the live function's own state is never changed.  An example whose call raises is
    ``(False, None)`` (unresolved), never dropped.
    """
    return [_run_one(fn, config, example) for example in examples]


def run_holdout_paired(
    fn: Callable[..., Any],
    candidate_config: Mapping[str, Any],
    incumbent_config: Mapping[str, Any],
    examples: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[bool, Any]], list[tuple[bool, Any]]]:
    """Run candidate and incumbent on each example in turn (paired, interleaved).

    Both configurations see the same example back to back, and the order alternates
    by row, so a provider outage or drift during the run hits both sides alike
    instead of confounding two sequential passes.
    """
    candidate: list[tuple[bool, Any]] = []
    incumbent: list[tuple[bool, Any]] = []
    for index, example in enumerate(examples):
        if index % 2 == 0:
            candidate.append(_run_one(fn, candidate_config, example))
            incumbent.append(_run_one(fn, incumbent_config, example))
        else:
            incumbent.append(_run_one(fn, incumbent_config, example))
            candidate.append(_run_one(fn, candidate_config, example))
    return candidate, incumbent


def monitor_counts(
    checker: Callable[[Any, Any], Any],
    violating: Sequence[str],
    examples: Sequence[Mapping[str, Any]],
    runs: Sequence[tuple[bool, Any]],
    labels: Collection[str] | None = None,
    stats: dict[str, int] | None = None,
) -> dict[str, Any]:
    """``{n, k, u, pl, complete}`` of the violation predicate over the evaluation set.

    A failed call, a checker error, a non-string answer and (when ``labels`` is
    given) an answer outside the declared label set are all *unresolved*; the last
    kind is also tallied in ``stats["out_of_label_set"]``.
    """
    n = len(examples)
    k = u = 0
    for example, (ok, output) in zip(examples, runs, strict=True):
        if not ok:
            u += 1
            continue
        try:
            label = checker(copy.deepcopy(example), copy.deepcopy(output))
        except Exception:  # noqa: BLE001 - a checker error is unresolved
            u += 1
            continue
        if not isinstance(label, str):
            u += 1
        elif labels is not None and label not in labels:
            u += 1
            if stats is not None:
                stats["out_of_label_set"] = stats.get("out_of_label_set", 0) + 1
        elif label in violating:
            k += 1
    return {"n": n, "k": k, "u": u, "pl": 0, "complete": True}


def _accuracy(
    examples: Sequence[Mapping[str, Any]], runs: Sequence[tuple[bool, Any]]
) -> list[float | None]:
    """Exact match per row; ``None`` (unresolved) when the call failed."""
    return [
        (1.0 if output == example.get("output") else 0.0) if ok else None
        for example, (ok, output) in zip(examples, runs, strict=True)
    ]


def examples_digest(examples: Sequence[Mapping[str, Any]]) -> str:
    """``sha256:`` of the canonical ``(input, output)`` content of a set of examples."""
    rows = [{"input": e.get("input"), "output": e.get("output")} for e in examples]
    body = json.dumps(rows, sort_keys=True, separators=(",", ":"), default=str)
    return _digest_of(body.encode("utf-8"))


def _dataset_rows(dataset: Any) -> list[dict[str, Any]]:
    """``{"input", "output"}`` rows of a loaded ``Dataset``."""
    return [
        {"input": ex.input_data, "output": ex.expected_output}
        for ex in dataset.examples
    ]


def dataset_digests(dataset: Any) -> set[str]:
    """Content digest of a loaded ``Dataset`` (the same digest a holdout is checked by)."""
    return {examples_digest(_dataset_rows(dataset))}


def eval_set_identity(
    dataset: Any, loader: Callable[[Any], Any] | None = None
) -> tuple[set[str], list[dict[str, Any]] | None]:
    """Digests that identify the optimization's evaluation set, and its rows.

    ``dataset`` is the optimized function's ``eval_dataset`` in any form the
    optimizer accepts (a path, a list of paths, a ``Dataset``, inline rows).  The
    rows are loaded with the OPTIMIZER'S OWN loader (``loader``) so the identity is
    what the optimizer really evaluated on.  When an evaluation set exists but its
    identity cannot be established, :class:`EvidenceUnavailable` is raised: a
    holdout that might be the optimization's own data is never promoted on.
    """
    if dataset is None:
        return set(), None
    if loader is None:
        # the optimizer's own resolution, not a re-implementation (it only needs
        # ``self`` for its static helper, so the class itself serves)
        from traigent.core.optimized_function import OptimizedFunction

        def loader(source: Any) -> Any:
            return OptimizedFunction._dataset_from_source(
                cast(Any, OptimizedFunction), source
            )

    try:
        loaded = loader(dataset)
        rows = _dataset_rows(loaded)
    except Exception:  # noqa: BLE001 - closed reason; never echo content
        raise EvidenceUnavailable("eval_set_identity_unknown") from None
    digests = {examples_digest(rows)}
    paths: list[Any] = []
    if isinstance(dataset, (str, Path)):
        paths = [dataset]
    elif isinstance(dataset, (list, tuple)):
        paths = [p for p in dataset if isinstance(p, (str, Path))]
    for path in paths:  # the files' own bytes, so a holdout copy is recognised too
        try:
            digests.add(_digest_of(Path(path).read_bytes()))
        except OSError:
            raise EvidenceUnavailable("eval_set_identity_unknown") from None
    return digests, rows


def _is_cost_objective(name: str) -> bool:
    lowered = name.lower()
    return any(word in lowered for word in ("cost", "price", "spend"))


def _unresolved_limit(monitors: Sequence[Mapping[str, Any]]) -> float:
    declared = [
        float(m["decision"]["max_unresolved_fraction"])
        for m in monitors
        if isinstance(m.get("decision"), Mapping)
        and "max_unresolved_fraction" in m["decision"]
    ]
    return min(declared) if declared else DEFAULT_MAX_UNRESOLVED_FRACTION


def build_promotion_evidence(
    module: Mapping[str, Any],
    *,
    fn: Callable[..., Any] | None,
    checkers: Mapping[str, Callable[[Any, Any], Any]] | None,
    holdout: Path | str | None,
    candidate_config: Mapping[str, Any],
    incumbent_config: Mapping[str, Any],
    checker_labels: Mapping[str, Sequence[str]] | None = None,
    eval_set_digests: Collection[str] = (),
    unpriced_models: Collection[str] = (),
) -> PromotionEvidence:
    """Measure the frozen candidate and the incumbent, paired, on the holdout.

    ``checker_labels`` maps each checker to the full label set it may answer with
    (TVL's checker registry); an answer outside it is unresolved.  A holdout whose
    byte or content digest is in ``eval_set_digests`` (the optimization's own
    evaluation set) is refused.  ``unpriced_models`` names the candidate's models
    with no price: their cost is unknown, never zero, so when cost is an objective
    the candidate is not promotable on cost (``cost_unknown_unpriced_model``) and
    otherwise the gate carries ``COST_UNKNOWN_UNPRICED_MODEL``.  Raises
    :class:`EvidenceUnavailable` when the evidence cannot be computed honestly.
    """
    monitors = [m for m in module.get("monitors", []) if m.get("phase") == "promotion"]
    if not monitors:
        raise EvidenceUnavailable("no_promotion_monitor")
    if holdout is None:
        raise EvidenceUnavailable("holdout_missing")
    if fn is None:
        raise EvidenceUnavailable("agent_function_missing")
    requirements = {r["name"]: r for r in module.get("requirements", [])}
    objectives = list(module.get("objectives", []))
    names = {str(o.get("name")) for o in objectives}
    if unpriced_models and any(_is_cost_objective(n) for n in names):
        raise EvidenceUnavailable("cost_unknown_unpriced_model")
    if names != {MEASURED_OBJECTIVE}:
        raise EvidenceUnavailable("objective_not_measurable")
    label_sets: dict[str, frozenset[str]] = {}
    for monitor in monitors:
        requirement = requirements.get(str(monitor.get("requirement")))
        if requirement is None:
            raise EvidenceUnavailable("requirement_missing")
        if "among" in requirement:
            # RFC 0003 s11.4.2 population: not evaluated locally; refuse, never count
            # out-of-population examples in the denominator.
            raise EvidenceUnavailable("requirement_among_unsupported")
        violation = requirement.get("violation") or {}
        checker_name = str(violation.get("checker"))
        if not callable((checkers or {}).get(checker_name)):
            raise EvidenceUnavailable("checker_missing")
        declared = (checker_labels or {}).get(checker_name)
        if not declared:
            raise EvidenceUnavailable("checker_labels_missing")
        label_sets[checker_name] = frozenset(str(x) for x in declared)
        if not set(violation.get("label_in", [])) <= label_sets[checker_name]:
            raise EvidenceUnavailable("checker_labels_inconsistent")
    digest, examples = read_holdout(holdout)
    held = set(eval_set_digests)
    if digest in held or examples_digest(examples) in held:
        raise EvidenceUnavailable("holdout_is_optimization_eval_set")
    candidate_runs, incumbent_runs = run_holdout_paired(
        fn, candidate_config, incumbent_config, examples
    )

    from tvl.promotion import epsilon_pareto_gate
    from tvl.promotion_monitor import evaluate_promotion_monitor

    verdicts: dict[str, str] = {}
    all_counts: dict[str, dict[str, Any]] = {}
    stats: dict[str, int] = {}
    for monitor in monitors:
        requirement = requirements[monitor["requirement"]]
        violation = requirement["violation"]
        counts = monitor_counts(
            (checkers or {})[violation["checker"]],
            list(violation.get("label_in", [])),
            examples,
            candidate_runs,
            labels=label_sets[violation["checker"]],
            stats=stats,
        )
        try:
            result = evaluate_promotion_monitor(requirement, monitor, counts)
        except Exception:  # noqa: BLE001 - TVL refusals are closed reasons
            raise EvidenceUnavailable("monitor_not_evaluable") from None
        verdict = str(result["verdict"])
        if verdict == "UNKNOWN":
            verdict = f"UNKNOWN({result['reason']})"
        verdicts[monitor["name"]] = verdict
        all_counts[monitor["name"]] = counts
    errors: list[str] = []
    if stats.get("out_of_label_set"):
        errors.append(E_CHECKER_LABEL_OUT_OF_SET)
    if unpriced_models:
        errors.append(COST_UNKNOWN)
    # The legacy decision is TVL's own comparison with the incumbent, paired on the
    # same holdout rows.  A row either side could not resolve is left out of the pair
    # (never scored 0.0) and counts toward the unresolved fraction.
    mine = _accuracy(examples, candidate_runs)
    theirs = _accuracy(examples, incumbent_runs)
    pairs = [
        (a, b)
        for a, b in zip(mine, theirs, strict=True)
        if a is not None and b is not None
    ]
    left_out = len(examples) - len(pairs)
    if left_out / len(examples) > _unresolved_limit(monitors):
        raise EvidenceUnavailable("too_many_unresolved")
    try:
        legacy, _evidence = epsilon_pareto_gate(
            {
                "objective_values": {
                    MEASURED_OBJECTIVE: {"samples": [b for _a, b in pairs]}
                }
            },
            {
                "objective_values": {
                    MEASURED_OBJECTIVE: {
                        "samples": [a for a, _b in pairs],
                        "paired": True,
                    }
                }
            },
            dict(module.get("promotion_policy") or {}),
            objectives,
            requirements=list(requirements.values()),
        )
    except Exception:  # noqa: BLE001 - e.g. scipy missing: no honest comparison
        raise EvidenceUnavailable("comparison_not_computable") from None
    if legacy not in {"Promote", "Reject", "NoDecision"}:
        raise EvidenceUnavailable("comparison_not_computable")
    return PromotionEvidence(
        str(legacy),
        verdicts,
        digest,
        all_counts,
        n=len(examples),
        unresolved_rows=left_out,
        errors=tuple(errors),
    )


__all__ = [
    "DEFAULT_MAX_UNRESOLVED_FRACTION",
    "COST_UNKNOWN",
    "E_CHECKER_LABEL_OUT_OF_SET",
    "E_NO_PROMOTION_EVIDENCE",
    "EvidenceUnavailable",
    "PromotionEvidence",
    "build_promotion_evidence",
    "dataset_digests",
    "eval_set_identity",
    "examples_digest",
    "holdout_digest",
    "load_holdout",
    "read_holdout",
    "monitor_counts",
    "run_holdout",
    "run_holdout_paired",
]
