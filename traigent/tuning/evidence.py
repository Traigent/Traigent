"""Local promotion evidence: a frozen candidate measured on a customer holdout.

The customer supplies ``checkers`` (local functions ``(example, output) -> label``) and a
``holdout`` (a JSONL file of ``{"input": {...}, "output": <expected>}`` examples).  This
module runs the agent function with a configuration on every holdout example, applies
each checker, and builds the counts ``{n, k, u, pl, complete}`` that TVL's promotion
monitors consume (RFC 0003 section 6.11, ``monitor_outcomes``).  Verdicts and the
legacy promotion decision are computed by TVL (``tvl.promotion_monitor``,
``tvl.promotion``); nothing is decided here.  Holdout content, checker labels and agent
outputs never leave the process: only counts, verdicts and a digest of the holdout file
are returned.  Anything that cannot be computed honestly raises
:class:`EvidenceUnavailable`, and the candidate stays ``NoDecision``.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

E_NO_PROMOTION_EVIDENCE = "E_NO_PROMOTION_EVIDENCE"
MEASURED_OBJECTIVE = "accuracy"  # exact match against the holdout's expected output


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
    from traigent.config.context import ConfigurationContext

    out: list[tuple[bool, Any]] = []
    for example in examples:
        # every invocation gets its own deep copies; the originals stay immutable
        try:
            with ConfigurationContext(copy.deepcopy(dict(config))):
                out.append((True, _call(fn, copy.deepcopy(example["input"]))))
        except Exception:  # noqa: BLE001 - a failed call is unresolved, content never echoed
            out.append((False, None))
    return out


def monitor_counts(
    checker: Callable[[Any, Any], Any],
    violating: Sequence[str],
    examples: Sequence[Mapping[str, Any]],
    runs: Sequence[tuple[bool, Any]],
) -> dict[str, Any]:
    """``{n, k, u, pl, complete}`` of the violation predicate over the evaluation set."""
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
        elif label in violating:
            k += 1
    return {"n": n, "k": k, "u": u, "pl": 0, "complete": True}


def _accuracy(
    examples: Sequence[Mapping[str, Any]], runs: Sequence[tuple[bool, Any]]
) -> list[float]:
    return [
        1.0 if ok and output == example.get("output") else 0.0
        for example, (ok, output) in zip(examples, runs, strict=True)
    ]


def build_promotion_evidence(
    module: Mapping[str, Any],
    *,
    fn: Callable[..., Any] | None,
    checkers: Mapping[str, Callable[[Any, Any], Any]] | None,
    holdout: Path | str | None,
    candidate_config: Mapping[str, Any],
    incumbent_config: Mapping[str, Any],
) -> PromotionEvidence:
    """Measure the frozen candidate (and the incumbent, paired) on the holdout.

    Raises :class:`EvidenceUnavailable` when the evidence cannot be computed honestly.
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
    if {o.get("name") for o in objectives} != {MEASURED_OBJECTIVE}:
        raise EvidenceUnavailable("objective_not_measurable")
    for monitor in monitors:
        requirement = requirements.get(str(monitor.get("requirement")))
        if requirement is None:
            raise EvidenceUnavailable("requirement_missing")
        if "among" in requirement:
            # RFC 0003 s11.4.2 population: not evaluated locally; refuse, never count
            # out-of-population examples in the denominator.
            raise EvidenceUnavailable("requirement_among_unsupported")
        checker_name = str((requirement.get("violation") or {}).get("checker"))
        if not callable((checkers or {}).get(checker_name)):
            raise EvidenceUnavailable("checker_missing")
    digest, examples = read_holdout(holdout)
    candidate_runs = run_holdout(fn, candidate_config, examples)
    incumbent_runs = run_holdout(fn, incumbent_config, examples)

    from tvl.promotion import epsilon_pareto_gate
    from tvl.promotion_monitor import evaluate_promotion_monitor

    verdicts: dict[str, str] = {}
    all_counts: dict[str, dict[str, Any]] = {}
    for monitor in monitors:
        requirement = requirements[monitor["requirement"]]
        violation = requirement["violation"]
        counts = monitor_counts(
            (checkers or {})[violation["checker"]],
            list(violation.get("label_in", [])),
            examples,
            candidate_runs,
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
    # The legacy decision is TVL's own comparison with the incumbent, paired on the
    # same holdout; monitors are joined by the relation, not here.
    try:
        legacy, _evidence = epsilon_pareto_gate(
            {
                "objective_values": {
                    MEASURED_OBJECTIVE: {"samples": _accuracy(examples, incumbent_runs)}
                }
            },
            {
                "objective_values": {
                    MEASURED_OBJECTIVE: {
                        "samples": _accuracy(examples, candidate_runs),
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
    return PromotionEvidence(str(legacy), verdicts, digest, all_counts)


__all__ = [
    "E_NO_PROMOTION_EVIDENCE",
    "EvidenceUnavailable",
    "PromotionEvidence",
    "build_promotion_evidence",
    "holdout_digest",
    "load_holdout",
    "read_holdout",
    "monitor_counts",
    "run_holdout",
]
