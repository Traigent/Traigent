#!/usr/bin/env python3
"""Track A — SDK surrogate evaluator (F1) — evaluator-coevolution sim.

Demonstrates the REAL shipped SDK surrogate-evaluator feature (F1): a cheap
surrogate judge (``judge_surrogate_lenient``) rides alongside the primary
evaluator over the SAME already-captured outputs, and the SDK attaches (a) a
per-example ``surrogate_score`` on every scored example, (b) the aggregate
mean ``surrogate_score`` on the trial metrics, and (c) a surrogate descriptor
(``evaluator_id``, ``metric_name``, ``config.fingerprint_source``).

What this demo proves, and where:

  1. LOCAL RESULT SHAPE — ``TrialResult.to_dict()`` is the SDK's in-process
     optimization result (what ``optimize_sync`` returns to your code). We assert
     the surrogate descriptor + aggregate ``surrogate_score`` on it AND the
     per-example ``surrogate_score`` on every entry of
     ``metadata['example_results'][i]['metrics']``. This is NOT the backend POST
     payload — the real submission rebuilds metadata from scratch.

  2. OFFLINE WIRE-CARRY PROOF — the real backend submission does NOT ship
     ``to_dict()``; it rebuilds the metadata via
     ``traigent.core.metadata_helpers.build_backend_metadata`` and assembles the
     POSTed dict via ``traigent.cloud.trial_operations.TrialOperations
     ._build_trial_result_data`` (see ``trial_operations`` ~557-577 / ~836-861).
     Both are pure, network-free builders, so we run them OFFLINE on the trial
     and assert the surrogate descriptor (``metadata.surrogate_evaluator``), the
     aggregate ``surrogate_score``, AND the per-example ``surrogate_score`` (in
     the rebuilt ``metadata.measures[i].metrics``) all SURVIVE into the built
     submission dict. This proves the SDK-built, POST-shaped submission dict
     carries the surrogate signal — it does NOT exercise the network, and it
     does NOT prove anything about backend ingest or cell-splitting into a
     configs x examples x evaluators tensor (that is demonstrated separately,
     by the backend-side simulation of feature F2 in TraigentBackend).

Mock boundary: the "agent" is a pure-Python synthetic model — no LLM call, no
network egress at all — so the ONLY thing that is faked here is the model's
output. The surrogate/primary scoring, the config sweep, the trial lifecycle,
the descriptor construction, the fingerprint hashing, and the
submission-payload builders are all the REAL shipped ``traigent`` SDK code
(traigent/api/decorators.py, traigent/core/optimization_pipeline.py,
traigent/core/trial_lifecycle.py, traigent/core/metadata_helpers.py,
traigent/cloud/trial_operations.py, traigent/utils/artifact_fingerprints.py) —
nothing about F1 itself is reimplemented or stubbed.

Self-contained: the small scenario slice this demo needs (the 10 dataset
examples and the surrogate's expected constant/aggregate score) is inlined
below as plain Python constants — see SCENARIO_EXAMPLES / TRACK_A_SURROGATE.
This mirrors (a subset of) the shared workspace scenario at
``demos/evaluator-coevolution-sim/scenario.json``; nothing is read from disk
at runtime, so this demo runs from a fresh SDK checkout with nothing but this
repo.

Run (from this repo root, keyless / offline, no venv needed):

    TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true ENVIRONMENT=development \\
    PYTHONPATH=$PWD python3 examples/advanced/surrogate_evaluator_coevolution_demo.py

Exits 0 and prints ``[TRIGGERED] F1 ...`` when the local-shape AND wire-carry
assertions pass; exits 1 and prints ``[NOT TRIGGERED]`` otherwise.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# --- mock boundary: belt-and-suspenders offline/no-egress env (the agent
# below never calls an LLM at all, so these are defensive, not load-bearing).
os.environ.setdefault("TRAIGENT_MOCK_LLM", "true")
os.environ.setdefault("TRAIGENT_OFFLINE_MODE", "true")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")

try:
    import traigent
except ImportError:  # pragma: no cover - support IDE / ad-hoc execution paths
    _sdk = os.environ.get("TRAIGENT_SDK_PATH") or str(
        Path(__file__).resolve().parents[2]
    )
    sys.path.insert(0, _sdk)
    import traigent

from traigent.api.decorators import EvaluationOptions  # noqa: E402
from traigent.cloud.trial_operations import TrialOperations  # noqa: E402
from traigent.config.types import TraigentConfig  # noqa: E402
from traigent.core.metadata_helpers import build_backend_metadata  # noqa: E402

FP_RE = re.compile(r"^fp1:[0-9a-f]{64}$")

# --- inlined scenario slice --------------------------------------------------
# Mirrors demos/evaluator-coevolution-sim/scenario.json's "examples" (id,
# input, expected_output only — the fields Track A needs) and
# "track_a_surrogate" sections, so this demo needs nothing but this file.
SCENARIO_EXAMPLES: list[dict[str, str]] = [
    {"id": "ex01", "input": "2 + 2 = ?", "expected_output": "4"},
    {"id": "ex02", "input": "Capital of France?", "expected_output": "paris"},
    {"id": "ex03", "input": "3 * 3 = ?", "expected_output": "9"},
    {"id": "ex04", "input": "Chemical symbol for gold?", "expected_output": "au"},
    {
        "id": "ex05",
        "input": "Largest planet in the solar system?",
        "expected_output": "jupiter",
    },
    {"id": "ex06", "input": "Square root of 144?", "expected_output": "12"},
    {"id": "ex07", "input": "Is water H2O? (yes/no)", "expected_output": "yes"},
    {"id": "ex08", "input": "Opposite of 'hot'?", "expected_output": "cold"},
    {
        "id": "ex09",
        "input": "How many legs does a spider have?",
        "expected_output": "8",
    },
    {
        "id": "ex10",
        "input": "Author of 'Romeo and Juliet'?",
        "expected_output": "shakespeare",
    },
]

TRACK_A_SURROGATE = {
    "surrogate_evaluator_name": "judge_surrogate_lenient",
    "surrogate_constant_score": 0.90,
    "expected_aggregate_surrogate_score": 0.90,
}

SURROGATE_NAME = TRACK_A_SURROGATE["surrogate_evaluator_name"]
SURROGATE_CONSTANT = TRACK_A_SURROGATE["surrogate_constant_score"]
EXPECTED_AGGREGATE_SURROGATE = TRACK_A_SURROGATE["expected_aggregate_surrogate_score"]

# Canonical question -> expected answer, straight from the inlined scenario so
# the synthetic agent's outputs are exactly what the primary exact-match
# scorer expects (correct-answer agent; the surrogate scores those SAME
# captured outputs, never re-executing anything).
_CANONICAL_ANSWERS: dict[str, str] = {
    ex["input"]: ex["expected_output"] for ex in SCENARIO_EXAMPLES
}

eval_dataset = [
    {"input": {"question": ex["input"]}, "expected_output": ex["expected_output"]}
    for ex in SCENARIO_EXAMPLES
]


def polymath_synthetic_agent_impl(question: str) -> str:
    """Pure-Python synthetic PolyMath agent — no LLM call, no network egress.

    Deterministically returns the canonical correct answer for each scenario
    question so the primary evaluator sees real (correct) captured outputs
    for the surrogate to score.
    """
    return _CANONICAL_ANSWERS.get(question, "UNKNOWN")


def exact_match(output, expected, llm_metrics=None) -> float:
    """Primary scoring_function: exact_match(output, expected_output) -> 1.0/0.0."""
    try:
        return (
            1.0 if str(output).strip().lower() == str(expected).strip().lower() else 0.0
        )
    except Exception:
        return 0.0


def judge_surrogate_lenient(output, expected_output=None, example=None) -> float:
    """The CHEAP surrogate evaluator (F1): lenient judge, constant 0.90 score.

    Scores the SAME captured output the primary evaluator already produced —
    it must never re-execute the agent. Per the inlined scenario's
    ``behavior_scores.per_behavior_cell_score.lenient``-derived
    ``TRACK_A_SURROGATE`` wiring, this judge is lenient and reports a
    constant 0.90 regardless of the (correct) output it is handed.
    """
    return SURROGATE_CONSTANT


@traigent.optimize(
    configuration_space={"temperature": [0.0, 0.7]},
    evaluation=EvaluationOptions(
        eval_dataset=eval_dataset,
        scoring_function=exact_match,
        surrogate_evaluator=judge_surrogate_lenient,
        surrogate_evaluator_name=SURROGATE_NAME,
    ),
    offline=True,
    algorithm="grid",
    max_trials=10,
)
def polymath_agent(question: str) -> str:
    return polymath_synthetic_agent_impl(question)


class _CheckFailed(Exception):
    """Raised with a human-readable reason when an assertion fails."""


def _check_local_shape(trial) -> None:
    """Assert the LOCAL SDK optimization result (TrialResult.to_dict()) carries
    the surrogate descriptor, the aggregate surrogate_score, AND a per-example
    surrogate_score on every scored example. This is what optimize_sync returns
    in-process — NOT the backend POST payload.
    """
    payload = trial.to_dict()
    metadata = payload.get("metadata") or {}
    metrics = payload.get("metrics") or {}

    descriptor = metadata.get("surrogate_evaluator")
    if not isinstance(descriptor, dict):
        raise _CheckFailed(
            f"trial {trial.trial_id}: metadata['surrogate_evaluator'] is ABSENT "
            "(surrogate descriptor did not attach to the local result)"
        )

    evaluator_id = descriptor.get("evaluator_id")
    metric_name = descriptor.get("metric_name")
    fingerprint_source = (descriptor.get("config") or {}).get("fingerprint_source")

    if evaluator_id != SURROGATE_NAME:
        raise _CheckFailed(
            f"trial {trial.trial_id}: evaluator_id={evaluator_id!r} != {SURROGATE_NAME!r}"
        )
    if metric_name != "surrogate_score":
        raise _CheckFailed(
            f"trial {trial.trial_id}: metric_name={metric_name!r} != 'surrogate_score'"
        )
    if not (isinstance(fingerprint_source, str) and FP_RE.match(fingerprint_source)):
        raise _CheckFailed(
            f"trial {trial.trial_id}: config.fingerprint_source={fingerprint_source!r} "
            "does not match ^fp1:[0-9a-f]{64}$"
        )

    aggregate_surrogate_score = metrics.get("surrogate_score")
    if aggregate_surrogate_score != EXPECTED_AGGREGATE_SURROGATE:
        raise _CheckFailed(
            f"trial {trial.trial_id}: metrics['surrogate_score']="
            f"{aggregate_surrogate_score!r} != {EXPECTED_AGGREGATE_SURROGATE}"
        )

    # The aggregate alone can be an echo — assert the REAL per-example
    # injection. apply_surrogate_scoring writes surrogate_score into each
    # scored example's metrics (traigent/core/trial_lifecycle.py ~187), which
    # trial_result_factory serialises to metadata['example_results'][i]['metrics']
    # (~265-269). Assert it is present + numeric for EVERY example.
    example_results = metadata.get("example_results")
    if not isinstance(example_results, list) or not example_results:
        raise _CheckFailed(
            f"trial {trial.trial_id}: metadata['example_results'] is missing/empty; "
            "cannot verify per-example surrogate scoring"
        )
    for i, ex in enumerate(example_results):
        ex_metrics = ex.get("metrics") if isinstance(ex, dict) else None
        surrogate = (
            ex_metrics.get("surrogate_score") if isinstance(ex_metrics, dict) else None
        )
        if not isinstance(surrogate, (int, float)) or isinstance(surrogate, bool):
            raise _CheckFailed(
                f"trial {trial.trial_id}: example_results[{i}].metrics['surrogate_score']="
                f"{surrogate!r} is absent or non-numeric (per-example surrogate did not land)"
            )

    n = len(example_results)
    print(
        f"  trial {trial.trial_id}: LOCAL result — descriptor OK, "
        f"aggregate metrics.surrogate_score={aggregate_surrogate_score}, "
        f"per-example surrogate_score present+numeric on all {n}/{n} examples "
        f"(metadata.example_results[i].metrics.surrogate_score)"
    )


def _build_submission_dict(trial, *, privacy: bool) -> dict:
    """Run the REAL, network-free backend submission builders on ``trial`` and
    return the dict that WOULD be POSTed — no HTTP call.

    Mirrors the live submit path in trial_operations (~836-861): rebuild metadata
    from scratch with build_backend_metadata, strip transport-only keys, then
    assemble result_data with _build_trial_result_data. _build_trial_result_data
    reads no instance state (see trial_operations.py ~566-578), so it is called
    unbound with ``self=None`` — no CloudClient / session / network needed.
    """
    cfg = TraigentConfig(privacy_enabled=privacy)
    wire_metadata = build_backend_metadata(
        trial, "accuracy", cfg, dataset_name="dataset"
    )
    clean_metrics = {
        k: v
        for k, v in (trial.metrics or {}).items()
        if k not in ("measures", "summary_stats")
    }
    return TrialOperations._build_trial_result_data(
        None,
        trial.trial_id,
        trial.config,
        clean_metrics,
        "completed",
        "offline",
        metadata=wire_metadata,
    )


def _check_wire_carry(trial, *, privacy: bool) -> None:
    """Assert the surrogate descriptor + aggregate + per-example surrogate_score
    SURVIVE the from-scratch metadata rebuild into the built submission dict.

    This proves the SDK-built, POST-shaped dict carries the surrogate signal.
    It does NOT make a network call and does NOT prove anything about backend
    ingest or cell-splitting — that is demonstrated separately by the backend
    simulation of feature F2 (TraigentBackend).
    """
    label = "privacy=ON" if privacy else "privacy=off"
    result_data = _build_submission_dict(trial, privacy=privacy)
    rmd = result_data.get("metadata") or {}

    descriptor = rmd.get("surrogate_evaluator")
    if (
        not isinstance(descriptor, dict)
        or descriptor.get("evaluator_id") != SURROGATE_NAME
    ):
        raise _CheckFailed(
            f"trial {trial.trial_id} [{label}]: surrogate descriptor did NOT survive the "
            f"backend metadata rebuild (result_data.metadata.surrogate_evaluator={descriptor!r})"
        )
    fp = (descriptor.get("config") or {}).get("fingerprint_source")
    if not (isinstance(fp, str) and FP_RE.match(fp)):
        raise _CheckFailed(
            f"trial {trial.trial_id} [{label}]: descriptor.config.fingerprint_source={fp!r} "
            "did not survive into the submission dict"
        )

    agg = rmd.get("surrogate_score")
    if agg != EXPECTED_AGGREGATE_SURROGATE:
        raise _CheckFailed(
            f"trial {trial.trial_id} [{label}]: aggregate surrogate_score did not survive "
            f"(result_data.metadata.surrogate_score={agg!r} != {EXPECTED_AGGREGATE_SURROGATE})"
        )

    measures = rmd.get("measures")
    if not isinstance(measures, list) or not measures:
        raise _CheckFailed(
            f"trial {trial.trial_id} [{label}]: result_data.metadata.measures missing/empty; "
            "per-example surrogate cannot be verified on the wire"
        )
    for i, m in enumerate(measures):
        m_metrics = m.get("metrics") if isinstance(m, dict) else None
        surrogate = (
            m_metrics.get("surrogate_score") if isinstance(m_metrics, dict) else None
        )
        if not isinstance(surrogate, (int, float)) or isinstance(surrogate, bool):
            raise _CheckFailed(
                f"trial {trial.trial_id} [{label}]: measures[{i}].metrics['surrogate_score']="
                f"{surrogate!r} did not survive into the submission dict"
            )

    n = len(measures)
    print(
        f"  trial {trial.trial_id}: WIRE [{label}] — built submission dict via "
        f"build_backend_metadata + _build_trial_result_data; surrogate descriptor "
        f"survived, aggregate surrogate_score={agg}, per-example surrogate_score "
        f"survived on all {n}/{n} rebuilt measures (result_data.metadata.measures[i].metrics)"
    )


def main() -> int:
    print("=" * 78)
    print("Track A — SDK surrogate evaluator (F1) — evaluator-coevolution sim")
    print("=" * 78)
    print(f"Dataset examples: {len(eval_dataset)} (inlined scenario slice)")
    print(
        f"Surrogate: {SURROGATE_NAME!r} (constant score {SURROGATE_CONSTANT}) "
        "attached via EvaluationOptions(surrogate_evaluator=...)"
    )
    print()

    # offline=True => no backend submission. This is the REAL in-process
    # OptimizationResult the SDK builds. NOTE: TrialResult.to_dict() is the
    # SDK's LOCAL optimization result shape (what optimize_sync returns), NOT
    # the backend POST payload — real submission rebuilds metadata from scratch
    # (see _check_wire_carry below, which runs the actual builders offline).
    result = polymath_agent.optimize_sync(algorithm="grid", max_trials=10)

    print(f"Trials run: {len(result.trials)}")
    print(f"Best config: {result.best_config}")
    print(f"Best score: {result.best_score}")
    print()

    if not result.trials:
        print("[NOT TRIGGERED] F1 SDK surrogate evaluator — no trials were produced")
        return 1

    try:
        print("Local SDK optimization result (TrialResult.to_dict()) checks:")
        for trial in result.trials:
            _check_local_shape(trial)
        print()
        print("Offline wire-carry proof (real backend submission builders, no HTTP):")
        for trial in result.trials:
            # off = full (non-privacy) submission; ON = redacted submission.
            # Assert surrogate survives BOTH paths — it is a content-free
            # numeric, so it is deliberately privacy-carried (Rule 4).
            _check_wire_carry(trial, privacy=False)
            _check_wire_carry(trial, privacy=True)
    except _CheckFailed as exc:
        print()
        print(f"[NOT TRIGGERED] F1 SDK surrogate evaluator — {exc}")
        return 1

    print()
    print(
        "[TRIGGERED] F1 SDK surrogate evaluator — the LOCAL SDK result carries the "
        f"surrogate descriptor {{evaluator_id: {SURROGATE_NAME!r}, "
        "metric_name: 'surrogate_score', config.fingerprint_source: <fp1:sha256>}}, "
        f"aggregate metrics.surrogate_score == {EXPECTED_AGGREGATE_SURROGATE}, AND a "
        "per-example surrogate_score on every scored example; and the descriptor + "
        "aggregate + per-example surrogate_score all SURVIVE the real (offline) backend "
        "submission builders (build_backend_metadata + _build_trial_result_data) into the "
        f"POST-shaped dict, in both full and privacy-redacted modes — real per-example "
        f"scoring by {SURROGATE_NAME!r}, aggregated by the SDK's own apply_surrogate_scoring, "
        "not an echo of these assertions."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
