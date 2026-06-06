#!/usr/bin/env python3
"""Composite knobs: execute a cascade, ride its telemetry on the measures channel.

This example is OFFLINE-BY-DEFAULT and deterministic — it makes NO LLM calls and
talks to NO backend. It demonstrates, end to end:

1. declaring an RFC 0002 ``binary_cascade`` pattern;
2. executing it in-trial over deterministic stub stages with ``execute_composite``
   + ``StageRunner`` (a cheap arm that escalates to an expert arm below a
   calibrated margin threshold);
3. flattening the run's §3.10 content-free telemetry with ``composite_measures``
   and merging it into the per-example metrics — the SAME numeric metrics that
   ride the Traigent *measures* wire channel in hybrid/cloud mode;
4. running the certified-selection promotion gate over the collected metrics to
   produce a CERTIFIED winner or an honest "no winner yet" — no fabricated
   success.

Run it::

    python examples/advanced/composite-knobs/composite_telemetry.py

Environment (all optional; offline by default):

- ``TRAIGENT_OFFLINE_MODE=true`` — already the default for this example; no
  network is touched regardless.

The "Hybrid / cloud wire" section near the bottom is COMMENTED and documented:
it shows where ``composite_measures(run)`` is merged into the metrics a
``@traigent.optimize``-decorated function returns so the composite telemetry
rides the existing trial-submission channel. It is intentionally not executed
here (it needs a live backend + credentials).

NOTE on claim scope: the promotion gate reports a *procedural* decision over the
provided samples (promote / reject / no decision). It does not assert any
guarantee about future inputs; treat its output as "this evidence, this policy".
"""

from __future__ import annotations

import os

# Offline by default: this example never needs a backend.
os.environ.setdefault("TRAIGENT_OFFLINE_MODE", "true")
os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")

from traigent.knobs.patterns import binary_cascade
from traigent.knobs.runtime import StageRunner, execute_composite
from traigent.knobs.telemetry import composite_measures
from traigent.tvl.models import PromotionPolicy
from traigent.tvl.promotion_gate import ObjectiveSpec, PromotionGate

# The calibrated gate threshold (a CVAR name). In a governed run this is
# certificate-backed; here we pass a plain calibrated value for the example.
GATE = "router_margin_threshold"

# The composite under test: a cheap arm that escalates to an expert arm when the
# cheap arm's vote margin falls below the calibrated threshold (§3.2 post-cascade).
COMPOSITE = binary_cascade(
    "answerer",
    base_stage="cheap",
    expert_stage="strong",
    threshold=GATE,
)


def _stage(outputs: list[str]) -> StageRunner:
    """A deterministic voting stage over a fixed output multiset (identity keys).

    Real stages would call a model; here we return fixed samples so the example
    is fully reproducible offline.
    """
    return StageRunner(
        run=lambda _item: list(outputs),
        key_fn=lambda x: x,
        samples=len(outputs),
    )


# Three evaluation items; the "correct" answer is STRONG for each.
_ITEMS = ["q0", "q1", "q2"]
_EXPECTED = "STRONG"


def _evaluate_config(
    variant: str, theta: float
) -> tuple[list[float], dict[str, float]]:
    """Evaluate one candidate config over the dataset (offline, deterministic).

    Returns the per-item accuracy samples and the LAST item's flattened composite
    telemetry (representative for printing). The dominance comparison uses the
    accuracy samples; the composite measures are content-free observability that
    rides the measures channel alongside accuracy.

    Stage behavior, by variant:

    - ``cheap``: unanimous WRONG cheap votes (margin 1.0 >= theta) => ACCEPTED at
      the cheap arm => wrong answer => accuracy 0.
    - ``strong``: split cheap votes (margin 1/3 < theta) => ESCALATES to the
      expert arm => correct answer => accuracy 1.

    The gate therefore creates real, observable dominance for the promotion gate.
    """
    cheap_outputs = (
        ["nope", "nope", "nope"] if variant == "cheap" else ["STRONG", "x", "y"]
    )
    accuracy_samples: list[float] = []
    last_measures: dict[str, float] = {}
    for _item in _ITEMS:
        run = execute_composite(
            COMPOSITE.structure,
            {"cheap": _stage(cheap_outputs), "strong": _stage([_EXPECTED])},
            config={"variant": variant},
            calibrated_values={GATE: theta},
        )
        accuracy_samples.append(1.0 if str(run.output) == _EXPECTED else 0.0)

        # --- The integration recipe (offline form) -----------------------------
        # Merge the composite's §3.10 telemetry into the per-item metrics. In a
        # hybrid/cloud run, this exact dict is what rides the measures channel.
        item_metrics: dict[str, float] = {"accuracy": accuracy_samples[-1]}
        item_metrics.update(composite_measures(run))
        last_measures = item_metrics

    return accuracy_samples, last_measures


def main() -> None:
    theta = 0.7  # calibrated margin threshold

    incumbent_acc, incumbent_measures = _evaluate_config("cheap", theta)
    candidate_acc, candidate_measures = _evaluate_config("strong", theta)

    print("Composite telemetry example (offline, deterministic)\n")
    print(f"  gate threshold (calibrated): {theta}")
    print(f"  incumbent (variant=cheap)  accuracy samples: {incumbent_acc}")
    print(f"    measures: {incumbent_measures}")
    print(f"  candidate (variant=strong) accuracy samples: {candidate_acc}")
    print(f"    measures: {candidate_measures}\n")

    # Certified selection over the collected metrics (a real statistical test —
    # no fabricated winner). With a small deterministic sample the gate may
    # honestly return "no decision"; that is a correct, non-faked outcome.
    gate = PromotionGate(
        PromotionPolicy(alpha=0.05, min_effect={"accuracy": 0.0}),
        [ObjectiveSpec("accuracy", "maximize")],
    )
    decision = gate.evaluate(
        incumbent_metrics={"accuracy": incumbent_acc},
        candidate_metrics={"accuracy": candidate_acc},
    )

    if decision.decision == "promote":
        print(f"CERTIFIED WINNER: variant=strong promoted — {decision.reason}")
    elif decision.decision == "reject":
        print(f"NO PROMOTION: candidate rejected — {decision.reason}")
    else:
        print(f"NO WINNER YET (honest): {decision.reason}")


# --------------------------------------------------------------------------- #
# Hybrid / cloud wire (documented, NOT executed here)                         #
# --------------------------------------------------------------------------- #
#
# In a hybrid or cloud run, you do not call the promotion gate yourself — the
# orchestrator does. Your decorated function executes the composite in-trial and
# merges the composite telemetry into the metrics it returns; those numeric
# metrics ride the existing measures channel to the backend:
#
#     import traigent
#
#     @traigent.optimize(
#         eval_dataset=...,
#         objectives=["accuracy"],
#         configuration_space={"variant": ["cheap", "strong"]},
#         default_config={"variant": "cheap"},
#         execution_mode="hybrid",
#     )
#     def answer(text: str) -> tuple[str, dict[str, float]]:
#         cfg = traigent.get_config()
#         params = dict(cfg)
#         run = execute_composite(
#             COMPOSITE.structure,
#             {"cheap": _stage([...]), "strong": _stage([_EXPECTED])},
#             config=params,
#             calibrated_values={GATE: params[GATE]},
#         )
#         # The composite_* keys become per-trial measures on the wire.
#         metrics = {"accuracy": 1.0 if str(run.output) == _EXPECTED else 0.0}
#         metrics.update(composite_measures(run))
#         return str(run.output), metrics
#
# Requires a reachable backend and credentials (TRAIGENT_API_KEY,
# TRAIGENT_BACKEND_URL) plus TRAIGENT_OFFLINE_MODE=false. See
# docs/concepts/composite-knobs.md for the strict certified-selection recipe.


if __name__ == "__main__":
    main()
