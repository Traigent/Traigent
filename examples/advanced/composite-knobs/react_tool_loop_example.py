#!/usr/bin/env python3
"""Composite knobs: run a ReAct-style tool loop offline and merge telemetry.

This example is deterministic and makes NO LLM calls and NO network calls. It
declares the ``react_tool_loop`` catalog pattern, executes it with a stubbed
tool-step loop body, evaluates a confidence signal over the threaded state, and
merges the composite telemetry into an ordinary metrics dict.

Run it::

    uv run python examples/advanced/composite-knobs/react_tool_loop_example.py
"""

from __future__ import annotations

import os

# Offline by default: this example never needs a backend.
os.environ.setdefault("TRAIGENT_OFFLINE_MODE", "true")
os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")

from traigent.knobs.patterns import react_tool_loop
from traigent.knobs.runtime import (
    LoopBodyResult,
    LoopBodyRunner,
    ResultKind,
    execute_composite,
)
from traigent.knobs.telemetry import merge_composite_measures

CONFIDENCE_MIN = "tool_confidence_min"

COMPOSITE = react_tool_loop(
    "qa_react",
    stage="react_tool_step",
    signal="tool_confidence",
    tool_confidence_min=CONFIDENCE_MIN,
    max_tool_calls=4,
)


def _tool_step(confidences: list[float]) -> LoopBodyRunner:
    """A deterministic one-tool-step body over a fixed confidence sequence."""
    counter = {"i": 0}

    def run(_item, state):
        i = counter["i"]
        counter["i"] += 1
        confidence = confidences[min(i, len(confidences) - 1)]
        tool_calls = [*state.get("tool_calls", ()), f"lookup-{i + 1}"]
        observations = [*state.get("observations", ()), f"observation-{i + 1}"]
        return LoopBodyResult(
            output=f"offline answer after {i + 1} step(s)",
            state={
                "scratchpad": f"thought-{i + 1}",
                "tool_calls": tool_calls,
                "observations": observations,
                "confidence": confidence,
                "last_error": None,
            },
        )

    return LoopBodyRunner(run=run)


def _confidence_signal(state) -> float:
    return float(state["confidence"])


def main() -> None:
    run = execute_composite(
        COMPOSITE.structure,
        {"react_tool_step": _tool_step([0.35, 0.66, 0.92])},
        config={"question": "offline demo"},
        calibrated_values={CONFIDENCE_MIN: 0.9},
        signals={"tool_confidence": _confidence_signal},
    )

    metrics: dict[str, float | int] = {
        "accepted": 1 if run.result_kind is ResultKind.OUTPUT else 0
    }
    merge_composite_measures(metrics, run)

    print("react_tool_loop example (offline, deterministic)\n")
    print(f"  result: {run.result_kind.value}")
    print(f"  output: {run.output}")
    print(f"  raw composite telemetry: {run.measures}")
    print(f"  merged metrics: {metrics}")


if __name__ == "__main__":
    main()
