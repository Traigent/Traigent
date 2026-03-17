#!/usr/bin/env python3
"""Scene 3: Traigent optimization — replay of real experiment results.

All values (accuracy, cost, latency) are from a real OpenAI API run.
This script replays those results using the canonical print_results_table.

Usage (from the Traigent repo root):
    python walkthrough/demo/scene3_optimize.py
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import print_optimization_config, print_results_table

from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema

OBJECTIVES = ObjectiveSchema.from_objectives(
    [
        ObjectiveDefinition("accuracy", orientation="maximize", weight=0.5),
        ObjectiveDefinition("cost", orientation="minimize", weight=0.2),
        ObjectiveDefinition("latency", orientation="minimize", weight=0.3),
    ]
)

CONFIG_SPACE = {
    "model": [
        "gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o",
        "gpt-5.2", "gpt-5-nano", "gpt-5.1",
    ],
    "prompt": ["minimal", "role_based"],
    "temperature": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "instructions": ["CoT", "direct"],
    "max_tokens": [50, 100, 200],
}

# Real results from walkthrough/real/09_rag_multi_objective.py
# (algorithm=random, seed=42, 16 completed trials)
TRIALS_DATA = [
    # (model, prompt, temp, instructions, max_tokens, accuracy, cost, latency)
    ("gpt-5.1",       "minimal",   0.0, "direct", 50,  0.180, 0.00055, 2.747),
    ("gpt-4o-mini",   "minimal",   0.1, "CoT",    200, 0.771, 0.00010, 3.728),
    ("gpt-5.2",       "minimal",   0.0, "CoT",    50,  0.128, 0.00079, 1.665),
    ("gpt-4o-mini",   "minimal",   0.8, "CoT",    200, 0.745, 0.00010, 3.783),
    ("gpt-5.1",       "role_based",0.3, "direct", 200, 0.811, 0.00062, 1.573),
    ("gpt-4o",        "minimal",   0.2, "direct", 100, 0.741, 0.00094, 2.070),
    ("gpt-4o",        "minimal",   0.3, "direct", 50,  0.710, 0.00062, 1.460),
    ("gpt-3.5-turbo", "role_based",0.1, "direct", 100, 0.874, 0.00008, 1.025),
    ("gpt-5-nano",    "role_based",0.0, "direct", 200, 0.050, 0.00008, 2.568),
    ("gpt-3.5-turbo", "role_based",0.1, "direct", 200, 0.868, 0.00008, 0.996),
    ("gpt-5-nano",    "role_based",0.9, "CoT",    200, 0.000, 0.00008, 2.457),
    ("gpt-3.5-turbo", "minimal",   0.3, "direct", 50,  0.776, 0.00010, 1.198),
    ("gpt-4o-mini",   "minimal",   0.6, "direct", 100, 0.738, 0.00006, 1.840),
    ("gpt-5.1",       "role_based",0.2, "direct", 100, 0.711, 0.00061, 1.604),
    ("gpt-4o-mini",   "role_based",0.1, "CoT",    200, 0.864, 0.00003, 1.163),
    ("gpt-5.1",       "minimal",   0.2, "direct", 100, 0.335, 0.00082, 2.234),
]

BEST_IDX = 7  # trial 8: gpt-3.5-turbo, 87.4%


# ── Lightweight stand-ins that satisfy print_results_table ──

@dataclass
class FakeTrial:
    config: dict[str, Any] = field(default_factory=dict)
    configuration: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    duration: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_successful(self) -> bool:
        return True


@dataclass
class FakeResult:
    best_config: dict[str, Any] = field(default_factory=dict)
    best_score: float = 0.0
    best_metrics: dict[str, float] = field(default_factory=dict)
    trials: list[FakeTrial] = field(default_factory=list)
    duration: float = 0.0

    def calculate_weighted_scores(self, **_) -> dict:
        return {"best_weighted_config": self.best_config}


def build_results() -> FakeResult:
    trials = []
    for model, prompt, temp, instr, mt, acc, cost, lat in TRIALS_DATA:
        cfg = {
            "model": model, "prompt": prompt, "temperature": temp,
            "instructions": instr, "max_tokens": mt,
        }
        trials.append(FakeTrial(
            config=cfg,
            configuration=cfg,
            metrics={"accuracy": acc, "cost": cost, "latency": lat},
            score=acc,
            duration=lat,
        ))

    best = TRIALS_DATA[BEST_IDX]
    best_cfg = {
        "model": best[0], "prompt": best[1], "temperature": best[2],
        "instructions": best[3], "max_tokens": best[4],
    }
    return FakeResult(
        best_config=best_cfg,
        best_score=best[5],
        best_metrics={"accuracy": best[5], "cost": best[6], "latency": best[7]},
        trials=trials,
        duration=sum(t[7] for t in TRIALS_DATA),
    )


def progress_animation():
    total = len(TRIALS_DATA)
    for i in range(1, total + 1):
        filled = int(30 * i / total)
        bar = "█" * filled + "░" * (30 - filled)
        best_so_far = max(t[5] for t in TRIALS_DATA[:i])
        print(f"\rOptimizing: [{bar}] {i}/{total}  Best accuracy: {best_so_far:.1%}  ", end="", flush=True)
        time.sleep(0.4)
    print("\n")


def main():
    print("Traigent — RAG Multi-Objective Optimization")
    print("=" * 55)
    print("Balancing accuracy (50%), cost (20%), latency (30%).\n")
    print_optimization_config(OBJECTIVES, CONFIG_SPACE)

    progress_animation()

    results = build_results()
    print_results_table(results, CONFIG_SPACE, OBJECTIVES, is_mock=False, task_type="rag_qa")

    print("\nBest Configuration Found:")
    print(f"  Model:        {results.best_config['model']}")
    print(f"  Prompt:       {results.best_config['prompt']}")
    print(f"  Temperature:  {results.best_config['temperature']}")
    print(f"  Instructions: {results.best_config['instructions']}")
    print(f"  Max Tokens:   {results.best_config['max_tokens']}")
    print(f"\nPerformance:")
    print(f"  Accuracy: {results.best_metrics['accuracy']:.1%}")
    print(f"  Cost:     ${results.best_metrics['cost']:.5f}")
    print(f"  Latency:  {results.best_metrics['latency']:.3f}s")
    print("\nNote: Results recorded from real OpenAI API calls. Replayed here to save time.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(130)
