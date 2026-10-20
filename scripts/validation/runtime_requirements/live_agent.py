"""Agent used by the PY-1 live-dev acceptance script (kwargs form, key = 2000 ms)."""

import traigent


@traigent.requirement(
    latency_ms=2000,
    max_violation_rate=0.05,
    window="1h",
    min_samples=100,
    on_breach="fix",
    max_fix_cost_usd=1.0,
)
@traigent.optimize(
    eval_dataset="live_dataset.jsonl",
    objectives=["accuracy"],
    configuration_space={"temperature": [0.0, 0.5]},
    injection_mode="seamless",
)
def answer(question: str, temperature: float = 0.0) -> str:
    return question
