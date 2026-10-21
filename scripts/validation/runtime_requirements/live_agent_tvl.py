"""Agent used by step 2 of the PY-1 live-dev script (TVL fragment form, key = 2500 ms)."""

import traigent


@traigent.requirement(tvl="live_latency_2500.tvl.yml", max_fix_cost_usd=1.0)
@traigent.optimize(
    eval_dataset="live_dataset.jsonl",
    objectives=["accuracy"],
    configuration_space={"temperature": [0.0, 0.5]},
    injection_mode="seamless",
)
def answer(question: str, temperature: float = 0.0) -> str:
    return question
