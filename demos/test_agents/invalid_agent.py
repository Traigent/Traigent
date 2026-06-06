"""Invalid agent configuration that violates constraints."""

import traigent


@traigent.optimize(
    eval_dataset="evals.jsonl",
    objectives=["accuracy"],
    configuration_space={
        "model": ["gpt-4-32k"],  # BLOCKED model - too expensive!
        "temperature": [0.9],
    },
)
def expensive_agent(query: str, **config) -> str:
    """Agent using an expensive blocked model."""
    model = config.get("model", "gpt-4-32k")
    return f"Response using {model}"
