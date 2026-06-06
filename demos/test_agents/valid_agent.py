"""Valid agent configuration that passes all constraints."""

import traigent


@traigent.optimize(
    eval_dataset="evals.jsonl",
    objectives=["accuracy", "cost"],
    configuration_space={
        "model": ["gpt-4o-mini"],  # Allowed model
        "temperature": [0.1, 0.3],
    },
)
def support_agent(query: str, **config) -> str:
    """Customer support agent with valid configuration."""
    # Placeholder implementation
    model = config.get("model", "gpt-4o-mini")
    return f"Response using {model}"
