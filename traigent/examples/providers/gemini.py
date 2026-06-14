"""Quickstart example — Google Gemini (AI Studio) via LiteLLM.

LiteLLM is a core Traigent dependency, so this runs with no extra
packages. (LangChain's ChatGoogleGenerativeAI is not mock-intercepted by
the SDK, so the keyless demo uses LiteLLM instead.)

    # mock (no key, no spend):
    python -m traigent.examples.providers.gemini
    # real:
    GEMINI_API_KEY=AIza... TRAIGENT_MOCK_LLM=false python -m traigent.examples.providers.gemini
"""

from __future__ import annotations

from traigent.examples.providers import get_provider
from traigent.examples.providers._bootstrap import ensure_packages
from traigent.examples.providers._demo import (
    DATASET,
    EvaluationOptions,
    build_config_space,
    configure_demo_env,
    demo_scorer,
    run_optimization,
)

_PROVIDER = get_provider("gemini")
ensure_packages([("litellm", None)])
configure_demo_env(_PROVIDER)

import litellm  # noqa: E402

import traigent  # noqa: E402


@traigent.optimize(
    configuration_space=build_config_space(_PROVIDER),
    objectives=["accuracy"],
    evaluation=EvaluationOptions(
        eval_dataset=DATASET, metric_functions={"accuracy": demo_scorer}
    ),
    execution_mode="edge_analytics",
)
def answer(question: str) -> str:
    cfg = traigent.get_config()
    # Call litellm.completion via the module attribute (not a `from litellm
    # import completion` binding) so Traigent's interceptor, which patches
    # litellm.completion at the module level, is the one that runs.
    response = litellm.completion(
        model=cfg["model"],
        messages=[{"role": "user", "content": question}],
        temperature=cfg["temperature"],
    )
    return str(response.choices[0].message.content)


def main() -> None:
    run_optimization(answer)


if __name__ == "__main__":
    main()
