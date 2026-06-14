"""Quickstart example — OpenRouter via LangChain.

OpenRouter is OpenAI-compatible, so it reuses langchain-openai's
ChatOpenAI with a custom base_url — no extra package beyond the OpenAI
one already in traigent[recommended].

    # mock (no key, no spend):
    python -m traigent.examples.providers.openrouter
    # real:
    OPENROUTER_API_KEY=sk-or-... TRAIGENT_MOCK_LLM=false python -m traigent.examples.providers.openrouter
"""

from __future__ import annotations

import os

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

_PROVIDER = get_provider("openrouter")
ensure_packages([("langchain_openai", "langchain-openai")])
configure_demo_env(_PROVIDER)

from langchain_openai import ChatOpenAI  # noqa: E402

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
    llm = ChatOpenAI(
        model=cfg["model"],
        temperature=cfg["temperature"],
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    return str(llm.invoke(question).content)


def main() -> None:
    run_optimization(answer)


if __name__ == "__main__":
    main()
