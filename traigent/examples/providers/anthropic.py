"""Quickstart example — Anthropic Claude via LangChain.

Optimizes a tiny Q&A agent across models/temperatures.

    # mock (no key, no spend):
    python -m traigent.examples.providers.anthropic
    # real:
    ANTHROPIC_API_KEY=sk-ant-... TRAIGENT_MOCK_LLM=false python -m traigent.examples.providers.anthropic
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

_PROVIDER = get_provider("anthropic")
ensure_packages([("langchain_anthropic", "langchain-anthropic")])
configure_demo_env(_PROVIDER)

from langchain_anthropic import ChatAnthropic  # noqa: E402

import traigent  # noqa: E402


@traigent.optimize(
    configuration_space=build_config_space(_PROVIDER),
    objectives=["accuracy"],
    evaluation=EvaluationOptions(
        eval_dataset=DATASET, metric_functions={"accuracy": demo_scorer}
    ),
    offline=True,
)
def answer(question: str) -> str:
    cfg = traigent.get_config()
    llm = ChatAnthropic(model=cfg["model"], temperature=cfg["temperature"])
    return str(llm.invoke(question).content)


def main() -> None:
    run_optimization(answer)


if __name__ == "__main__":
    main()
