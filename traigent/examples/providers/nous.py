"""Quickstart example — Nous Portal (Hermes) via LangChain.

Nous Portal is OpenAI-compatible, so it reuses langchain-openai's ChatOpenAI
with a custom base_url — no extra package beyond the OpenAI one already in
traigent[recommended].

Unlike the other providers, Nous authenticates with a **short-lived JWT minted
from a refresh token** (OAuth), not a static API key. ``get_nous_api_key()``
(see traigent.integrations.llms.nous_auth) resolves a current token: set
NOUS_API_KEY to a pre-minted JWT (no auto-refresh), or set NOUS_REFRESH_TOKEN /
log in with the Hermes CLI (~/.hermes/auth.json) for auto-refresh.

    # mock (no key, no spend):
    python -m traigent.examples.providers.nous
    # real:
    NOUS_API_KEY=... TRAIGENT_MOCK_LLM=false python -m traigent.examples.providers.nous
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
from traigent.integrations.llms.nous_auth import NOUS_BASE_URL, get_nous_api_key

_PROVIDER = get_provider("nous")
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
    offline=True,
)
def answer(question: str) -> str:
    cfg = traigent.get_config()
    llm = ChatOpenAI(
        model=cfg["model"],
        temperature=cfg["temperature"],
        base_url=NOUS_BASE_URL,
        # Resolved per trial so a long run stays ahead of JWT expiry. In the
        # keyless mock demo this returns the seeded NOUS_API_KEY with no network.
        api_key=get_nous_api_key(),
    )
    return str(llm.invoke(question).content)


def main() -> None:
    run_optimization(answer)


if __name__ == "__main__":
    main()
