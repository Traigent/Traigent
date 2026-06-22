"""Find the best model and temperature for your task - in one decorator.

No API keys, no LLM provider calls, no spend. The bundled quickstart
is the load-bearing demo on the website funnel; everything below is
structured to make that promise true. LiteLLM may still attempt an
import-time model-cost-map fetch from raw.githubusercontent.com and fall back
to bundled pricing data when the network is unavailable; the guarantee is no
provider spend, not zero outbound packets.

The hermetic env-var setup that makes "no provider calls" work has to
run before ``traigent/__init__.py`` itself executes (because that module
pulls in optional deps like LiteLLM that may try to fetch model cost
maps at import time). It can't live here — by the time this file
executes, the parent package has already loaded. Instead, the bootstrap
lives at the very top of :mod:`traigent.__init__`, gated on a
``sys.argv`` check that detects quickstart invocations
(``traigent quickstart`` and ``python -m traigent.examples.quickstart``).
This file is the *demo*; the SDK package itself owns the bootstrap.

Set ``TRAIGENT_API_KEY`` to sync results to the portal even while LLM
calls stay mocked. For a real run with actual LLM calls, see
``walkthrough/real/01_tuning_qa.py``.
"""

import asyncio
import os
import sys

# Flip the in-code mock-mode flag too. The bootstrap in traigent/__init__.py
# already set TRAIGENT_MOCK_LLM=true so the env-var path is active, but
# calling the API here makes mock mode visible in code review and proves
# the recommended user-facing API works (it would also activate mock if
# the bootstrap somehow didn't fire).
from traigent.testing import enable_mock_mode_for_quickstart

enable_mock_mode_for_quickstart()

from traigent.examples.quickstart._env import configure_quickstart_env

configure_quickstart_env(os.environ)

try:
    import litellm  # noqa: E402
except ModuleNotFoundError as exc:
    missing = exc.name or "litellm"
    print(
        f"[traigent] Missing required quickstart dependency '{missing}'. "
        f'Run: {sys.executable} -m pip install "litellm==1.87.1"',
        file=sys.stderr,
    )
    raise SystemExit(1) from None

import traigent  # noqa: E402
from traigent.api.decorators import EvaluationOptions  # noqa: E402

DATASET = [
    {"input": {"question": "What is the capital of France?"}, "output": "Paris"},
    {"input": {"question": "What is 2 + 2?"}, "output": "4"},
    {
        "input": {"question": "Who wrote Romeo and Juliet?"},
        "output": "William Shakespeare",
    },
]
CONFIG_SPACE = {
    "model": ["gpt-4o-mini", "gpt-4o"],
    "temperature": [0.0, 0.7, 1.0],
}

_SYSTEM_PROMPT = (
    "Answer in as few words as possible. Give only the answer itself, nothing else."
)

# Model-correlated scores so the demo's results table actually ranks
# trials. The mock interceptor returns a generic placeholder string —
# a real ``contains``-style metric on that output would always score 0,
# making every trial look identical and hiding the optimization signal.
# This scorer is independent of the LLM output: it produces non-zero,
# model-correlated values so users see a meaningful "best config" hit.
# A custom scoring function is the right escape hatch for mock-mode
# demos; users who switch to real LLMs replace it with their own
# accuracy metric (see walkthrough/real/01_tuning_qa.py).
_MODEL_DEMO_SCORES = (("gpt-4o", 0.85), ("gpt-4o-mini", 0.65))


def _model_demo_score(model: object) -> float:
    for model_name, score in _MODEL_DEMO_SCORES:
        if str(model) == model_name:
            return score
    return 0.5


def _demo_scorer(
    output: str,
    expected: str,
    config: dict[str, object] | None = None,
    **_kwargs: object,
) -> float:
    """Mock-mode demo scorer — produces model-correlated, deterministic scores.

    Ignores ``output`` (which is a generic mock string in mock mode) and
    returns a score derived from the trial's ``model`` and
    ``temperature``. Lower temperature gets a tiny boost so trials
    within the same model rank distinctly.

    If the SDK does not forward the trial config as a ``config=`` kwarg
    to metric functions (Greptile review #2 — silent zero-score risk),
    fall back to :func:`traigent.get_config` so the demo doesn't
    silently degrade into "every trial scores the same".
    """
    cfg = config
    if not cfg:
        # Fall back to the active trial's config from the SDK context.
        # This may itself be empty if called outside an optimization
        # (e.g. during validation) — in that case we return a neutral
        # mid-range score rather than silently asserting a winner.
        try:
            from traigent.api.functions import get_config

            cfg = get_config() or {}
        except Exception:
            cfg = {}
    base = _model_demo_score(cfg.get("model"))
    temperature_raw = cfg.get("temperature", 0.5)
    temperature = (
        float(temperature_raw)
        if isinstance(temperature_raw, (int, float, str))
        else 0.5
    )
    return max(0.0, base - 0.05 * temperature)


@traigent.optimize(
    configuration_space=CONFIG_SPACE,
    objectives=["accuracy"],
    evaluation=EvaluationOptions(
        eval_dataset=DATASET, metric_functions={"accuracy": _demo_scorer}
    ),
)
def answer(question: str) -> str:
    """Call an LLM with the current trial's config (intercepted in mock mode)."""
    cfg = traigent.get_config()

    # Call through the litellm module attribute so Traigent's mock
    # interceptor, which patches litellm.completion, is the call path.
    response = litellm.completion(
        model=str(cfg["model"]),
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=float(cfg["temperature"]),
    )
    return str(response.choices[0].message.content)


def main() -> None:
    """Synchronous entry point used by the CLI subcommand and ``python -m``."""
    asyncio.run(answer.optimize(max_trials=6))


if __name__ == "__main__":
    main()
