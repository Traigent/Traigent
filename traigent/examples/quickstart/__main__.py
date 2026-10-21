"""Keyless plumbing check for Traigent - one decorator, mock LLM, simulated scores.

This demo verifies the install and the optimize loop end to end. Its scores
come from a hardcoded table, so it does not compare models or pick a winner.

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
        f'Run: {sys.executable} -m pip install "litellm>=1.100.5,<2"',
        file=sys.stderr,
    )
    raise SystemExit(1) from None

import traigent  # noqa: E402
from traigent.api.decorators import EvaluationOptions  # noqa: E402
from traigent.examples.quickstart._messages import completion_message  # noqa: E402
from traigent.utils.callbacks import ProgressBarCallback  # noqa: E402

DATASET = [
    {"input": {"question": "What is the capital of France?"}, "output": "Paris"},
    {"input": {"question": "What is 2 + 2?"}, "output": "4"},
    {
        "input": {"question": "Who wrote Romeo and Juliet?"},
        "output": "William Shakespeare",
    },
]
CONFIG_SPACE = {
    "model": ["gpt-4.1-mini", "gpt-4.1"],
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
# model-correlated values so the results table has distinct rows. The scores are simulated and
# are never presented as a winner.
# A custom scoring function is the right escape hatch for mock-mode
# demos; users who switch to real LLMs replace it with their own
# accuracy metric (see walkthrough/real/01_tuning_qa.py).
_MODEL_DEMO_SCORES = (("gpt-4.1", 0.85), ("gpt-4.1-mini", 0.65))


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
    # show_best=False: the keyless demo is a plumbing check with no winner, so
    # the results table must not star a trial or print "Overall Best" (#1690).
    result = asyncio.run(
        answer.optimize(
            max_trials=6,
            callbacks=[ProgressBarCallback(show_best=False)],
        )
    )

    # Fail closed on an all-failed run. If every trial errors (or the run is
    # otherwise misconfigured), optimize() returns NORMALLY with best_score=None
    # / best_config=None — it does not raise — so without this guard the quickstart
    # would exit 0 on a run that produced no winner, i.e. a silent all-failed run
    # that looks like success (CLAUDE.md Rule 2: no fake completion). The import
    # guard above only covers the missing-litellm case.
    if result.best_score is None or not result.best_config:
        print(
            "[traigent] Quickstart produced no successful trials — no best "
            "configuration was found (every trial failed). Re-run with "
            "TRAIGENT_LOG_LEVEL=DEBUG to see the per-trial errors.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # Keyless demo scores are simulated: report completion, never a winner.
    print(completion_message(len(getattr(result, "trials", None) or [])))


if __name__ == "__main__":
    main()
