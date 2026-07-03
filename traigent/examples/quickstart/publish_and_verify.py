"""Run a mock LLM optimization that can publish to the Traigent portal.

This example is the positive companion to ``python -m traigent.examples.quickstart``:

* With no ``TRAIGENT_API_KEY``, it stays fully offline and prints local results.
* With ``TRAIGENT_API_KEY`` set to a key that carries ``experiments:write``, it
  keeps LLM calls mocked but creates a backend session and submits per-trial
  results to the portal.

Run:

    TRAIGENT_MOCK_LLM=true python -m traigent.examples.quickstart.publish_and_verify

To publish the run:

    TRAIGENT_API_KEY=<key-with-experiments-write> TRAIGENT_MOCK_LLM=true \
      python -m traigent.examples.quickstart.publish_and_verify

After a key-backed run completes, open {portal_url} and look for the
experiment named "Quickstart Publish and Verify". If the backend returned a
direct URL, the script prints it.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from traigent.examples._portal import PORTAL_URL

if __doc__ is not None:
    __doc__ = __doc__.format(portal_url=PORTAL_URL)

EXPERIMENT_NAME = "Quickstart Publish and Verify"
CONFIG_SPACE = {
    "model": ["gpt-4o-mini", "gpt-4o"],
    "temperature": [0.0, 0.7],
}
DATASET = [
    {"input": {"question": "What is the capital of France?"}, "output": "Paris"},
    {"input": {"question": "What is 2 + 2?"}, "output": "4"},
]
SYSTEM_PROMPT = (
    "Answer in as few words as possible. Give only the answer itself, nothing else."
)


def _demo_scorer(
    output: str,
    expected: str,
    config: dict[str, object] | None = None,
    **_kwargs: object,
) -> float:
    """Deterministic mock scorer so the example ranks trials without real LLMs."""
    cfg = config
    if not cfg:
        try:
            from traigent.api.functions import get_config

            cfg = get_config() or {}
        except Exception:
            cfg = {}

    model_score = 0.85 if str(cfg.get("model")) == "gpt-4o" else 0.65
    temperature_raw = cfg.get("temperature", 0.5)
    temperature = (
        float(temperature_raw)
        if isinstance(temperature_raw, (int, float, str))
        else 0.5
    )
    return max(0.0, model_score - 0.05 * temperature)


def build_answer(litellm_module: Any) -> Any:
    """Build the decorated answer function after mock/offline env is configured."""
    import traigent
    from traigent.api.decorators import EvaluationOptions

    # algorithm="grid" is the current local-search selector. The runtime mode for
    # that path is local, and backend tracking remains enabled unless the
    # no-key quickstart bootstrap forced offline mode. Avoid passing the deprecated
    # execution_mode="edge_analytics" decorator option here: current policy maps
    # that legacy selector to offline=True for backward-compatible no-egress runs.
    @traigent.optimize(
        experiment_name=EXPERIMENT_NAME,
        configuration_space=CONFIG_SPACE,
        objectives=["accuracy"],
        evaluation=EvaluationOptions(
            eval_dataset=DATASET,
            metric_functions={"accuracy": _demo_scorer},
        ),
        algorithm="grid",
    )
    def answer(question: str) -> str:
        cfg = traigent.get_config()
        response = litellm_module.completion(
            model=str(cfg["model"]),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=float(cfg["temperature"]),
        )
        return str(response.choices[0].message.content)

    return answer


def _print_portal_verification(result: Any) -> None:
    """Print where to verify a portal-tracked run after optimize() completes."""
    cloud_url = getattr(result, "cloud_url", None)
    experiment_id = getattr(result, "experiment_id", None)
    run_id = getattr(result, "experiment_run_id", None)

    if os.environ.get("TRAIGENT_API_KEY"):
        print(
            "[traigent] Portal publish path was enabled: local "
            "search with backend session + per-trial submission."
        )
        if cloud_url:
            print(f"[traigent] Open this portal run: {cloud_url}")
        else:
            print(
                f"[traigent] Open {PORTAL_URL} and look for "
                f'experiment "{EXPERIMENT_NAME}".'
            )
        if experiment_id:
            suffix = f", run {run_id}" if run_id else ""
            print(f"[traigent] Backend experiment {experiment_id}{suffix}.")
        return

    print(
        "[traigent] No TRAIGENT_API_KEY was set, so this mock run stayed "
        "offline and will not appear on the portal."
    )
    print(
        "[traigent] To publish: set TRAIGENT_API_KEY to a key with "
        "experiments:write and rerun this module with TRAIGENT_MOCK_LLM=true."
    )


def main() -> None:
    """Run the publish-and-verify quickstart."""
    from traigent.examples.quickstart._env import configure_quickstart_env
    from traigent.testing import enable_mock_mode_for_quickstart

    os.environ.setdefault("TRAIGENT_MOCK_LLM", "true")
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    enable_mock_mode_for_quickstart()
    configure_quickstart_env(os.environ)

    try:
        import litellm
    except ModuleNotFoundError as exc:
        missing = exc.name or "litellm"
        print(
            f"[traigent] Missing required quickstart dependency '{missing}'. "
            f'Run: {sys.executable} -m pip install "litellm>=1.87.1,<2"',
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    answer = build_answer(litellm)
    result = asyncio.run(answer.optimize(max_trials=4))

    if result.best_score is None or not result.best_config:
        print(
            "[traigent] Publish-and-verify example produced no successful trials.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(
        f"[traigent] Publish-and-verify complete - best config {result.best_config} "
        f"scored {result.best_score:.3f}."
    )
    _print_portal_verification(result)


if __name__ == "__main__":
    main()
