"""Shared scaffolding for the per-provider quickstart examples.

Keeps each provider script down to the part that actually differs — the
import and the LLM call — while this module owns the boring, identical
bits: mock-mode setup, the bundled dataset, a deterministic demo scorer,
the config space, and the optimize runner.

Design notes:

* **Mock by default.** ``TRAIGENT_MOCK_LLM`` defaults to on, so the
  examples run with no API keys and no provider spend (LLM calls are
  intercepted). Set ``TRAIGENT_MOCK_LLM=false`` for a real run.
* **Mock scorer.** In mock mode the interceptor returns a generic
  placeholder string, so an output-based metric would score every trial
  identically and hide the optimization signal. :func:`demo_scorer`
  ignores the (mock) output and returns a deterministic,
  model-correlated score so the results table ranks. Real runs should
  replace it with a real accuracy metric.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
from typing import Any

from traigent.api.decorators import EvaluationOptions

_PACKAGE_DIR = Path(__file__).resolve().parent
DATASET = str(_PACKAGE_DIR / "qa_samples.jsonl")

# So the SDK's dataset-path validation accepts the bundled file regardless
# of the user's working directory (mirrors the bundled quickstart).
os.environ.setdefault("TRAIGENT_DATASET_ROOT", str(_PACKAGE_DIR))

SYSTEM_PROMPT = (
    "Answer in as few words as possible. Give only the answer itself, nothing else."
)


def _stable_model_score(model: object) -> float:
    """Deterministic pseudo-score in [0.55, 0.95] keyed on the model name.

    Stable across runs and distinct per model, so the mock results table
    ranks trials meaningfully without depending on the (mocked) output.
    """
    digest = hashlib.sha256(str(model).encode("utf-8")).hexdigest()
    return 0.55 + (int(digest, 16) % 41) / 100.0


def demo_scorer(
    output: str,
    expected: str,
    config: dict[str, object] | None = None,
    **_kwargs: object,
) -> float:
    """Model-correlated, deterministic demo scorer (see module docstring)."""
    cfg = config
    if not cfg:
        try:
            from traigent.api.functions import get_config

            cfg = get_config() or {}
        except Exception:
            cfg = {}
    base = _stable_model_score(cfg.get("model"))
    temperature_raw = cfg.get("temperature", 0.5)
    temperature = (
        float(temperature_raw)
        if isinstance(temperature_raw, (int, float, str))
        else 0.5
    )
    return max(0.0, base - 0.05 * temperature)


def build_config_space(provider: dict[str, Any]) -> dict[str, Any]:
    """Config space derived from the manifest entry (models + temperatures)."""
    return {
        "model": list(provider["models"]),
        "temperature": [0.0, 0.7],
    }


def configure_demo_env(provider: dict[str, Any]) -> bool:
    """Set up mock-or-real mode for one provider; return ``True`` if mock.

    In mock mode we seed placeholder values for the provider's required
    env vars (so LangChain clients can be *constructed*; the call itself
    is intercepted) and force offline mode when no portal key is set.
    LiteLLM providers don't need the placeholders, but seeding them is
    harmless.
    """
    is_mock = os.environ.get("TRAIGENT_MOCK_LLM", "true").strip().lower() != "false"
    if not is_mock:
        return False

    for var in provider.get("env_vars", []):
        name = var["name"]
        if name == "TRAIGENT_MOCK_LLM":
            continue
        os.environ.setdefault(name, "mock-key-for-demos")  # pragma: allowlist secret

    if not os.environ.get("TRAIGENT_API_KEY"):
        os.environ["TRAIGENT_OFFLINE_MODE"] = "true"

    from traigent.testing import enable_mock_mode_for_quickstart

    enable_mock_mode_for_quickstart()
    return is_mock


def _trial_count(value: Any) -> int | None:
    """Best-effort count of a trials attribute that may be an int or a sized collection."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return len(value)
    except TypeError:
        return None


def run_optimization(
    answer_fn: Any, *, max_trials: int = 6, algorithm: str = "grid"
) -> Any:
    """Run the optimization synchronously and fail loudly if nothing succeeded.

    The optimizer aggregates per-trial errors rather than raising, so a run
    where every trial failed (e.g. a real run with a missing/invalid API key)
    would otherwise return a result and exit 0. For a demo that is misleading,
    so we surface it as a non-zero exit with an actionable message.
    """
    result = asyncio.run(answer_fn.optimize(max_trials=max_trials, algorithm=algorithm))

    successful = _trial_count(getattr(result, "successful_trials", None))
    total = _trial_count(getattr(result, "trials", None))
    success_rate = getattr(result, "success_rate", None)
    no_success = successful == 0 or success_rate == 0 or success_rate == 0.0
    had_trials = total is None or total > 0

    if no_success and had_trials:
        raise SystemExit(
            "All trials failed — no examples succeeded. In a real run this almost "
            "always means a missing/invalid provider key or an unavailable model. "
            "Set the provider credentials and a valid model, or drop "
            "TRAIGENT_MOCK_LLM=false to run the keyless mock demo."
        )
    return result


__all__ = [
    "DATASET",
    "EvaluationOptions",
    "SYSTEM_PROMPT",
    "build_config_space",
    "configure_demo_env",
    "demo_scorer",
    "run_optimization",
]
