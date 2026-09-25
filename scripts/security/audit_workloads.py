"""Workloads exercised by ``behaviour_audit.py``.

Each workload runs inside a fresh interpreter that already has the audit hook
installed. None of them needs a real LLM key or a real backend: LLM calls go
through the SDK's mock mode and the backend URL points at the recorder's
127.0.0.1 stub (see ``behaviour_audit.child_env``).

Keep SDK imports INSIDE the workload functions: import-time behaviour is part
of what is being recorded, and ``import_litellm_first`` depends on ordering.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import runpy
import sys
from collections.abc import Callable
from typing import Any

_litellm: Any = None  # bound lazily; see _seamless_answer

_DATASET = [
    {"input": {"question": "What is the capital of France?"}, "output": "Paris"},
    {"input": {"question": "What is 2 + 2?"}, "output": "4"},
]


def _score(output: str, expected: str, **_kwargs: object) -> float:
    return 1.0 if str(expected).lower() in str(output).lower() else 0.5


def quickstart() -> None:
    """Keyless packaged quickstart (``python -m traigent.examples.quickstart``)."""
    spec = importlib.util.find_spec("traigent")
    assert spec is not None and spec.origin is not None
    main_py = os.path.join(
        os.path.dirname(spec.origin), "examples", "quickstart", "__main__.py"
    )
    # The SDK's quickstart bootstrap keys off sys.argv[0] (traigent/__init__.py
    # _is_quickstart_invocation); set it exactly as ``python -m`` would, and
    # drop the dummy key so this is the keyless path a first-time user runs.
    sys.argv = [main_py]
    os.environ.pop("TRAIGENT_API_KEY", None)
    runpy.run_module("traigent.examples.quickstart", run_name="__main__")


def _run_optimize(*, injection_mode: str = "context") -> None:
    # traigent BEFORE litellm, on purpose: this is the SDK-default case, where
    # `import traigent` pins litellm's bundled price table before litellm's
    # own import-time fetch. The litellm-first order (what an import sorter
    # produces) is measured separately by import_litellm_first.
    import traigent  # isort: skip
    from traigent.api.decorators import EvaluationOptions  # isort: skip
    from traigent.testing import enable_mock_mode_for_quickstart  # isort: skip

    import litellm  # isort: skip

    enable_mock_mode_for_quickstart()
    globals()["_litellm"] = litellm

    if injection_mode == "seamless":
        answer = traigent.optimize(
            configuration_space={"model": ["gpt-4o-mini", "gpt-4o"]},
            objectives=["accuracy"],
            injection_mode="seamless",
            evaluation=EvaluationOptions(
                eval_dataset=_DATASET, metric_functions={"accuracy": _score}
            ),
        )(_seamless_answer)

    else:

        @traigent.optimize(
            configuration_space={
                "model": ["gpt-4o-mini", "gpt-4o"],
                "temperature": [0.0, 0.7],
            },
            objectives=["accuracy"],
            evaluation=EvaluationOptions(
                eval_dataset=_DATASET, metric_functions={"accuracy": _score}
            ),
        )
        def answer(question: str) -> str:
            cfg = traigent.get_config()
            response = litellm.completion(
                model=str(cfg["model"]),
                messages=[{"role": "user", "content": question}],
                temperature=float(cfg["temperature"]),
            )
            return str(response.choices[0].message.content)

    result = asyncio.run(answer.optimize(max_trials=2))
    if result.best_config is None:
        raise RuntimeError("optimize() produced no successful trial")


def _seamless_answer(question: str) -> str:
    # Module-level on purpose: seamless mode re-parses this source with
    # inspect.getsource and recompiles it against the module globals, so it
    # must not close over locals, and its AST validator rejects `import`
    # statements (traigent/config/ast_transformer.py DANGEROUS_NODES). The
    # module global ``_litellm`` is bound by _run_optimize.
    model = "gpt-4o-mini"
    response = _litellm.completion(
        model=model, messages=[{"role": "user", "content": question}]
    )
    return str(response.choices[0].message.content)


def optimize_mock() -> None:
    """``@traigent.optimize`` in mock-LLM mode, dummy API key, stub backend."""
    _run_optimize()


def optimize_seamless() -> None:
    """Same run with ``injection_mode="seamless"`` (AST rewrite + compile)."""
    _run_optimize(injection_mode="seamless")


def import_litellm_first() -> None:
    """``import litellm`` BEFORE ``import traigent``, then the mock run."""
    import litellm  # noqa: F401

    import traigent  # noqa: F401

    _run_optimize()


def token_count_llama() -> None:
    """SDK cost helper pricing a Llama prompt (litellm HF-tokenizer path)."""
    from traigent.utils.cost_calculator import calculate_prompt_cost

    try:
        calculate_prompt_cost("hello world, how are you?", "meta-llama/Llama-3-8b")
    except Exception as exc:  # noqa: BLE001 - unpriced model is fine; egress is the point
        print(f"token_count_llama: {type(exc).__name__}: {exc}", file=sys.stderr)


def canary_new_host() -> None:
    """Control: one stdlib request to an unlisted host (recorder sensitivity)."""
    import urllib.request

    import traigent  # noqa: F401 - same import-time baseline as the real runs

    try:
        urllib.request.urlopen("https://egress-canary.invalid/", timeout=5)  # noqa: S310
    except OSError as exc:
        print(f"canary_new_host: {type(exc).__name__}: {exc}", file=sys.stderr)


WORKLOADS: dict[str, Callable[[], None]] = {
    "quickstart": quickstart,
    "optimize_mock": optimize_mock,
    "optimize_seamless": optimize_seamless,
    "import_litellm_first": import_litellm_first,
    "token_count_llama": token_count_llama,
    "canary_new_host": canary_new_host,
}
