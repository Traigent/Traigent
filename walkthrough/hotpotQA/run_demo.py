#!/usr/bin/env python3
"""Minimal HotpotQA optimization walkthrough using Traigent."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Mapping

# Optuna-backed optimizers are enabled by default.
os.environ.setdefault("TRAIGENT_MOCK_LLM", "true")
ROOT_DIR = Path(__file__).resolve().parents[2]
os.environ.setdefault("TRAIGENT_DATASET_ROOT", str(ROOT_DIR))

try:
    import traigent
except ImportError:  # pragma: no cover - support direct script execution
    import importlib
    import sys

    if str(ROOT_DIR) not in sys.path:
        sys.path.append(str(ROOT_DIR))
    traigent = importlib.import_module("traigent")
from paper_experiments.case_study_rag.dataset import (
    dataset_path,
    load_case_study_dataset,
)
from paper_experiments.case_study_rag.metrics import build_hotpot_metric_functions
from paper_experiments.case_study_rag.simulator import generate_case_study_answer

os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")


DATASET = str(dataset_path())
USE_MOCK = os.getenv("TRAIGENT_MOCK_LLM", "true").lower() == "true"


def _validate_real_credentials(model: str) -> None:
    lowered = model.lower()
    if "gpt" in lowered and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "Set OPENAI_API_KEY before running in real mode with GPT models."
        )
    if any(
        keyword in lowered for keyword in ("claude", "haiku", "sonnet")
    ) and not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "Set ANTHROPIC_API_KEY before running in real mode with Claude/Haiku models."
        )
    if "gpt" not in lowered and not any(
        keyword in lowered for keyword in ("claude", "haiku", "sonnet")
    ):
        raise ValueError(
            f"Unknown model '{model}'. Supported: gpt-*, claude-*, haiku-*."
        )


def _invoke_openai(model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Install the 'openai' package to run in real mode with GPT models."
        ) from exc

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def _invoke_anthropic(
    model: str, prompt: str, temperature: float, max_tokens: int
) -> str:
    try:
        from anthropic import Anthropic
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Install the 'anthropic' package to run in real mode with Claude/Haiku models."
        ) from exc

    client = Anthropic()
    lowered = model.lower()
    resolved_model = (
        "claude-3-5-haiku-20241022"
        if "haiku" in lowered
        else "claude-3-5-sonnet-20241022" if "sonnet" in lowered else model
    )

    response = client.messages.create(
        model=resolved_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    first_block = response.content[0]
    return (
        getattr(first_block, "text", "").strip()
        if hasattr(first_block, "text")
        else str(first_block)
    )


def generate_real_answer(
    question: str,
    context: list[str],
    config: Mapping[str, Any] | dict[str, Any],
) -> str:
    model = str(config.get("model", "gpt-4o-mini"))
    temperature = float(config.get("temperature", 0.3))
    retriever_k = max(1, int(config.get("retriever_k", 4)))
    prompt_style = str(config.get("prompt_style", "vanilla"))
    reranker = str(config.get("retrieval_reranker", config.get("reranker", "none")))
    max_tokens = int(config.get("max_output_tokens", 384))

    _validate_real_credentials(model)

    selected_context = context[:retriever_k] if context else []
    context_block = (
        "\n\n".join(selected_context)
        if selected_context
        else "No retrieved context provided."
    )
    reasoning_instruction = (
        "Think step by step and explain how each retrieved passage contributes before giving the final answer."
        if prompt_style == "cot"
        else "Answer concisely using evidence from the retrieved passages."
    )

    prompt = (
        f"Question: {question}\n\n"
        f"Retrieved context (top-{retriever_k}, reranker={reranker}):\n{context_block}\n\n"
        f"{reasoning_instruction}\n"
        "Conclude with a sentence that starts with 'Answer:' followed by the final answer."
    )

    lowered = model.lower()
    if "gpt" in lowered:
        return _invoke_openai(model, prompt, temperature, max_tokens)
    return _invoke_anthropic(model, prompt, temperature, max_tokens)


@traigent.optimize(
    eval_dataset=DATASET,
    objectives=["quality", "latency_p95_ms", "cost_usd_per_1k"],
    configuration_space={
        "model": ["gpt-4o", "gpt-4o-mini", "haiku-3.5"],
        "temperature": [0.1, 0.3, 0.7],
        "retriever_k": [3, 5, 8],
        "prompt_style": ["vanilla", "cot"],
        "retrieval_reranker": ["none", "mono_t5"],
        "max_output_tokens": [256, 384],
    },
    metric_functions=build_hotpot_metric_functions(mock_mode=USE_MOCK),
    mock_mode_config={"enabled": USE_MOCK, "override_evaluator": False},
)
def hotpot_agent(question: str, context: list[str] | None = None) -> str:
    """Simulated HotpotQA agent; context is captured via Traigent instrumentation."""

    context = context or []
    config: dict[str, Any] = traigent.get_config()
    if USE_MOCK:
        return generate_case_study_answer(question, config)
    return generate_real_answer(question, context, config)


async def main() -> None:
    result = await hotpot_agent.optimize(
        algorithm="optuna_nsga2",
        max_trials=12,
        parallel_config={"trial_concurrency": 4},
    )

    print("Best configuration:")
    for key, value in result.best_config.items():
        print(f"  {key}: {value}")

    print("\nBest metrics:")
    for key, value in result.best_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    dataset = load_case_study_dataset()
    example = dataset.examples[0]
    hotpot_agent.set_config(result.best_config)
    preview = hotpot_agent(
        question=str(example.input_data.get("question", "")),
        context=list(example.input_data.get("context", [])),
    )

    print("\nSample question:")
    print(f"  {example.input_data.get('question', '')}")
    print("\nModel answer with best config:")
    print(preview)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130)
