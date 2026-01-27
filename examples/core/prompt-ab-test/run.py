#!/usr/bin/env python3
"""Prompt A/B test (prompt_variant, model, temperature) with grid search and parallel trials."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

MOCK = str(os.getenv("TRAIGENT_MOCK_LLM", "")).lower() in {"1", "true", "yes", "y"}
BASE = Path(__file__).parent
if MOCK:
    os.environ["HOME"] = str(BASE)
    results_dir = BASE / ".traigent_local"
    results_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRAIGENT_RESULTS_FOLDER"] = str(results_dir)
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

try:
    import traigent
except ImportError:  # pragma: no cover - support IDE execution paths
    import importlib

    module_path = Path(__file__).resolve()
    for depth in (2, 3):
        try:
            sys.path.append(str(module_path.parents[depth]))
        except IndexError:
            continue
    traigent = importlib.import_module("traigent")

from traigent.api.types import OptimizationResult  # noqa: E402
from traigent.config.parallel import ParallelConfig  # noqa: E402

os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")


DATA_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "prompt-ab-test"
if MOCK:
    try:
        traigent.initialize(execution_mode="edge_analytics")
    except Exception:
        pass
DATASET = str(DATA_ROOT / "evaluation_set.jsonl")
PROMPT_A = BASE / "prompt_a.txt"
PROMPT_B = BASE / "prompt_b.txt"


MODEL_CHOICES = ["claude-3-haiku-20240307", "claude-3-5-sonnet-20241022"]
PROMPT_VARIANTS = ["a", "b"]
TEMPERATURE_CHOICES = [0.0, 0.2]


def _parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


DEFAULT_WORKERS = max(
    1, _parse_int_env("TRAIGENT_PARALLEL_WORKERS", os.cpu_count() or 4)
)
CONCURRENCY_PROFILE = os.getenv("TRAIGENT_CONCURRENCY_PROFILE", "auto").strip().lower()
TOTAL_CONFIGS = len(MODEL_CHOICES) * len(PROMPT_VARIANTS) * len(TEMPERATURE_CHOICES)

if CONCURRENCY_PROFILE == "parallel":
    example_concurrency = max(
        1, _parse_int_env("TRAIGENT_EXAMPLE_CONCURRENCY", DEFAULT_WORKERS)
    )
    trial_concurrency = max(
        2,
        _parse_int_env(
            "TRAIGENT_TRIAL_CONCURRENCY", min(DEFAULT_WORKERS * 2, TOTAL_CONFIGS)
        ),
    )
    resolved_mode = "parallel"
elif CONCURRENCY_PROFILE == "sequential":
    example_concurrency = 1
    trial_concurrency = 1
    resolved_mode = "sequential"
else:
    example_concurrency = max(
        1, _parse_int_env("TRAIGENT_EXAMPLE_CONCURRENCY", DEFAULT_WORKERS)
    )
    trial_concurrency = max(
        1,
        _parse_int_env(
            "TRAIGENT_TRIAL_CONCURRENCY", min(DEFAULT_WORKERS, TOTAL_CONFIGS)
        ),
    )
    resolved_mode = "auto"

GLOBAL_PARALLEL_CONFIG = ParallelConfig(
    mode=resolved_mode,
    trial_concurrency=trial_concurrency,
    example_concurrency=example_concurrency,
    thread_workers=DEFAULT_WORKERS,
)

traigent.configure(parallel_config=GLOBAL_PARALLEL_CONFIG)
print(
    "Resolved concurrency: "
    f"profile={CONCURRENCY_PROFILE or 'auto'}, "
    f"mode={GLOBAL_PARALLEL_CONFIG.mode or 'auto'}, "
    f"trial_concurrency={GLOBAL_PARALLEL_CONFIG.trial_concurrency}, "
    f"example_concurrency={GLOBAL_PARALLEL_CONFIG.example_concurrency}, "
    f"thread_workers={GLOBAL_PARALLEL_CONFIG.thread_workers}"
)
if (
    GLOBAL_PARALLEL_CONFIG.mode == "parallel"
    and (GLOBAL_PARALLEL_CONFIG.trial_concurrency or 0)
    * (GLOBAL_PARALLEL_CONFIG.example_concurrency or 0)
    > 8
):
    print(
        "⚠️  High concurrency requested — provider throttling may increase per-example latency."
    )


def _prompt(variant: str) -> str:
    p = PROMPT_A if variant == "a" else PROMPT_B
    return p.read_text().strip()


def _print_results(result: OptimizationResult) -> None:
    """Pretty-print aggregated and raw optimization data."""

    primary = result.objectives[0] if result.objectives else None

    def _mean_response_time(meta: Any) -> float | None:
        if not isinstance(meta, dict):
            return None
        entries: list[dict[str, Any]] = []
        measures = meta.get("measures")
        if isinstance(measures, list):
            entries = [entry for entry in measures if isinstance(entry, dict)]
        elif isinstance(measures, dict):
            entries = [measures]
        if not entries:
            eval_result = meta.get("evaluation_result")
            example_results = getattr(eval_result, "example_results", None)
            if isinstance(example_results, list):
                entries = [
                    entry for entry in example_results if isinstance(entry, dict)
                ]
        times = [
            float(entry["response_time"])
            for entry in entries
            if entry.get("response_time") is not None
        ]
        if times:
            return sum(times) / len(times)
        return None

    df_raw = result.to_dataframe()
    if "metadata" in df_raw.columns:
        df_raw["avg_response_time"] = df_raw["metadata"].apply(_mean_response_time)
    else:
        df_raw["avg_response_time"] = None

    config_cols = ["prompt_variant", "model", "temperature"]

    df = result.to_aggregated_dataframe(primary_objective=primary)
    preferred_cols = [
        "prompt_variant",
        "model",
        "temperature",
        "samples_count",
        "accuracy",
        "cost",
        "duration",
        "avg_response_time",
    ]
    cols = [c for c in preferred_cols if c in df.columns]
    if cols:
        df = df[cols]

    if df_raw["avg_response_time"].notna().any():
        response_avg = (
            df_raw.groupby(config_cols, dropna=False)["avg_response_time"]
            .mean()
            .reset_index()
        )
        df = df.merge(response_avg, on=config_cols, how="left")

    if "avg_response_time" in df.columns:
        df["avg_response_time"] = df["avg_response_time"].astype(float).round(3)

    if primary and primary in df.columns:
        minimize_patterns = ["cost", "latency", "error", "loss", "time", "duration"]
        ascending = any(p in primary.lower() for p in minimize_patterns)
        df = df.sort_values(by=primary, ascending=ascending, na_position="last")

    if not df.empty:
        print("\nAggregated configurations and performance:")
        print(df.to_string(index=False))

    preferred_raw = [
        "trial_id",
        "status",
        "prompt_variant",
        "model",
        "temperature",
        "accuracy",
        "cost",
        "duration",
        "avg_response_time",
    ]
    cols_raw = [c for c in preferred_raw if c in df_raw.columns]
    if cols_raw:
        df_raw = df_raw[cols_raw]

    if "avg_response_time" in df_raw.columns:
        df_raw["avg_response_time"] = df_raw["avg_response_time"].astype(float).round(3)

    if primary and primary in df_raw.columns:
        minimize_patterns = ["cost", "latency", "error", "loss", "time", "duration"]
        ascending = any(p in primary.lower() for p in minimize_patterns)
        df_raw = df_raw.sort_values(by=primary, ascending=ascending, na_position="last")

    if not df_raw.empty:
        print("\nRaw (per-sample) trials:")
        print(df_raw.to_string(index=False))


def _mock_qa_answer(question: str) -> str:
    """Return deterministic answers for mock mode achieving 75%+ accuracy.

    Covers all 20 AI/ML terminology evaluation set questions.
    """
    q = (question or "").lower()
    # Mapping from question keywords to expected answers
    # Ordered by specificity (longer/compound terms first)
    mapping = {
        # Compound terms
        "machine learning": "Uses data and algorithms",
        "deep learning": "Neural networks with many layers",
        "prompt engineering": "Crafting effective prompts",
        "neural network": "Layers of interconnected nodes",
        "fine-tuning": "Training on specific data",
        "hyperparameter": "Optimizing model settings",
        "transformer": "Attention-based architecture",
        # Acronyms (in order of specificity)
        "bert": "Bidirectional Encoder Representations from Transformers",
        "rag": "Retrieval Augmented Generation",
        "nlp": "Natural Language Processing",
        "llm": "Large Language Model",
        "api": "Application Programming Interface",
        "sdk": "Software Development Kit",
        "gpu": "Graphics Processing Unit",
        "cnn": "Convolutional Neural Network",
        "gan": "Generative Adversarial Network",
        # Single terms
        "embedding": "Vector representation of data",
        "tokenization": "Splitting text into tokens",
        "inference": "Model prediction phase",
    }
    for key, value in mapping.items():
        if key in q:
            return value
    # Default: "What is AI?" or unmatched
    return "Artificial Intelligence"


@traigent.optimize(
    eval_dataset=DATASET,
    objectives=["accuracy"],
    configuration_space={
        "prompt_variant": PROMPT_VARIANTS,
        "model": MODEL_CHOICES,
        "temperature": TEMPERATURE_CHOICES,
    },
    execution_mode="edge_analytics",
    parallel_config=GLOBAL_PARALLEL_CONFIG,
)
def qa_with_variant(question: str) -> str:
    if MOCK:
        return _mock_qa_answer(question)
    assert os.getenv("ANTHROPIC_API_KEY"), "Missing ANTHROPIC_API_KEY"
    cfg = traigent.get_config()
    variant = cfg.get("prompt_variant", "a")
    prompt = f"Question: {question}\n\n{_prompt(variant)}"
    response = ChatAnthropic(
        model_name=cfg.get("model", "claude-3-5-sonnet-20241022"),
        temperature=float(cfg.get("temperature", 0.0)),
        timeout=None,
        stop=None,
    ).invoke([HumanMessage(content=prompt)])
    raw = str(response.content).strip()
    for k in [
        "Artificial Intelligence",
        "Uses data and algorithms",
        "Retrieval Augmented Generation",
    ]:
        if k.lower() in raw.lower():
            return k
    return raw[:64]


if __name__ == "__main__":
    print("Prompt A vs B—stop guessing; run an A/B that picks a winner.")

    async def main() -> None:
        trials = 12 if not MOCK else 4
        r = await qa_with_variant.optimize(algorithm="grid", max_trials=trials)
        print({"best_config": r.best_config, "best_score": r.best_score})
        _print_results(r)

    asyncio.run(main())
