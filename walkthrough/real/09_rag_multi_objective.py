#!/usr/bin/env python3
# ruff: noqa: E402
"""Example 9: RAG Multi-Objective - Balance accuracy, cost, and latency.

Demonstrates how CoT instructions that burn a small max_tokens budget before
reaching the answer, a minimal vs role-based prompt, the wrong temperature, or
the wrong model can tank accuracy - and how Traigent finds the sweet spot
across the three weighted objectives (accuracy 50%, cost 20%, latency 30%).

Retrieval is deliberately held FIXED here (similarity search, k=3, see
``rag_agent`` below): every axis this example sweeps is generation-side and
listed in ``CONFIG_SPACE`` - model, prompt, temperature, instructions,
max_tokens. For an example that sweeps the RETRIEVAL parameters themselves,
see ``walkthrough/real/05_rag_parallel.py``, whose ``CONFIG_SPACE`` varies
``k`` and ``retrieval_method``.

Usage (run in a terminal from repo root, works without activating venv):
    export OPENAI_API_KEY="your-key"  # pragma: allowlist secret
    .venv/bin/python walkthrough/real/09_rag_multi_objective.py

If OPENAI_API_KEY is missing, this script exits with an error and suggests
running the mock walkthrough instead.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import maybe_run_mock_example

maybe_run_mock_example(__file__)

try:
    from langchain_community.vectorstores import FAISS
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except Exception as exc:
    raise SystemExit(
        "Missing dependencies for this example. Install with "
        '"pip install traigent[integrations]" or install faiss-cpu and '
        "langchain-openai."
    ) from exc

from utils.helpers import (
    build_results_table_callback,
    configure_logging,
    print_cost_estimate,
    print_estimated_time,
    print_optimization_config,
    sanitize_traigent_api_key,
)
from utils.scoring import semantic_overlap_score

import traigent
from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema

sanitize_traigent_api_key()
configure_logging()
logging.getLogger("tokencost.costs").setLevel(logging.ERROR)

os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")

traigent.initialize(offline=True)

# Dataset path relative to this file
DATASETS = Path(__file__).parent.parent / "datasets"

OBJECTIVES = ObjectiveSchema.from_objectives(
    [
        ObjectiveDefinition("accuracy", orientation="maximize", weight=0.5),
        ObjectiveDefinition("cost", orientation="minimize", weight=0.2),
        ObjectiveDefinition("latency", orientation="minimize", weight=0.3),
    ]
)

CONFIG_SPACE = {
    "model": [
        "gpt-3.5-turbo",
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-5.2",
        "gpt-5-nano",
        "gpt-5.1",
    ],
    "prompt": ["minimal", "role_based"],
    "temperature": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "instructions": ["CoT", "direct"],
    "max_tokens": [50, 100, 200],
}

KNOWLEDGE_BASE = [
    "Traigent optimizes AI applications without code changes",
    "Seamless mode intercepts and overrides hardcoded LLM parameters",
    "Parameter mode provides explicit configuration control via function parameters",
    "Local, cloud, and hybrid_api execution modes",
    "Local mode keeps all data on your machine for complete privacy",
    "edge_analytics mode has been removed; local mode (offline=True) runs "
    "optimization on-device with zero egress",
    "Grid search, random search, and Bayesian optimization",
    "A decorator that enables automatic optimization of LLM functions",
    "Using the objectives parameter with metrics like accuracy, cost, latency",
    "A dictionary defining the hyperparameters and their possible values to optimize",
    "Through weighted objective definitions using ObjectiveSchema with maximize/minimize orientations",
    "Cloud mode runs LLM calls locally but uses Traigent's backend for optimization intelligence, sending only metrics",
    "It can optimize both retrieval parameters like k and method, plus generation parameters like model and temperature",
    "Seamless mode auto-overrides LLM calls; context mode adds config to prompts",
    "Custom evaluators let you define your own scoring logic for specialized use cases",
    "Yes, through multi-objective optimization with configurable weights for each metric",
    "Local storage and offline=True for zero-egress data control; "
    "privacy_enabled and edge_analytics mode are deprecated",
    "Via adapters that intercept LangChain LLM calls and inject optimized configurations",
    "A JSONL file or Dataset object containing input/output pairs for evaluation",
    "By running trials, evaluating against objectives, and selecting the config with best weighted score",
]

_vectorstore = None


def get_vectorstore() -> FAISS:
    """Build vectorstore once (lazy init)."""
    global _vectorstore
    if _vectorstore is None:
        embeddings = OpenAIEmbeddings()
        _vectorstore = FAISS.from_texts(KNOWLEDGE_BASE, embeddings)
    return _vectorstore


EVAL_DATASET = DATASETS / "rag_questions.jsonl"

# Counted, not hardcoded: the printed cost estimate previously said 20 examples
# against a 13-row dataset, overstating the spend by ~1.5x. Deriving it here
# means the estimate cannot drift from the file again.
EVAL_DATASET_SIZE = sum(
    1 for line in EVAL_DATASET.read_text(encoding="utf-8").splitlines() if line.strip()
)


@traigent.optimize(
    eval_dataset=str(EVAL_DATASET),
    objectives=OBJECTIVES,
    scoring_function=semantic_overlap_score,
    configuration_space=CONFIG_SPACE,
    injection_mode="context",  # default injection mode, added explicitly for clarity
    offline=True,
)
def rag_agent(question: str) -> str:
    """RAG agent: retrieves context, answers with configurable generation settings."""
    config = traigent.get_config()

    # Retrieve top-3 relevant documents. k is deliberately FIXED and not in
    # CONFIG_SPACE: this example sweeps generation-side knobs only (see the
    # module docstring); 05_rag_parallel.py is the retrieval sweep.
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n".join(d.page_content for d in docs)

    model = config.get("model", "gpt-3.5-turbo")
    temperature = config.get("temperature", 0.3)
    max_tokens = config.get("max_tokens", 100)
    prompt_style = config.get("prompt", "minimal")
    instructions = config.get("instructions", "direct")

    # Prompt style: minimal vs role-based preamble
    if prompt_style == "role_based":
        system = "You are a precise question-answering assistant. Use only the provided context."
        base_prompt = (
            f"{system}\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
    else:
        base_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    # Instruction mode: direct answer vs chain-of-thought
    # Note: CoT with low max_tokens truncates reasoning before reaching the answer
    if instructions == "CoT":
        prompt = f"Think step by step before answering.\n{base_prompt}"
    else:
        prompt = base_prompt

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    try:
        response = llm.invoke(prompt)
        return str(response.content)
    except Exception as exc:
        print(f"LLM call failed: {type(exc).__name__}: {exc}")
        return f"Error: {type(exc).__name__}: {exc}"


async def main() -> None:
    print("Traigent Example 9: RAG Multi-Objective Optimization")
    print("=" * 55)
    print("Balancing accuracy (50%), cost (20%), latency (30%).")
    print_optimization_config(OBJECTIVES, CONFIG_SPACE)
    print_cost_estimate(
        models=CONFIG_SPACE["model"],
        dataset_size=EVAL_DATASET_SIZE,
        task_type="rag_qa",
        num_trials=18,
    )

    print_estimated_time("09_rag_multi_objective.py")
    results = await rag_agent.optimize(
        algorithm="random",
        max_trials=18,
        random_seed=42,
        callbacks=[build_results_table_callback(is_mock=False, show_progress=True)],
    )

    print("\nBest Configuration Found:")
    print(f"  Model:        {results.best_config.get('model')}")
    print(f"  Prompt:       {results.best_config.get('prompt')}")
    print(f"  Temperature:  {results.best_config.get('temperature')}")
    print(f"  Instructions: {results.best_config.get('instructions')}")
    print(f"  Max Tokens:   {results.best_config.get('max_tokens')}")

    print("\nPerformance:")
    print(f"  Accuracy: {results.best_metrics.get('accuracy', 0):.2%}")
    print(f"  Cost:     ${results.best_metrics.get('cost', 0):.6f}")
    print(f"  Latency:  {results.best_metrics.get('latency', 0):.0f}ms")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130) from None
