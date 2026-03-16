#!/usr/bin/env python3
"""Example 9: RAG Multi-Objective — Balance accuracy, cost, and latency.

Optimizes a RAG pipeline over model, prompt style, temperature, instruction
mode, and generation token budget (max_tokens). Demonstrates that even with
the best model, wrong settings (high temperature, CoT with a tiny token
budget) can tank accuracy — and Traigent finds the sweet spot automatically.

Usage (run in a terminal from repo root, works without activating venv):
    export OPENAI_API_KEY="your-key"  # pragma: allowlist secret
    .venv/bin/python walkthrough/real/09_rag_multi_objective.py
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

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
    configure_logging,
    print_cost_estimate,
    print_estimated_time,
    print_optimization_config,
    print_results_table,
    require_openai_key,
    sanitize_traigent_api_key,
)
from utils.scoring import semantic_overlap_score

import traigent
from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema

require_openai_key("09_rag_multi_objective.py")
sanitize_traigent_api_key()
configure_logging()
logging.getLogger("tokencost.costs").setLevel(logging.ERROR)

os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")

traigent.initialize(execution_mode="edge_analytics")

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
    "Local, cloud, and hybrid execution modes",
    "Local mode keeps all data on your machine for complete privacy",
    "Edge analytics mode runs optimization locally while sending analytics",
    "Grid search, random search, and Bayesian optimization",
    "A decorator that enables automatic optimization of LLM functions",
    "Using the objectives parameter with metrics like accuracy, cost, latency",
    "A dictionary defining the hyperparameters and their possible values to optimize",
    "Through weighted objective definitions using ObjectiveSchema with maximize/minimize orientations",
    "Hybrid mode runs LLM calls locally but uses cloud for optimization intelligence with privacy enabled",
    "It can optimize both retrieval parameters like k and method, plus generation parameters like model and temperature",
    "Seamless mode auto-overrides LLM calls; context mode adds config to prompts",
    "Custom evaluators let you define your own scoring logic for specialized use cases",
    "Yes, through multi-objective optimization with configurable weights for each metric",
    "Local storage, privacy_enabled flag, and edge_analytics mode for data control",
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


@traigent.optimize(
    eval_dataset=str(DATASETS / "rag_questions.jsonl"),
    objectives=OBJECTIVES,
    scoring_function=semantic_overlap_score,
    configuration_space=CONFIG_SPACE,
    injection_mode="context",
    execution_mode="edge_analytics",
)
def rag_agent(question: str) -> str:
    """RAG agent: retrieves context, answers with configurable generation settings."""
    config = traigent.get_config()

    # Retrieve top-3 relevant documents
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
        base_prompt = f"{system}\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
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
        dataset_size=20,
        task_type="rag_qa",
        num_trials=18,
    )

    print_estimated_time("09_rag_multi_objective.py")
    results = await rag_agent.optimize(
        algorithm="random",
        max_trials=18,
        show_progress=True,
        random_seed=42,
        timeout=600,
    )

    print_results_table(results, CONFIG_SPACE, OBJECTIVES, is_mock=False)

    print("\nBest Configuration Found:")
    print(f"  Model:        {results.best_config.get('model')}")
    print(f"  Prompt:       {results.best_config.get('prompt')}")
    print(f"  Temperature:  {results.best_config.get('temperature')}")
    print(f"  Instructions: {results.best_config.get('instructions')}")
    print(f"  Max Tokens:   {results.best_config.get('max_tokens')}")

    print("\nPerformance:")
    print(f"  Accuracy: {results.best_metrics.get('accuracy', 0):.2%}")
    print(f"  Cost:     ${results.best_metrics.get('cost', 0):.6f}")
    print(f"  Latency:  {results.best_metrics.get('latency', 0):.3f}s")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130)
