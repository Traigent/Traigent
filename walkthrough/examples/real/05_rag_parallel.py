#!/usr/bin/env python3
"""Example 5: RAG Optimization - Parallel evaluation (default on).

Usage (run in a terminal from repo root, works without activating venv):
    export OPENAI_API_KEY="your-key"
    .venv/bin/python walkthrough/examples/real/05_rag_parallel.py
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

import traigent
from traigent.config.parallel import ParallelConfig

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

require_openai_key("05_rag_parallel.py")
sanitize_traigent_api_key()
configure_logging()
logging.getLogger("tokencost.costs").setLevel(logging.ERROR)

os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")

traigent.initialize(execution_mode="edge_analytics")

# Dataset path relative to this file
DATASETS = Path(__file__).parent.parent / "datasets"
OBJECTIVES = ["accuracy", "cost"]
CONFIG_SPACE = {
    "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
    "temperature": [0.0, 0.3, 0.7],
    "k": [1, 3, 5],
    "retrieval_method": ["similarity", "keyword"],
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
    """Build vectorstore once."""
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
    injection_mode="context",  # default injection mode, added explicitly for clarity
    execution_mode="edge_analytics",
)
def rag_qa(question: str) -> str:
    """RAG question answering."""
    config = traigent.get_config()
    k = config.get("k", 3)
    method = config.get("retrieval_method", "similarity")

    # Retrieve documents
    if method == "keyword":
        lowered = question.lower()
        docs = [
            doc
            for doc in KNOWLEDGE_BASE
            if any(token in doc.lower() for token in lowered.split())
        ]
        if not docs:
            docs = KNOWLEDGE_BASE[:k]
        context = "\n".join(docs[:k])
    else:
        vectorstore = get_vectorstore()
        docs = vectorstore.similarity_search(question, k=k)
        context = "\n".join([d.page_content for d in docs])

    # Generate answer
    llm = ChatOpenAI(
        model=config.get("model", "gpt-3.5-turbo"),
        temperature=config.get("temperature", 0.3),
    )

    prompt = (
        "Use the context to answer. Keep the answer short and reuse key terms "
        "from the context when possible. Prefer copying the single most relevant "
        "sentence verbatim.\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    try:
        response = llm.invoke(prompt)
        return str(response.content)
    except Exception as exc:
        print(f"LLM call failed: {type(exc).__name__}: {exc}")
        return f"Error: {type(exc).__name__}: {exc}"


async def main() -> None:
    print("Traigent Example 5: RAG Optimization (parallel eval on by default)")
    print("=" * 50)
    print("Optimizing retrieval (k/method) and generation (model/temp).")
    print_optimization_config(OBJECTIVES, CONFIG_SPACE)
    print_cost_estimate(
        models=CONFIG_SPACE["model"],
        dataset_size=20,
        task_type="rag_qa",
        num_trials=10,
    )

    parallel_enabled = os.getenv("TRAIGENT_PARALLEL", "1").lower() not in (
        "0",
        "false",
        "no",
    )
    parallel_config = None
    if parallel_enabled:
        parallel_config = ParallelConfig(mode="parallel", example_concurrency=2)
        print("Parallel eval enabled (example_concurrency=2).")
        print("Pause-on-error prompts require sequential trials (parallel eval off).")
        print("To disable parallel eval: set TRAIGENT_PARALLEL=0")
    else:
        print("Parallel eval disabled. To enable: set TRAIGENT_PARALLEL=1")

    print_estimated_time("05_rag_parallel.py")
    results = await rag_qa.optimize(
        algorithm="random",
        max_trials=10,
        show_progress=True,
        random_seed=42,
        parallel_config=parallel_config,
    )

    print_results_table(results, CONFIG_SPACE, OBJECTIVES, is_mock=False)

    print("\nBest Configuration Found:")
    print(f"  Retrieval k: {results.best_config.get('k')}")
    if "retrieval_method" in results.best_config:
        print(f"  Method: {results.best_config.get('retrieval_method')}")
    print(f"  Model: {results.best_config.get('model')}")
    print(f"  Temperature: {results.best_config.get('temperature')}")

    print(f"\nAccuracy: {results.best_metrics.get('accuracy', 0):.2%}")


if __name__ == "__main__":
    asyncio.run(main())
