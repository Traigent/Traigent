#!/usr/bin/env python3
"""Hello World — Real LangChain LLM + RAG (parameter injection)."""

from __future__ import annotations

import asyncio
import json
import os
import pprint
import random
import sys
import time
from pathlib import Path

MOCK = str(os.getenv("TRAIGENT_MOCK_LLM", "")).lower() in {"1", "true", "yes", "y"}
BASE = Path(__file__).parent
if MOCK:
    os.environ["HOME"] = str(BASE)
    results_dir = BASE / ".traigent_local"
    results_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRAIGENT_RESULTS_FOLDER"] = str(results_dir)

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage


class RetrieverBase:
    def __init__(self, docs: list[Document]) -> None:
        self._docs = docs
        self.k = 3

    def get_relevant_documents(
        self, query: str
    ) -> list[Document]:  # pragma: no cover - protocol
        raise NotImplementedError


try:  # pragma: no cover - optional dependency
    from langchain_community.retrievers import (
        BM25Retriever,  # type: ignore[import-not-found]
    )

    class _BM25RetrieverWrapper(RetrieverBase):
        def __init__(self, docs: list[Document]) -> None:
            super().__init__(docs)
            self._delegate = BM25Retriever.from_documents(docs)

        def get_relevant_documents(self, query: str) -> list[Document]:
            self._delegate.k = self.k
            return self._delegate.get_relevant_documents(query)

    def _build_retriever(docs: list[Document]) -> RetrieverBase:
        return _BM25RetrieverWrapper(docs)

except Exception:  # pragma: no cover - fallback retriever

    class _FallbackRetriever(RetrieverBase):
        def get_relevant_documents(self, query: str) -> list[Document]:
            tokens = query.lower().split()
            scored: list[tuple[int, Document]] = []
            for doc in self._docs:
                score = sum(doc.page_content.lower().count(token) for token in tokens)
                scored.append((score, doc))
            scored.sort(key=lambda item: item[0], reverse=True)
            return [doc for score, doc in scored[: self.k] if score > 0]

    def _build_retriever(docs: list[Document]) -> RetrieverBase:
        return _FallbackRetriever(docs)


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
from traigent.utils.error_handler import APIKeyError  # noqa: E402

os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")


# Uncomment to enable verbose logging
# traigent.configure(logging_level="DEBUG")

# Set to True to enable verbose invocation output in the console
__VERBOSE__ = os.getenv("TRAIGENT_VERBOSE", "").lower() in {"1", "true", "yes"}

DATA_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "hello-world"
if MOCK:
    # Redirect home to repo-local to avoid sandbox home write
    os.environ.setdefault("HOME", str(BASE))
    traigent.initialize(execution_mode="edge_analytics")
DATASET = str(DATA_ROOT / "evaluation_set.jsonl")
PROMPT_PATH = BASE / "prompt.txt"
CONTEXT_PATH = DATA_ROOT / "context_documents.jsonl"


def _prompt() -> str:
    return PROMPT_PATH.read_text().strip()


def _load_docs() -> list[Document]:
    if not CONTEXT_PATH.exists():
        return []
    docs: list[Document] = []
    with open(CONTEXT_PATH) as f:
        for line in f:
            obj = json.loads(line)
            docs.append(Document(page_content=obj["page_content"]))
    return docs


_PROMPT = _prompt()
_DOCS = _load_docs()
_RETRIEVER: RetrieverBase | None = _build_retriever(_DOCS) if _DOCS else None


def _mock_answer(question: str) -> str:
    """Return deterministic answers for mock mode to keep tests stable."""
    q = (question or "").lower()
    mapping = {
        "rag": "Retrieval Augmented Generation",
        "machine learning": "Uses data and algorithms",
        "instruction": "A list of steps",
    }
    for key, value in mapping.items():
        if key in q:
            return value
    if any(token in q for token in ("never", "private")):
        return "A private key"
    return "Artificial Intelligence"


def _generate_mock_telemetry(model: str, use_rag: bool, fast_mode: bool = True) -> dict:
    """Generate realistic mock telemetry data for cost and latency.

    Args:
        model: The model name being tested
        use_rag: Whether RAG is enabled
        fast_mode: If True, use shorter latencies for faster testing (default: True)
    """
    # Model-specific cost ranges (per 1M tokens: input, output)
    model_costs = {
        "claude-3-5-sonnet-20241022": (3.0, 15.0),  # $3/$15 per 1M tokens
        "claude-3-opus-20240229": (15.0, 75.0),  # $15/$75 per 1M tokens
        "claude-3-haiku-20240307": (0.25, 1.25),  # $0.25/$1.25 per 1M tokens
    }

    # Model-specific latency ranges (seconds)
    # In fast_mode, latencies are scaled down 10x for faster testing
    latency_scale = 0.1 if fast_mode else 1.0
    model_latency = {
        "claude-3-5-sonnet-20241022": (0.8 * latency_scale, 2.5 * latency_scale),
        "claude-3-opus-20240229": (1.2 * latency_scale, 3.5 * latency_scale),
        "claude-3-haiku-20240307": (0.3 * latency_scale, 1.2 * latency_scale),
    }

    # Get model-specific parameters or use defaults
    input_cost, output_cost = model_costs.get(model, (3.0, 15.0))
    min_latency, max_latency = model_latency.get(
        model, (0.5 * latency_scale, 2.0 * latency_scale)
    )

    # Realistic token counts for Q&A
    # RAG uses more input tokens due to retrieved context
    input_tokens = random.randint(150, 300) if use_rag else random.randint(80, 150)
    output_tokens = random.randint(50, 120)

    # Calculate cost
    cost = (input_tokens * input_cost / 1_000_000) + (
        output_tokens * output_cost / 1_000_000
    )

    # Add some variance to latency (RAG might be slightly slower due to retrieval)
    latency = random.uniform(min_latency, max_latency)
    if use_rag:
        latency *= random.uniform(1.0, 1.3)  # 0-30% slower with RAG

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost": round(cost, 6),
        "latency": round(latency, 3),
    }


def _build_context(question: str, use_rag: bool, top_k: int) -> str | None:
    if not use_rag or _RETRIEVER is None:
        return None
    _RETRIEVER.k = max(1, top_k)
    docs = _RETRIEVER.get_relevant_documents(question)
    if not docs:
        return None
    return "\n\n".join(doc.page_content for doc in docs)


def _compose_prompt(question: str, context: str | None) -> str:
    prefix = f"Context:\n{context}\n\n" if context else ""
    return f"{prefix}Question: {question}\n\n{_PROMPT}"


def _normalize_answer(raw: str) -> str:
    canonical = {
        "artificial intelligence": "Artificial Intelligence",
        "uses data and algorithms": "Uses data and algorithms",
        "retrieval augmented generation": "Retrieval Augmented Generation",
        "a list of steps": "A list of steps",
        "a private key": "A private key",
    }
    lowered = raw.lower()
    for key, value in canonical.items():
        if key in lowered:
            return value
    return raw[:128]


def _mock_cost_metric(
    output=None, expected=None, example=None, config=None, llm_metrics=None
):
    """Custom cost metric that generates realistic values in mock mode.

    Args:
        output: The actual output from the function
        expected: The expected output (not used for cost)
        example: The example object with input/output/metadata
        config: The configuration dictionary with model, use_rag, etc.
        llm_metrics: Captured LLM telemetry metrics (in real mode)

    Returns:
        float: Cost value in dollars
    """
    # If we have real LLM metrics, use them
    if llm_metrics and "total_cost" in llm_metrics and llm_metrics["total_cost"] > 0:
        return llm_metrics["total_cost"]

    # Otherwise, generate mock cost based on configuration
    if MOCK and config:
        model = config.get("model", "claude-3-5-sonnet-20241022")
        use_rag = bool(config.get("use_rag", True))
        telemetry = _generate_mock_telemetry(model, use_rag, fast_mode=True)
        return telemetry["cost"]

    return 0.0


def _invoke_llm(prompt: str, model: str, temperature: float) -> str:
    response = ChatAnthropic(
        model_name=model,
        temperature=temperature,
        timeout=None,
        stop=None,
    ).invoke([HumanMessage(content=prompt)])
    return str(response.content).strip()


@traigent.optimize(
    eval_dataset=DATASET,
    objectives=["cost"],
    metric_functions={
        "cost": _mock_cost_metric
    },  # Custom cost metric for realistic mock data
    configuration_space={
        "model": [
            "claude-3-5-sonnet-20241022",
            # "claude-3-opus-20240229",
            "claude-3-haiku-20240307",
        ],
        "temperature": [0.0],
        "use_rag": [True, False],
        "top_k": [1, 2, 3],
    },
    execution_mode="edge_analytics",
)
def answer_question(question: str) -> str:
    """Answer a question using either mock mode or real LLM API."""
    if MOCK:
        cfg = traigent.get_config()
        model = cfg.get("model", "claude-3-5-sonnet-20241022")
        use_rag = bool(cfg.get("use_rag", True))

        # Generate realistic telemetry (fast_mode=True for testing)
        telemetry = _generate_mock_telemetry(model, use_rag, fast_mode=True)

        # Simulate realistic API latency
        time.sleep(telemetry["latency"])

        # Store telemetry for Traigent to capture
        if hasattr(answer_question, "_mock_telemetry"):
            answer_question._mock_telemetry = telemetry

        return _mock_answer(question)

    if os.getenv("ANTHROPIC_API_KEY") is None:
        raise APIKeyError("ANTHROPIC_API_KEY")

    cfg = traigent.get_config()
    model = cfg.get("model", "claude-3-5-sonnet-20241022")
    temperature = float(cfg.get("temperature", 0.2))
    context = _build_context(
        question,
        use_rag=bool(cfg.get("use_rag", True)),
        top_k=int(cfg.get("top_k", 2)),
    )
    prompt = _compose_prompt(question, context)
    raw = _invoke_llm(prompt, model=model, temperature=temperature)
    if __VERBOSE__:
        pprint.pprint(cfg, indent=4, width=120)
        pprint.pprint(prompt, indent=4, width=120)
        pprint.pprint(raw, indent=4, width=120)
    return _normalize_answer(raw)


if __name__ == "__main__":
    print("Wondering whether RAG helps and what top_k to use?")

    async def main() -> None:
        trials = 10  # Change this number to set max configurations
        r: OptimizationResult = await answer_question.optimize(
            algorithm="random", max_trials=trials
        )
        print({"best_config": r.best_config, "best_score": r.best_score})

        # Print aggregated table (one row per configuration) for readability
        df = r.to_aggregated_dataframe(primary_objective="cost")
        preferred_cols = [
            "model",
            "temperature",
            "use_rag",
            "top_k",
            "samples_count",
            "accuracy",
            "cost",
            "duration",
        ]
        cols = [c for c in preferred_cols if c in df.columns]
        if cols:
            df = df[cols]

        # Sort by primary objective if available
        primary = r.objectives[0] if r.objectives else None
        if primary and primary in df.columns:
            assert isinstance(primary, str)
            minimize_patterns = ["cost", "latency", "error", "loss", "time", "duration"]
            ascending = any(p in primary.lower() for p in minimize_patterns)
            df = df.sort_values(by=primary, ascending=ascending, na_position="last")  # type: ignore[call-arg]

        print("\nAggregated configurations and performance:")
        print(df.to_string(index=False))

        # Also show raw per-sample table for debugging (optional)
        df_raw = r.to_dataframe()
        cols_raw = [
            "trial_id",
            "status",
            "model",
            "temperature",
            "use_rag",
            "top_k",
            "accuracy",
            "cost",
            "duration",
        ]
        cols_raw = [c for c in cols_raw if c in df_raw.columns]
        if cols_raw:
            df_raw = df_raw[cols_raw]
        primary = r.objectives[0] if r.objectives else None
        if primary and primary in df_raw.columns:
            assert isinstance(primary, str)
            minimize_patterns = ["cost", "latency", "error", "loss", "time", "duration"]
            ascending = any(p in primary.lower() for p in minimize_patterns)
            df_raw = df_raw.sort_values(by=primary, ascending=ascending, na_position="last")  # type: ignore[call-arg]
        print("\nRaw (per-sample) trials:")
        print(df_raw.to_string(index=False))

    asyncio.run(main())
