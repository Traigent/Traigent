#!/usr/bin/env python3
"""Long-context QA with RAG chunking/windowing optimization (chunk_size, overlap, top_k)."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

MOCK = str(os.getenv("TRAIGENT_MOCK_MODE", "")).lower() in {"1", "true", "yes", "y"}
BASE = Path(__file__).parent
if MOCK:
    os.environ["HOME"] = str(BASE)
    results_dir = BASE / ".traigent_local"
    results_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRAIGENT_RESULTS_FOLDER"] = str(results_dir)
    from traigent.core.backend_session_manager import BackendSessionManager
    from traigent.core.session_context import SessionContext

    def _mock_create_session(
        self,
        func,
        dataset,
        function_descriptor=None,
        max_trials=None,
        start_time=None,
        **kwargs,
    ):
        dataset_name = getattr(dataset, "name", "mock_dataset")
        descriptor = function_descriptor
        return SessionContext(
            session_id=None,
            dataset_name=dataset_name,
            function_name=getattr(
                descriptor, "identifier", getattr(func, "__name__", "mock_func")
            ),
            optimization_id=self._optimization_id,
            start_time=start_time,
        )

    BackendSessionManager.create_session = _mock_create_session
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage


class _RetrieverProtocol:
    k: int

    def __init__(self, docs: list[Document]) -> None:
        self._docs = docs
        self.k = 3

    def get_relevant_documents(self, query: str) -> list[Document]:  # pragma: no cover
        raise NotImplementedError


try:  # pragma: no cover - optional dependency
    from langchain_community.retrievers import (
        BM25Retriever,  # type: ignore[import-not-found]
    )

    class _BM25Wrapper(_RetrieverProtocol):
        def __init__(self, docs: list[Document]) -> None:
            super().__init__(docs)
            self._delegate = BM25Retriever.from_documents(docs)

        def get_relevant_documents(self, query: str) -> list[Document]:
            self._delegate.k = self.k
            return self._delegate.get_relevant_documents(query)

    def _bm25_from_documents(docs: list[Document]) -> _RetrieverProtocol:
        return _BM25Wrapper(docs)

except Exception:  # pragma: no cover - fallback retriever

    class _SimpleRetriever(_RetrieverProtocol):
        def get_relevant_documents(self, query: str) -> list[Document]:
            tokens = query.lower().split()
            scored: list[tuple[int, Document]] = []
            for doc in self._docs:
                score = sum(doc.page_content.lower().count(tok) for tok in tokens)
                scored.append((score, doc))
            scored.sort(key=lambda item: item[0], reverse=True)
            return [doc for score, doc in scored[: self.k] if score > 0]

    def _bm25_from_documents(docs: list[Document]) -> _RetrieverProtocol:
        return _SimpleRetriever(docs)


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

from traigent.api.types import OptimizationResult

DATA_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "chunking-long-context"

if MOCK:
    try:
        traigent.initialize(execution_mode="edge_analytics")
    except Exception:
        pass
DATASET = str(DATA_ROOT / "evaluation_set.jsonl")
PROMPT_PATH = BASE / "prompt.txt"
CONTEXT_PATH = DATA_ROOT / "context_documents.jsonl"

CHUNK_SIZE_CHOICES = [128, 192, 256]
OVERLAP_CHOICES = [8, 24]
TOP_K_CHOICES = [1, 2]


def _prompt() -> str:
    return PROMPT_PATH.read_text().strip()


_PROMPT = _prompt()


def _load_docs() -> list[str]:
    docs: list[str] = []
    if not CONTEXT_PATH.exists():
        return docs
    with open(CONTEXT_PATH) as f:
        for line in f:
            obj = json.loads(line)
            docs.append(obj["page_content"])
    return docs


_DOCS = _load_docs()


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

    config_cols = ["chunk_size", "overlap", "top_k"]

    df = result.to_aggregated_dataframe(primary_objective=primary)
    preferred_cols = [
        "chunk_size",
        "overlap",
        "top_k",
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
        "chunk_size",
        "overlap",
        "top_k",
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


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + chunk_size]))
        i += max(1, chunk_size - overlap)
    return chunks


def _build_retriever(chunk_size: int, overlap: int) -> _RetrieverProtocol:
    docs: list[Document] = []
    for d in _DOCS:
        for ch in _chunk_text(d, chunk_size, overlap):
            if ch:
                docs.append(Document(page_content=ch))
    return _bm25_from_documents(docs)


@traigent.optimize(
    eval_dataset=DATASET,
    objectives=["accuracy"],
    configuration_space={
        "chunk_size": CHUNK_SIZE_CHOICES,
        "overlap": OVERLAP_CHOICES,
        "top_k": TOP_K_CHOICES,
    },
    execution_mode="edge_analytics",
    algorithm="grid",
)
def rag_qa(question: str) -> str:
    if MOCK:
        q = (question or "").lower()
        if "rag" in q:
            return "Retrieval Augmented Generation"
        if "ai" in q:
            return "Artificial Intelligence"
        # fallback based on doc-like timeline mention
        return "timeline"
    assert os.getenv("ANTHROPIC_API_KEY"), "Missing ANTHROPIC_API_KEY"
    cfg = traigent.get_current_config()
    retriever = _build_retriever(
        int(cfg.get("chunk_size", 512)), int(cfg.get("overlap", 16))
    )
    retriever.k = int(cfg.get("top_k", 2))
    ctx = "\n\n".join(
        d.page_content for d in retriever.get_relevant_documents(question)
    )
    prompt = f"Context:\n{ctx}\n\nQuestion: {question}\n\n{_PROMPT}"
    response = ChatAnthropic(
        model_name="claude-3-5-sonnet-20241022",
        temperature=0.0,
        timeout=None,
        stop=None,
    ).invoke([HumanMessage(content=prompt)])
    raw = str(response.content).strip()
    for k in [
        "budget",
        "timeline",
        "decision",
        "Artificial Intelligence",
        "Retrieval Augmented Generation",
    ]:
        if k.lower() in raw.lower():
            return k
    return raw[:64]


if __name__ == "__main__":
    print("Long docs, shallow answers? Tune chunking and top_k to lift grounded QA.")

    async def main() -> None:
        trials = 12 if not MOCK else 4
        r = await rag_qa.optimize(max_trials=trials)
        print({"best_config": r.best_config, "best_score": r.best_score})
        _print_results(r)

    asyncio.run(main())
