#!/usr/bin/env python3
# ruff: noqa: E402
"""Advanced walkthrough: Bedrock multi-agent RAG through LiteLLM.

Run from the repository root:

    TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true TRAIGENT_COST_APPROVED=true \
        .venv/bin/python walkthrough/real/advanced/04_multi_agent_bedrock.py

For a real run, unset TRAIGENT_MOCK_LLM and configure local AWS Bedrock
credentials with the standard AWS environment variables or profile.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

# Keep LiteLLM on its bundled model-cost map. Importing LiteLLM otherwise tries
# to refresh the map before falling back to the local backup.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")
os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")

SCRIPT_DIR = Path(__file__).parent
WALKTHROUGH_ROOT = (SCRIPT_DIR / ".." / "..").resolve()
DATASET_PATH = str((WALKTHROUGH_ROOT / "datasets" / "rag_questions.jsonl").resolve())
RESULTS_FOLDER = WALKTHROUGH_ROOT / ".traigent_local"
JOBLIB_FOLDER = RESULTS_FOLDER / "joblib"
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
JOBLIB_FOLDER.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("TRAIGENT_DATASET_ROOT", str(WALKTHROUGH_ROOT))
os.environ.setdefault("TRAIGENT_RESULTS_FOLDER", str(RESULTS_FOLDER))
os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(JOBLIB_FOLDER))

if os.getenv("TRAIGENT_OFFLINE_MODE", "").lower() in {"1", "true", "yes"}:
    os.environ.pop("TRAIGENT_BACKEND_URL", None)
    os.environ.pop("TRAIGENT_API_KEY", None)

sys.path.insert(0, str(WALKTHROUGH_ROOT))

try:
    import litellm
    from langchain_community.vectorstores import FAISS
    from langchain_core.embeddings import Embeddings
except Exception as exc:
    raise SystemExit(
        "Missing advanced Bedrock walkthrough dependencies. Install with "
        '"pip install -e ".[integrations]"" or install litellm, '
        "langchain-core, langchain-community, and faiss-cpu."
    ) from exc

from utils.scoring import semantic_overlap_score

import traigent
from traigent.api.parameter_ranges import Choices
from traigent.api.types import AgentDefinition
from traigent.observability import add_agent_span
from traigent.utils.cost_calculator import cost_from_tokens
from traigent.utils.env_config import is_mock_llm

# Curated generator candidates.
#
# NOTE on Legacy/EOL models (issue #1180): Bedrock reports an inference profile
# as ACTIVE in list-inference-profiles even when invoking it returns
# ResourceNotFoundException at the account level — so a pricing/list-based check
# (e.g. tests/cost_coverage/test_model_price_coverage.py) green-lights models
# that fail every call. The only reliable signal is to *invoke* the model.
#
# This list was refreshed to models that were invoke-verified reachable in
# us-east-1 on 2026-06-07. The three Claude 3 / 3.5 profiles previously shipped
# here (claude-3-5-haiku, claude-3-5-sonnet-v2, claude-3-haiku) returned
# ResourceNotFoundException on every converse call and were removed.
_CANDIDATE_BEDROCK_MODEL_IDS = [
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.amazon.nova-micro-v1:0",
    "us.amazon.nova-lite-v1:0",
    "us.meta.llama3-1-8b-instruct-v1:0",
    "us.meta.llama3-1-70b-instruct-v1:0",
]


def _reachable_bedrock_models(candidates: list[str]) -> list[str]:
    """Invoke-or-skip preflight: keep only models that actually answer a call.

    Legacy/EOL gating is invisible to list-inference-profiles (issue #1180), so
    the durable guard is a real 1-token invoke. Unreachable models are dropped
    from the grid with a clear notice instead of silently scoring 0/N and
    inflating the run's "success rate". Skipped via TRAIGENT_BEDROCK_SKIP_PREFLIGHT=1
    (and entirely in mock mode). If *every* candidate fails the probe — e.g. no
    AWS credentials at all — the full candidate list is returned unchanged so the
    walkthrough still runs and surfaces the underlying error rather than an empty grid.
    """
    if is_mock_llm() or os.getenv("TRAIGENT_BEDROCK_SKIP_PREFLIGHT", "").lower() in {
        "1",
        "true",
        "yes",
    }:
        return list(candidates)

    reachable: list[str] = []
    for model_id in candidates:
        try:
            litellm.completion(
                model=f"bedrock/{model_id}",
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            reachable.append(model_id)
        except Exception as exc:  # noqa: BLE001 - any invoke failure means skip
            err = type(exc).__name__
            print(f"  ⊘ skipping bedrock/{model_id}: not reachable ({err})")

    if not reachable:
        print(
            "  ⚠ no candidate Bedrock model was reachable; "
            "running the full list so the underlying error surfaces."
        )
        return list(candidates)
    return reachable


BEDROCK_GENERATOR_MODEL_IDS = _reachable_bedrock_models(_CANDIDATE_BEDROCK_MODEL_IDS)

DEFAULT_EMBEDDING_MODEL = os.getenv(
    "BEDROCK_EMBED_MODEL_ID",
    "bedrock/amazon.titan-embed-text-v2:0",
)

# Knowledge base for a *non-ceiling* RAG eval (issue #1182).
#
# The previous KB/dataset asked questions whose verbatim answer was a single
# retrieved sentence, so every model copied the context and scored ~100% —
# optimization had no signal to differentiate configs. This KB instead states
# the *rules* (e.g. "grid trials = product of candidate counts", which modes are
# on-device) and the dataset asks questions that require **applying** those rules
# (counting, multi-factor multiplication, multi-step reasoning). That spreads
# model accuracy (strong models ~0.96, mid ~0.82, small ~0.68 — measured live on
# Bedrock 2026-06-07), so the multi-objective accuracy/cost optimizer has real
# signal. Anchors (single-fact lookups) keep the floor non-zero.
KNOWLEDGE_BASE = [
    (
        "Traigent optimizes AI applications without code changes by wrapping a "
        "function with the @traigent.optimize decorator."
    ),
    (
        "Traigent supports four execution modes: local, cloud, hybrid, and "
        "edge_analytics. Local keeps all data on the user's own machine and sends "
        "nothing externally. Cloud runs everything on Traigent servers. Hybrid runs "
        "the LLM calls locally but uses the cloud for optimization intelligence. "
        "Edge_analytics runs optimization locally and sends only anonymized analytics."
    ),
    (
        "Traigent provides three optimization algorithms: grid search, random "
        "search, and Bayesian optimization. Grid search evaluates every combination "
        "in the configuration space. Random search samples a fixed number of random "
        "combinations. Bayesian optimization uses results from previous trials to "
        "choose the next combination and is the most sample-efficient of the three."
    ),
    (
        "A configuration space is a dictionary mapping each tunable parameter to its "
        "candidate values, expressed with Choices. For grid search, the number of "
        "trials equals the product of the number of candidate values across every "
        "parameter."
    ),
    (
        "Multi-objective optimization uses ObjectiveSchema. Each objective has a "
        "weight and the weights must sum to 1.0. Accuracy is maximized, while cost "
        "and latency are minimized."
    ),
    (
        "Seamless mode intercepts and overrides hardcoded LLM parameters "
        "automatically and requires no prompt changes. Context-injection mode "
        "instead modifies the prompt by adding the configuration to it."
    ),
    (
        "Traigent integrates with LangChain through adapters that intercept "
        "LangChain LLM calls and inject the optimized configuration."
    ),
    (
        "The eval_dataset parameter accepts a JSONL file of input and output pairs "
        "that is used to score each candidate configuration."
    ),
    (
        "Privacy: local mode and edge_analytics keep raw data on-device; the "
        "privacy_enabled flag prevents prompts and outputs from being transmitted "
        "when using hybrid mode."
    ),
    (
        "Traigent's dashboard visualizes the Pareto frontier of accuracy versus cost "
        "so users can compare configurations after a run completes."
    ),
    (
        "Traigent can resume an interrupted optimization run from its last completed "
        "trial."
    ),
    (
        "A measure in Traigent is a named numeric metric recorded for a "
        "configuration run."
    ),
]


class BedrockTitanLiteLLMEmbeddings(Embeddings):
    """Tiny LangChain-compatible wrapper for Bedrock Titan via LiteLLM."""

    def __init__(self, model: str, *, dimensions: int = 128) -> None:
        self.model = model
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if is_mock_llm():
            return [self._mock_embedding(text) for text in texts]

        response = litellm.embedding(model=self.model, input=texts)
        return [self._embedding_at(response, index) for index in range(len(texts))]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if is_mock_llm():
            return [self._mock_embedding(text) for text in texts]

        response = await litellm.aembedding(model=self.model, input=texts)
        return [self._embedding_at(response, index) for index in range(len(texts))]

    async def aembed_query(self, text: str) -> list[float]:
        return (await self.aembed_documents([text]))[0]

    def _mock_embedding(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in _tokens(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]

    @staticmethod
    def _embedding_at(response: Any, index: int) -> list[float]:
        data = _get_field(response, "data") or []
        item = data[index]
        embedding = _get_field(item, "embedding")
        if embedding is None:
            raise ValueError(
                "LiteLLM embedding response did not include data.embedding"
            )
        return [float(value) for value in embedding]


def _tokens(text: str) -> list[str]:
    return [
        token.strip(".,:;!?()[]{}\"'").lower()
        for token in text.split()
        if token.strip(".,:;!?()[]{}\"'")
    ]


def _get_field(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _response_text(response: Any) -> str:
    choices = _get_field(response, "choices") or []
    first_choice = choices[0]
    message = _get_field(first_choice, "message")
    content = _get_field(message, "content")
    return str(content or "")


def _usage_counts(response: Any) -> tuple[int, int]:
    usage = _get_field(response, "usage")
    prompt_tokens = _get_field(usage, "prompt_tokens") or _get_field(
        usage,
        "input_tokens",
    )
    completion_tokens = _get_field(usage, "completion_tokens") or _get_field(
        usage,
        "output_tokens",
    )
    return int(prompt_tokens or 0), int(completion_tokens or 0)


def _cost_usd(model_id: str, input_tokens: int, output_tokens: int) -> float:
    try:
        input_cost, output_cost = cost_from_tokens(
            input_tokens,
            output_tokens,
            model_id,
            strict=False,
        )
        return float(input_cost + output_cost)
    except Exception:
        return 0.0


def _answer_from_context(document: str) -> str:
    marker = "Answer:"
    if marker in document:
        return document.split(marker, 1)[1].strip()
    return document.strip()


def _trial_measure_count(results: Any) -> int:
    total = 0
    for trial in results.trials:
        measures = trial.metadata.get("measures")
        if measures:
            total += len(measures)
            continue

        for example_result in trial.metadata.get("example_results") or []:
            metrics = _get_field(example_result, "metrics")
            if metrics:
                total += 1
    return total


def _trial_example_result_count(results: Any) -> int:
    return sum(
        len(trial.metadata.get("example_results") or []) for trial in results.trials
    )


embeddings = BedrockTitanLiteLLMEmbeddings(DEFAULT_EMBEDDING_MODEL)
vector_store = FAISS.from_texts(KNOWLEDGE_BASE, embeddings)

retriever_agent = AgentDefinition(
    display_name="Bedrock Titan Retriever",
    parameter_keys=["k"],
    measure_ids=["retriever_latency_ms", "retrieved_docs"],
    order=1,
    agent_type="retriever",
)
generator_agent = AgentDefinition(
    display_name="Bedrock Generator",
    parameter_keys=["model", "temperature"],
    measure_ids=[
        "generator_latency_ms",
        "generator_input_tokens",
        "generator_output_tokens",
        "generator_cost_usd",
    ],
    primary_model="model",
    order=2,
    agent_type="llm",
)

traigent.initialize(execution_mode="edge_analytics")


@traigent.optimize(
    k=Choices([2], name="k", agent="retriever"),
    model=Choices(
        BEDROCK_GENERATOR_MODEL_IDS,
        name="model",
        agent="generator",
    ),
    temperature=Choices([0.0], name="temperature", agent="generator"),
    objectives=["accuracy", "cost"],
    scoring_function=semantic_overlap_score,
    eval_dataset=DATASET_PATH,
    agents={
        "retriever": retriever_agent,
        "generator": generator_agent,
    },
    execution_mode="edge_analytics",
)
async def bedrock_multi_agent_rag(question: str) -> str:
    """Answer a Traigent RAG question with retriever and generator spans."""
    config = traigent.get_config()
    k = int(config.get("k", 2))
    model_id = str(config.get("model", BEDROCK_GENERATOR_MODEL_IDS[0]))
    litellm_model = f"bedrock/{model_id}"
    temperature = float(config.get("temperature", 0.0))

    retrieve_start = time.perf_counter()
    docs = vector_store.similarity_search(question, k=k)
    retriever_latency_ms = (time.perf_counter() - retrieve_start) * 1000.0
    context = "\n\n".join(doc.page_content for doc in docs)

    add_agent_span(
        "retriever",
        span_type="agent",
        input_tokens=len(_tokens(question)),
        output_tokens=len(_tokens(context)),
        latency_ms=retriever_latency_ms,
        model=DEFAULT_EMBEDDING_MODEL,
        metadata={
            "k": k,
            "retrieved_docs": len(docs),
            "context_chars": len(context),
        },
    )

    messages = [
        {
            "role": "system",
            "content": (
                "Answer using only the supplied context. Return the answer "
                "sentence only, with no preamble."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:",
        },
    ]

    generator_start = time.perf_counter()
    response = await litellm.acompletion(
        model=litellm_model,
        messages=messages,
        temperature=temperature,
        max_tokens=96,
    )
    generator_latency_ms = (time.perf_counter() - generator_start) * 1000.0
    input_tokens, output_tokens = _usage_counts(response)
    cost_usd = _cost_usd(model_id, input_tokens, output_tokens)

    add_agent_span(
        "generator",
        span_type="agent",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        latency_ms=generator_latency_ms,
        model=litellm_model,
        metadata={
            "temperature_milli": int(temperature * 1000),
            "retrieved_docs": len(docs),
        },
    )

    if is_mock_llm() and docs:
        return _answer_from_context(docs[0].page_content)

    return _response_text(response)


async def main() -> None:
    total_trials = len(BEDROCK_GENERATOR_MODEL_IDS)
    print("Traigent Advanced Walkthrough: Bedrock Multi-Agent RAG")
    print("=" * 62)
    print("Execution mode: edge_analytics")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Embedding model: {DEFAULT_EMBEDDING_MODEL}")
    print("Generator models:")
    for model_id in BEDROCK_GENERATOR_MODEL_IDS:
        print(f"  - bedrock/{model_id}")
    example_count = sum(1 for _ in open(DATASET_PATH, encoding="utf-8") if _.strip())
    print(f"Grid: {total_trials} trials x {example_count} examples")

    results = await bedrock_multi_agent_rag.optimize(
        algorithm="grid",
        max_trials=total_trials,
        show_progress=True,
        random_seed=42,
        timeout=600,
    )

    successful_trials = sum(1 for trial in results.trials if trial.is_successful)
    example_result_count = _trial_example_result_count(results)
    measure_count = _trial_measure_count(results)

    print("\nRun Summary:")
    print(f"  trials: {len(results.trials)}")
    print(f"  successful_trials: {successful_trials}")
    print(f"  example_results: {example_result_count}")
    print(f"  measures: {measure_count}")

    print("\nBest Configuration:")
    print(f"  model: {results.best_config.get('model')}")
    print(f"  k: {results.best_config.get('k')}")
    print(f"  temperature: {results.best_config.get('temperature')}")

    print("\nBest Metrics:")
    print(f"  accuracy: {results.best_metrics.get('accuracy', 0.0):.2%}")
    print(f"  cost: ${results.best_metrics.get('cost', 0.0):.6f}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130) from None
