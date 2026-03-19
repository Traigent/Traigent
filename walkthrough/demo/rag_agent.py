#!/usr/bin/env python3
"""RAG QA agent - evaluate with configurable parameters.

Usage (run from repo root):
    export OPENAI_API_KEY="your-key"
    .venv/bin/python walkthrough/demo/rag_agent.py --model gpt-5.2 --temperature 0.0 --max-tokens 50 --instructions CoT
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from langchain_community.vectorstores import FAISS
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except Exception as exc:
    raise SystemExit(
        "Missing dependencies. Install with: pip install traigent[integrations]"
    ) from exc

from utils.scoring import semantic_overlap_score

DATASETS = Path(__file__).parent.parent / "datasets"

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
    global _vectorstore
    if _vectorstore is None:
        embeddings = OpenAIEmbeddings()
        _vectorstore = FAISS.from_texts(KNOWLEDGE_BASE, embeddings)
    return _vectorstore


def rag_agent(
    question: str, model: str, temperature: float, max_tokens: int, instructions: str
) -> str:
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n".join(doc.page_content for doc in docs)

    base_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    if instructions == "CoT":
        prompt = f"Think step by step before answering.\n{base_prompt}"
    else:
        prompt = base_prompt

    llm = ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens)
    try:
        response = llm.invoke(prompt)
        return str(response.content)
    except Exception as exc:
        return f"Error: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG agent with given config."
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--instructions", choices=["direct", "CoT"], default="direct")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running.")

    print("RAG Agent - Evaluation")
    print("=" * 50)
    print(f"Model:        {args.model}")
    print(f"Temperature:  {args.temperature}")
    print(f"Max tokens:   {args.max_tokens}")
    print(f"Instructions: {args.instructions}")
    print()

    examples: list[dict[str, object]] = []
    with open(DATASETS / "rag_questions.jsonl", encoding="utf-8") as handle:
        for line in handle:
            examples.append(json.loads(line.strip()))

    scores = []
    for i, example in enumerate(examples, 1):
        question = example["input"]["question"]
        expected = example["output"]
        start = time.time()
        output = rag_agent(
            question, args.model, args.temperature, args.max_tokens, args.instructions
        )
        latency = time.time() - start
        score = semantic_overlap_score(output, expected)
        scores.append(score)
        print(f"[{i:2d}/20] score={score:.2f}  ({latency:.2f}s)  Q: {question[:55]}")

    accuracy = sum(scores) / len(scores)
    print()
    print(f"Accuracy: {accuracy:.1%}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130)
