#!/usr/bin/env python3
"""RAG QA agent — evaluate with configurable parameters.

Scene 1 (before Traigent):
    export OPENAI_API_KEY="your-key"
    python rag_agent.py --model gpt-5.2 --temperature 0.0 --max-tokens 50 --instructions CoT

Scene 4+5 (with Traigent) — add two lines marked ADD:
    import traigent                          # ADD
    ...
    @traigent.optimize(config_space=CONFIG, eval_dataset="questions.jsonl", objectives=OBJECTIVES)  # ADD
    def rag_agent(...):
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "walkthrough"))

try:
    from langchain_community.vectorstores import FAISS
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except Exception as exc:
    raise SystemExit(
        "Missing dependencies. Install with: pip install traigent[integrations]"
    ) from exc

from utils.scoring import semantic_overlap_score

DATASETS = Path(__file__).parent.parent / "walkthrough" / "datasets"

# --- Traigent config (used by Scene 4+5 decorator) ---
CONFIG = {
    "model":        ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "gpt-5.1", "gpt-5.2", "gpt-5-nano"],
    "temperature":  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "max_tokens":   [50, 100, 200],
    "instructions": ["direct", "CoT"],
}
OBJECTIVES = [
    {"name": "accuracy", "direction": "maximize", "weight": 0.5},
    {"name": "cost",     "direction": "minimize",  "weight": 0.2},
    {"name": "latency",  "direction": "minimize",  "weight": 0.3},
]
# --- end Traigent config ---

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
    "A dictionary defining the hyperparameters and their possible values to explore",
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


def rag_agent(question: str, model: str, temperature: float, max_tokens: int, instructions: str) -> str:
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n".join(d.page_content for d in docs)

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
    parser = argparse.ArgumentParser(description="Evaluate RAG agent with given config.")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--instructions", choices=["direct", "CoT"], default="direct")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running.")

    print("RAG Agent — Evaluation")
    print("=" * 50)
    print(f"Model:        {args.model}")
    print(f"Temperature:  {args.temperature}")
    print(f"Max tokens:   {args.max_tokens}")
    print(f"Instructions: {args.instructions}")
    print()

    examples = []
    with open(DATASETS / "rag_questions.jsonl") as f:
        for line in f:
            examples.append(json.loads(line.strip()))

    total = len(examples)
    scores = []
    total_cost = 0.0
    total_latency = 0.0

    COST_PER_OUTPUT_TOKEN = {
        "gpt-4o-mini": 0.60 / 1_000_000,
        "gpt-4o": 10.0 / 1_000_000,
        "gpt-3.5-turbo": 1.50 / 1_000_000,
    }
    cost_per_tok = COST_PER_OUTPUT_TOKEN.get(args.model, 2.0 / 1_000_000)

    for i, ex in enumerate(examples, 1):
        question = ex["input"]["question"]
        expected = ex["output"]
        t0 = time.time()
        output = rag_agent(question, args.model, args.temperature, args.max_tokens, args.instructions)
        latency = time.time() - t0
        score = semantic_overlap_score(output, expected)
        scores.append(score)

        out_tokens = len(output.split())
        total_cost += out_tokens * cost_per_tok
        total_latency += latency

        accuracy = sum(scores) / i
        filled = int(20 * i / total)
        bar = "█" * filled + "░" * (20 - filled)
        print(f"\rEvaluating: [{bar}] {i}/{total}  Accuracy: {accuracy:.1%}  Cost: ${total_cost:.4f}  Latency: {latency:.2f}s  ", end="", flush=True)

    avg_latency = total_latency / total
    print(f"\n\nAccuracy: {sum(scores) / total:.1%}   Cost: ${total_cost:.4f}   Avg latency: {avg_latency:.2f}s")


if __name__ == "__main__":
    main()
