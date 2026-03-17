#!/usr/bin/env python3
"""Example: Multi-Agent Optimization - Real LangChain Version

Demonstrates a real RAG pipeline with retriever and generator agents.

Requirements:
- Set OPENAI_API_KEY environment variable
- pip install langchain langchain-openai faiss-cpu

Run with: python 03_multi_agent.py
"""

import asyncio
import os
from pathlib import Path

import traigent
from traigent.api.parameter_ranges import Choices, IntRange, Range
from traigent.api.types import AgentDefinition

# Compute dataset path relative to this script
SCRIPT_DIR = Path(__file__).parent
WALKTHROUGH_ROOT = (SCRIPT_DIR / ".." / "..").resolve()
DATASET_PATH = str((WALKTHROUGH_ROOT / "datasets" / "simple_questions.jsonl").resolve())

# Allow dataset sandbox to find files under the walkthrough directory
os.environ.setdefault("TRAIGENT_DATASET_ROOT", str(WALKTHROUGH_ROOT))

# Check for API key before initialization
if not os.getenv("OPENAI_API_KEY"):
    print("Error: Set OPENAI_API_KEY environment variable to run this example")
    print("Example: export OPENAI_API_KEY='your-key-here'")  # pragma: allowlist secret
    exit(1)

try:
    from langchain_community.vectorstores import FAISS
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ImportError:
    print("Error: Required packages not installed")
    print(
        "Install with: pip install langchain-core langchain-openai langchain-community faiss-cpu"
    )
    exit(1)

# Initialize Traigent
traigent.initialize(execution_mode="edge_analytics")

# Sample documents for the vector store
SAMPLE_DOCS = [
    "The capital of France is Paris.",
    "Python is a popular programming language for machine learning.",
    "Machine learning is a subset of artificial intelligence.",
    "LangChain is a framework for building LLM applications.",
    "The Eiffel Tower is located in Paris, France.",
]

# Create vector store (simplified for example)
print("Initializing vector store...")
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_texts(SAMPLE_DOCS, embeddings)

# Define agents
retriever_agent = AgentDefinition(
    display_name="Retriever Agent",
    parameter_keys=["k", "search_type"],
    measure_ids=["retrieval_latency"],
)
generator_agent = AgentDefinition(
    display_name="Generator Agent",
    parameter_keys=["temperature", "model"],
    measure_ids=["generation_latency", "generation_cost"],
)


@traigent.optimize(
    # Retriever parameters
    k=IntRange(1, 5, default=2, name="k"),
    search_type=Choices(["similarity", "mmr"], name="search_type"),
    # Generator parameters
    temperature=Range.temperature(),
    model=Choices(["gpt-4o-mini", "gpt-4o"], name="model"),
    # Agent configuration
    agents={
        "retriever": retriever_agent,
        "generator": generator_agent,
    },
    objectives=["accuracy", "latency"],
    eval_dataset=DATASET_PATH,
    execution_mode="edge_analytics",
)
async def rag_agent(question: str) -> str:
    """Real RAG agent with LangChain."""
    config = traigent.get_config()

    # Retrieval phase
    k = config["k"]
    search_type = config["search_type"]

    if search_type == "mmr":
        docs = vector_store.max_marginal_relevance_search(question, k=k)
    else:
        docs = vector_store.similarity_search(question, k=k)

    context = "\n".join(doc.page_content for doc in docs)

    # Generation phase
    llm = ChatOpenAI(
        model=config["model"],
        temperature=config["temperature"],
        max_tokens=256,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer based on the context:\n{context}"),
            ("user", "{question}"),
        ]
    )

    chain = prompt | llm
    response = await chain.ainvoke({"context": context, "question": question})

    return response.content


async def main() -> None:
    print("Traigent Example: Multi-Agent RAG with LangChain")
    print("=" * 60)

    print("\nAgent Definitions:")
    print(f"  Retriever: params={retriever_agent.parameter_keys}")
    print(f"  Generator: params={generator_agent.parameter_keys}")

    print("\nRunning optimization (this will make real API calls)...")
    results = await rag_agent.optimize(algorithm="random", max_trials=6, random_seed=42)

    print("\nBest Configuration:")
    print(f"  Retriever k: {results.best_config.get('k')}")
    print(f"  Search type: {results.best_config.get('search_type')}")
    print(f"  Temperature: {results.best_config.get('temperature')}")
    print(f"  Model: {results.best_config.get('model')}")

    print("\nPerformance:")
    print(f"  Accuracy: {results.best_metrics.get('accuracy', 0):.2%}")
    print(f"  Latency: {results.best_metrics.get('latency', 0):.3f}s")
    print(f"  Cost: ${results.best_metrics.get('cost', 0):.6f}")


if __name__ == "__main__":
    asyncio.run(main())
