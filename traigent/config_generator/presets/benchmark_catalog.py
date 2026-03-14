"""Known benchmark templates by agent type.

Each benchmark template provides a name, description, and example
schema that describes what evaluation examples should look like.
"""

from __future__ import annotations

from traigent.config_generator.types import BenchmarkSpec


def get_benchmark_for_agent_type(agent_type: str) -> BenchmarkSpec | None:
    """Return a benchmark template for the given agent type, or *None*."""
    return _BENCHMARK_CATALOG.get(agent_type)


def all_benchmark_types() -> frozenset[str]:
    """Return all agent types that have a benchmark template."""
    return frozenset(_BENCHMARK_CATALOG.keys())


_BENCHMARK_CATALOG: dict[str, BenchmarkSpec] = {
    "rag": BenchmarkSpec(
        name="RAG Question Answering",
        description=(
            "Evaluate retrieval-augmented generation: given a question and "
            "optional context, produce a grounded answer."
        ),
        example_schema={
            "input": {"question": "str", "context": "str (optional)"},
            "output": "str (expected answer)",
        },
        source="catalog",
    ),
    "chat": BenchmarkSpec(
        name="Conversational QA",
        description=(
            "Evaluate conversational ability: given a user message, "
            "produce a helpful, accurate response."
        ),
        example_schema={
            "input": {"message": "str"},
            "output": "str (expected response)",
        },
        source="catalog",
    ),
    "code_gen": BenchmarkSpec(
        name="Code Generation",
        description=(
            "Evaluate code generation: given a problem description, "
            "produce correct, executable code."
        ),
        example_schema={
            "input": {"prompt": "str", "language": "str (optional)"},
            "output": "str (expected code)",
        },
        source="catalog",
    ),
    "summarization": BenchmarkSpec(
        name="Document Summarization",
        description=(
            "Evaluate summarization quality: given a document, "
            "produce a faithful, concise summary."
        ),
        example_schema={
            "input": {"document": "str", "max_length": "int (optional)"},
            "output": "str (expected summary)",
        },
        source="catalog",
    ),
    "classification": BenchmarkSpec(
        name="Text Classification",
        description=(
            "Evaluate classification accuracy: given text, "
            "predict the correct label."
        ),
        example_schema={
            "input": {"text": "str"},
            "output": "str (expected label)",
        },
        source="catalog",
    ),
    "general_llm": BenchmarkSpec(
        name="General LLM Evaluation",
        description=(
            "General-purpose evaluation: given an input prompt, "
            "produce a correct output."
        ),
        example_schema={
            "input": {"prompt": "str"},
            "output": "str (expected output)",
        },
        source="catalog",
    ),
}
