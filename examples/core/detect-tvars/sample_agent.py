"""Sample agent with tunable variables for detection demo.

This file contains realistic LLM agent patterns that the Traigent
detection engine can analyze. It is intentionally NOT executable —
it exists purely as static analysis input.

Variables are intentionally assigned but unused (the LLM calls that
would consume them are commented out) so the file runs without
real SDK dependencies.
"""


def answer_question(query: str, context: str) -> str:  # noqa: ARG001
    """RAG-style agent with multiple tunable parameters."""
    # LLM configuration
    temperature = 0.7  # noqa: F841
    model_name = "gpt-4o"  # noqa: F841
    max_tokens = 1024  # noqa: F841

    # Retrieval configuration
    k = 5  # noqa: F841
    chunk_size = 512  # noqa: F841

    # These would normally use real SDK clients:
    # llm = ChatOpenAI(temperature=temperature, model=model_name, max_tokens=max_tokens)
    # docs = vector_store.similarity_search(query, k=k)
    # return llm.invoke(formatted_prompt)
    return f"answer to {query}"


def summarize_document(text: str) -> str:
    """Summarizer showing constructor-tracing pattern.

    The DataFlowDetectionStrategy traces `t` and `n` backward through
    the ChatOpenAI constructor to the invoke() sink — even though
    the variable names are not recognizable LLM parameter names.
    """
    t = 0.3  # not named "temperature"  # noqa: F841
    n = 2048  # not named "max_tokens"  # noqa: F841

    # llm = ChatOpenAI(temperature=t, max_tokens=n)
    # return llm.invoke(f"Summarize: {text}")
    return f"summary of {text}"


def multi_step_agent(query: str) -> str:
    """Agent with transitive variable chains.

    Shows how DataFlow traces through intermediate assignments.
    """
    base_temp = 0.5  # noqa: F841
    adjusted_temp = base_temp  # 1 hop  # noqa: F841
    final_temp = adjusted_temp  # 2 hops from base_temp  # noqa: F841

    model = "gpt-4o-mini"  # noqa: F841
    prompt = f"Answer the question: {query}"  # noqa: F841

    # llm.invoke(prompt, temperature=final_temp, model=model)
    return f"answer to {query}"
