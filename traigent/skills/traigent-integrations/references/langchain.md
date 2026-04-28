# LangChain Integration Reference

## Overview

Traigent integrates with LangChain to optimize model selection, temperature, and other parameters across chains, agents, and RAG pipelines. There are two approaches:

1. **Manual config injection** - call `traigent.get_config()` and construct LangChain objects yourself
2. **Auto override** - let Traigent intercept LangChain model instantiation via `auto_override_frameworks` or `framework_targets`

## Installation

```bash
pip install traigent[integrations]

# Plus the LangChain provider packages you need
pip install langchain-openai langchain-anthropic langchain-google-genai
```

## auto_override_frameworks

When `auto_override_frameworks=True`, Traigent automatically discovers and overrides LLM client constructors so that the model and parameters from the current trial's configuration are injected:

```python
import traigent
from langchain_openai import ChatOpenAI

@traigent.optimize(
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4o"],
        "temperature": [0.0, 0.5, 1.0],
    },
    objectives=["accuracy"],
    max_trials=6,
    auto_override_frameworks=True,
)
def my_chain(text):
    # These arguments are overridden by Traigent during optimization
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    return llm.invoke(text).content
```

During optimization, each trial replaces the `model` and `temperature` arguments with the trial's configuration values. After optimization, calling `apply_best_config()` locks in the best configuration.

### How It Works

1. Traigent scans for known framework classes (OpenAI, Anthropic, LangChain wrappers)
2. During a trial, `__init__` calls to those classes are intercepted
3. Configuration parameters (`model`, `temperature`, etc.) are replaced with the trial values
4. After the trial, the override is removed

### Supported Auto-Discovery Targets

- `langchain_openai.ChatOpenAI`
- `langchain_anthropic.ChatAnthropic`
- `langchain_google_genai.ChatGoogleGenerativeAI`
- `openai.OpenAI`, `openai.AsyncOpenAI`
- `anthropic.Anthropic`, `anthropic.AsyncAnthropic`

## framework_targets

For finer control, specify exactly which classes to override:

```python
@traigent.optimize(
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4o"],
        "temperature": [0.0, 0.5],
    },
    objectives=["accuracy"],
    max_trials=8,
    framework_targets=["langchain_openai.ChatOpenAI"],
)
def my_func(text):
    # Only ChatOpenAI is overridden, other LLM classes are untouched
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    return llm.invoke(text).content
```

This is useful when your function uses multiple LLM clients and you only want to optimize one of them.

## Manual Config Injection (Recommended for Complex Chains)

For chains with multiple components, manual injection gives you full control:

```python
import traigent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

@traigent.optimize(
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4o"],
        "temperature": [0.0, 0.3, 0.7],
        "system_prompt_style": ["concise", "detailed", "academic"],
    },
    objectives=["accuracy"],
    max_trials=12,
)
def qa_chain(question):
    config = traigent.get_config()

    system_prompts = {
        "concise": "Answer briefly and accurately.",
        "detailed": "Provide a thorough, well-structured answer.",
        "academic": "Answer with academic rigor, citing reasoning.",
    }

    llm = ChatOpenAI(
        model=config["model"],
        temperature=config["temperature"],
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompts[config["system_prompt_style"]]),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question})
```

## RAG Chain Optimization

Optimize retrieval-augmented generation by including retrieval parameters in the config space:

```python
import traigent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

@traigent.optimize(
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4o"],
        "temperature": [0.0, 0.3],
        "top_k": [3, 5, 10],
        "prompt_template": [
            "Answer based on the context:\n{context}\n\nQuestion: {question}",
            "Context: {context}\n\nUsing only the above, answer: {question}",
        ],
    },
    objectives=["accuracy"],
    max_trials=12,
)
def rag_answer(question):
    config = traigent.get_config()

    # Retriever with optimized top_k
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local("my_index", embeddings)
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": config["top_k"]}
    )

    # LLM with optimized model and temperature
    llm = ChatOpenAI(
        model=config["model"],
        temperature=config["temperature"],
    )

    # Chain with optimized prompt
    prompt = ChatPromptTemplate.from_template(config["prompt_template"])
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke(question)

results = rag_answer.optimize(dataset="rag_eval.jsonl")
```

## Programmatic Framework Override API

For advanced use cases, you can manage overrides programmatically:

```python
from traigent.integrations import enable_framework_overrides, disable_framework_overrides

# Enable overrides for specific targets
enable_framework_overrides(["langchain_openai.ChatOpenAI"])

# ... run your code ...

# Disable when done
disable_framework_overrides()
```

## Tips

- Use manual config injection for production code where you want explicit control
- Use `auto_override_frameworks` for quick experimentation and prototyping
- When optimizing RAG, include retrieval parameters (top_k, chunk_size) alongside model parameters
- LangChain's streaming and async interfaces work normally with Traigent config injection
- Always construct LangChain objects inside the decorated function, not at module level
