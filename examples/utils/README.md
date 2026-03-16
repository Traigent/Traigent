# Examples Utilities

Shared helper modules used across Traigent examples. These utilities reduce code duplication and ensure consistent behavior.

## What's Here

| File | Purpose |
|------|---------|
| `setup.py` | Common setup for all examples (mock mode, imports, initialization) |
| `base_example.py` | Base class with temp file management and sample datasets |
| `mock_langchain.py` | Full LangChain mock classes (ChatAnthropic, ChatOpenAI, messages, retrievers) |
| `langchain_compat.py` | Lightweight LangChain compatibility layer |

## Quick Start

Most examples use `setup.py` for initialization:

```python
from pathlib import Path
import sys

# Add utils to path
sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "utils"))

from setup import quick_setup, get_example_dataset

# Initialize everything (mock mode, paths, traigent)
is_mock = quick_setup(Path(__file__).parent)

# Get dataset path
dataset_path = get_example_dataset("rag-optimization")
```

## Module Reference

### setup.py

Main setup utilities:

- `is_mock_mode()` - Returns `True` if `TRAIGENT_MOCK_LLM=true`
- `setup_mock_environment(base_path)` - Sets HOME and results folder for sandboxed execution
- `setup_traigent_import()` - Adds traigent to sys.path if not installed
- `initialize_traigent(mode)` - Initializes traigent with execution mode
- `quick_setup(base_path, mode)` - All-in-one setup function
- `get_example_dataset(name, filename)` - Returns path to a dataset file

### base_example.py

Base class for structured examples:

```python
from base_example import BaseExample, display_optimization_results

class MyExample(BaseExample):
    def get_config(self) -> dict:
        return {"objective": "accuracy", ...}

# Display results consistently
display_optimization_results(result)
```

Sample datasets included:
- `SENTIMENT_DATASET` - 5 sentiment analysis samples
- `QA_DATASET` - 3 Q&A samples
- `CLASSIFICATION_DATASET` - 4 spam classification samples

### mock_langchain.py

Full LangChain mocks for offline testing:

```python
from mock_langchain import ChatOpenAI, HumanMessage, SystemMessage

# Works without LangChain installed
llm = ChatOpenAI(model="gpt-4", temperature=0.7)
response = llm.invoke([HumanMessage(content="Hello")])
```

Available classes:
- `ChatAnthropic`, `ChatOpenAI` - LLM clients with `invoke()` and `ainvoke()`
- `HumanMessage`, `SystemMessage`, `AIMessage` - Message types
- `Document` - Document with `page_content` and `metadata`
- `BM25Retriever` - Retriever with `from_documents()` and `get_relevant_documents()`

### langchain_compat.py

Simpler compatibility layer:

```python
from langchain_compat import ChatOpenAI, HumanMessage, extract_content

llm = ChatOpenAI(model="gpt-4")
response = llm.invoke([HumanMessage(content="Hi")])
text = extract_content(response)  # Get content as string
```

## When to Use Which

| Scenario | Use |
|----------|-----|
| Starting a new example | `setup.py` → `quick_setup()` |
| Need LangChain classes without the dependency | `mock_langchain.py` (full) or `langchain_compat.py` (minimal) |
| Building a structured example class | `base_example.py` → `BaseExample` |
| Just need sample data | `base_example.py` → `SENTIMENT_DATASET`, etc. |

## Environment Variables

| Variable | Effect |
|----------|--------|
| `TRAIGENT_MOCK_LLM=true` | Enables mock mode (no API calls) |
| `TRAIGENT_RESULTS_FOLDER` | Override results directory |
| `TRAIGENT_COST_APPROVED=true` | Auto-approve cost prompts (set by default) |
