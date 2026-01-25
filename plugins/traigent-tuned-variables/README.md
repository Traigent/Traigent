# Traigent Tuned Variables Plugin

Domain-aware tuned variables and variable analysis for the Traigent SDK.

## Features

- **Domain Presets**: Pre-configured parameter ranges for LLM, RAG, and prompting
- **Variable Analyzer**: Centralized post-optimization analysis with elimination suggestions
- **TunedCallable**: Composition pattern for function-valued variables
- **DSPy Integration**: Optional DSPy prompt optimization adapter

## Installation

```bash
pip install traigent-tuned-variables

# With DSPy support
pip install traigent-tuned-variables[dspy]
```

## Quick Start

### Domain Presets

```python
import traigent
from traigent_tuned_variables.presets import LLMPresets, RAGPresets

@traigent.optimize(
    temperature=LLMPresets.temperature(creative=True),
    max_tokens=LLMPresets.max_tokens(task="medium"),
    model=LLMPresets.model(provider="openai", tier="balanced"),
    k=RAGPresets.k_retrieval(),
    objectives=["accuracy", "cost"],
)
def my_agent(query: str) -> str:
    ...
```

### Variable Analysis

```python
from traigent_tuned_variables.analysis import VariableAnalyzer

result = my_agent.optimize()
analysis = VariableAnalyzer(result).analyze("accuracy")

# Get elimination suggestions
for suggestion in analysis.elimination_suggestions:
    print(f"{suggestion.variable}: {suggestion.action} - {suggestion.reason}")

# Get refined configuration space for next optimization
refined_space = analysis.refined_space
```

### TunedCallable

```python
from traigent_tuned_variables.callables import Retrievers, ContextFormatters

@traigent.optimize(
    retriever=Retrievers.as_choices(),
    context_format=ContextFormatters.as_choices(),
    objectives=["accuracy"],
)
def rag_agent(query: str) -> str:
    config = traigent.get_config()
    docs = Retrievers.invoke(config["retriever"], vector_store, query)
    formatted = ContextFormatters.invoke(config["context_format"], docs)
    ...
```

## API Reference

### Presets

- `LLMPresets.temperature(conservative=False, creative=False)` - Temperature range
- `LLMPresets.top_p()` - Nucleus sampling parameter
- `LLMPresets.max_tokens(task="short"|"medium"|"long")` - Token limit
- `LLMPresets.model(provider=None, tier="balanced")` - Model selection
- `RAGPresets.k_retrieval(max_k=10)` - Retrieval depth
- `RAGPresets.chunk_size()` - Document chunk size
- `PromptingPresets.strategy()` - Prompting strategy selection
- `PromptingPresets.context_format()` - Context formatting options

### Analysis

- `VariableAnalyzer(result, **kwargs)` - Create analyzer from OptimizationResult
- `analyzer.analyze(objective)` - Run full analysis
- `analyzer.get_variable_importance(objective)` - Get importance scores
- `analyzer.get_dominated_values(variable, objectives)` - Find dominated values
- `analyzer.get_refined_space(objectives)` - Get pruned configuration space

### Callables

- `Retrievers.as_choices()` - Get retriever selection as Choices
- `Retrievers.invoke(name, *args, **kwargs)` - Invoke retriever by name
- `ContextFormatters.as_choices()` - Get formatter selection as Choices
- `ContextFormatters.invoke(name, docs)` - Invoke formatter by name

## License

MIT License - see LICENSE for details.
