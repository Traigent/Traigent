# Traigent Evaluation Guide

How Traigent scores your runs and how to customize it.

## Dataset Format

Use JSONL with `input` and optional `output`/`expected_output` keys (both are accepted):

```jsonl
{"input": {"question": "What is 2+2?"}, "output": "4"}
{"input": {"question": "Capital of France?"}, "output": "Paris"}
{"input": {"query": "Sentiment of: Great product!"}, "output": "positive"}
```

Minimal creator:

```python
import json

data = [
    {"input": {"question": "What is 2+2?"}, "output": "4"},
    {"input": {"question": "Capital of France?"}, "output": "Paris"},
]
with open("qa_samples.jsonl", "w") as f:
    for row in data:
        f.write(json.dumps(row) + "\n")
```

## Evaluation Modes

### 1) Default semantic similarity

- Embedding-based comparison (OpenAI embeddings by default); set `OPENAI_API_KEY` unless running in `TRAIGENT_MOCK_MODE`.
- Great for natural language tasks where paraphrasing is fine.

```python
import traigent
from langchain_openai import ChatOpenAI

@traigent.optimize(
    eval_dataset="qa_samples.jsonl",
    objectives=["accuracy", "cost"],
)
def qa_agent(question: str) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    return llm.invoke(f"Question: {question}\nAnswer:").content
```

### 2) Custom evaluators

Use your own scoring function when exact rules or domain metrics matter.

```python
def exact_match(output: str, expected: str) -> float:
    return 1.0 if output.strip().lower() == expected.strip().lower() else 0.0

@traigent.optimize(
    scoring_function=exact_match,
    eval_dataset="data.jsonl",
    objectives=["accuracy"],
)
def strict_agent(query: str) -> str:
    ...
```

Multi-metric example:

```python
from traigent.evaluators.base import EvaluationResult

def combined_eval(output: str, expected: str) -> EvaluationResult:
    exact = 1.0 if output == expected else 0.0
    length_penalty = max(0.0, 1.0 - abs(len(output) - len(expected)) / 100)
    return EvaluationResult(
        accuracy=exact,
        custom_metrics={"exact_match": exact, "length_score": length_penalty},
    )
```

### 3) Mock mode

`TRAIGENT_MOCK_MODE=true` skips external LLM/API calls and synthesizes metrics—ideal for CI, demos, and budget-safe smoke tests.

```bash
export TRAIGENT_MOCK_MODE=true
python your_optimization_script.py
```

## Troubleshooting

- **Semantic similarity fails**: Confirm `OPENAI_API_KEY` is set or switch to a custom evaluator.
- **Scores all zeros**: Check that dataset `output`/`expected_output` values are non-empty strings.
- **Dataset errors**: Run `traigent validate path/to/dataset.jsonl` to see the exact row/field causing issues.

## Custom Evaluators - Advanced Patterns

### Pattern 1: RAG Quality Evaluator

```python
from traigent.metrics.rag import (
    calculate_relevance,
    calculate_faithfulness,
    calculate_context_precision
)

def rag_evaluator(output: str, expected: str, context: Dict[str, Any]) -> float:
    """Evaluate RAG system quality"""
    retrieved_docs = context.get("retrieved_docs", [])

    # Calculate RAG-specific metrics
    relevance = calculate_relevance(output, retrieved_docs)
    faithfulness = calculate_faithfulness(output, retrieved_docs)
    answer_similarity = semantic_similarity(output, expected)

    # Weighted combination
    score = (
        0.4 * answer_similarity +
        0.3 * relevance +
        0.3 * faithfulness
    )

    return score
```

### Pattern 2: Classification Evaluator

```python
from sklearn.metrics import f1_score, precision_score, recall_score

def classification_evaluator(output: str, expected: str) -> Dict[str, float]:
    """Multi-class classification evaluator"""
    # Parse outputs
    predicted_class = extract_class(output)
    true_class = extract_class(expected)

    # Store for batch metrics
    predictions.append(predicted_class)
    true_labels.append(true_class)

    # Calculate instance metrics
    correct = 1.0 if predicted_class == true_class else 0.0

    return {
        "accuracy": correct,
        "precision": precision_score([true_class], [predicted_class], average='weighted'),
        "recall": recall_score([true_class], [predicted_class], average='weighted'),
        "f1": f1_score([true_class], [predicted_class], average='weighted')
    }
```

### Pattern 3: Code Generation Evaluator

```python
import ast
import subprocess

def code_evaluator(output: str, expected: str, context: Dict[str, Any]) -> float:
    """Evaluate generated code quality"""
    test_cases = context.get("test_cases", [])

    # Check syntax validity
    try:
        ast.parse(output)
        syntax_valid = 1.0
    except SyntaxError:
        return 0.0  # Invalid syntax = fail

    # Run test cases
    passed_tests = 0
    for test_input, test_output in test_cases:
        try:
            result = execute_code(output, test_input)
            if result == test_output:
                passed_tests += 1
        except Exception:
            continue

    test_score = passed_tests / len(test_cases) if test_cases else 0.0

    # Compare with expected solution
    similarity = code_similarity(output, expected)

    return 0.5 * test_score + 0.3 * similarity + 0.2 * syntax_valid
```

## Troubleshooting

### Issue: 0.0% Accuracy

**Symptoms:**
```
Trial 1/10: accuracy=0.00, cost=$0.15
Trial 2/10: accuracy=0.00, cost=$0.12
...
Best accuracy: 0.00
```

**Solutions:**

1. **Verify Dataset Format**
   ```python
   import json

   # Check first line of dataset
   with open("qa_samples.jsonl", "r") as f:
       first_line = f.readline()
       data = json.loads(first_line)
       print(f"Keys: {data.keys()}")
       print(f"Input: {data.get('input')}")
       print(f"Expected: {data.get('expected_output')}")
   ```

2. **Check API Keys**
   ```bash
   # Verify OpenAI API key for embeddings
   echo $OPENAI_API_KEY

   # Or use custom evaluator if no API key
   ```

3. **Use Custom Evaluator**
   ```python
   def debug_evaluator(output: str, expected: str) -> float:
       print(f"Output: '{output}'")
       print(f"Expected: '{expected}'")
       score = 1.0 if output.lower() == expected.lower() else 0.0
       print(f"Score: {score}")
       return score
   ```

4. **Enable Debug Mode**
   ```python
   @traigent.optimize(
       debug_evaluation=True,  # Print detailed eval info
       eval_dataset="data.jsonl"
   )
   ```

### Issue: Inconsistent Accuracy

**Symptoms:** Accuracy varies wildly between runs (20% one run, 80% next run)

**Solutions:**

1. **Fix Random Seeds**
   ```python
   import random
   import numpy as np

   random.seed(42)
   np.random.seed(42)

   @traigent.optimize(
       random_seed=42,  # Deterministic optimization
       eval_dataset="data.jsonl"
   )
   ```

2. **Increase Sample Size**
   ```python
   # Use more evaluation examples
   # Minimum recommended: 50-100 examples
   ```

3. **Use Temperature=0 for Deterministic Output**
   ```python
   configuration_space={
       "model": ["gpt-4o-mini", "gpt-4o"],
       "temperature": [0.0]  # Deterministic
   }
   ```

### Issue: Evaluation Too Slow

**Solutions:**

1. **Use Smaller Dataset**
   ```python
   from traigent.evaluators.base import Dataset

   # Sample subset for faster evaluation
   full_dataset = Dataset.from_jsonl("large_dataset.jsonl")
   sample_dataset = full_dataset.sample(n=100, random_state=42)
   ```

2. **Enable Parallel Evaluation**
   ```python
   @traigent.optimize(
       parallel_config={
           "example_concurrency": 8,  # Evaluate 8 examples in parallel
           "trial_concurrency": 4     # Run 4 trials in parallel
       }
   )
   ```

3. **Use Mock Mode for Testing**
   ```bash
   export TRAIGENT_MOCK_MODE=true
   ```

### Issue: Out of Memory

**Solutions:**

1. **Use a Smaller Dataset Subset**
   ```python
   @traigent.optimize(
       eval_dataset="large_dataset_subset.jsonl"  # Use a representative subset
   )
   ```

2. **Use Streaming Evaluation**
   ```python
   from traigent.evaluators.base import StreamingDataset

   dataset = StreamingDataset.from_jsonl("large_dataset.jsonl")
   ```

## Best Practices

### 1. Dataset Quality

- **Size**: Minimum 50-100 examples, ideal 500-1000
- **Coverage**: Cover all edge cases and common scenarios
- **Balance**: Ensure balanced distribution across categories
- **Quality**: Manually review and validate expected outputs

### 2. Evaluator Selection

- **Default (Semantic)**: Best for natural language generation
- **Exact Match**: Best for structured output (JSON, classifications)
- **Custom**: Required for domain-specific metrics

### 3. Optimization Settings

```python
@traigent.optimize(
    eval_dataset="validated_dataset.jsonl",
    objectives=["accuracy", "cost"],

    # Use appropriate evaluator
    evaluator=custom_evaluator,  # if needed

    # Enable debugging during development
    debug_evaluation=True,

    # Set reasonable limits
    max_trials=50,
    timeout_per_trial=60,  # seconds

    # Enable parallel processing
    parallel_config={
        "example_concurrency": 4,
        "trial_concurrency": 2
    }
)
```

### 4. Iteration Workflow

1. **Start Small**: Test with 10-20 examples in mock mode
2. **Validate**: Ensure evaluator works correctly
3. **Scale Up**: Increase to 100+ examples
4. **Optimize**: Run full optimization with real API calls
5. **Verify**: Manually review best configuration results

## Related Documentation

- [Quick Start Guide](quickstart.md)
- [Execution Modes](execution-modes.md)
- [API Reference](../api-reference/)
- [Examples](../../examples/)

---

**Need Help?**
- [GitHub Issues](https://github.com/Traigent/Traigent/issues)
- [Discord Community](https://discord.gg/traigent)
- [Documentation](https://docs.traigent.ai)
