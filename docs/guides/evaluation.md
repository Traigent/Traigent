# TraiGent Evaluation Guide

## Overview

TraiGent evaluates your AI agent's performance by comparing actual outputs to expected results. This guide explains evaluation methods, dataset formats, custom evaluators, and troubleshooting.

## Table of Contents

- [Dataset Format](#dataset-format)
- [Evaluation Methods](#evaluation-methods)
- [Custom Evaluators](#custom-evaluators)
- [Mock Mode Testing](#mock-mode-testing)
- [Troubleshooting](#troubleshooting)

## Dataset Format

Evaluation datasets use JSONL (JSON Lines) format with the following structure:

```jsonl
{"input": {"question": "What is 2+2?"}, "expected_output": "4"}
{"input": {"question": "Capital of France?"}, "expected_output": "Paris"}
{"input": {"query": "Sentiment of: Great product!"}, "expected_output": "positive"}
```

### Key Fields

- **`input`**: Dictionary containing all input parameters for your function
- **`expected_output`**: The correct/desired output for comparison

### Example Dataset Creation

```python
import json

# Create evaluation dataset
examples = [
    {"input": {"question": "What is 2+2?"}, "expected_output": "4"},
    {"input": {"question": "Capital of France?"}, "expected_output": "Paris"},
    {"input": {"question": "Who wrote Hamlet?"}, "expected_output": "Shakespeare"}
]

# Save as JSONL
with open("qa_samples.jsonl", "w") as f:
    for example in examples:
        f.write(json.dumps(example) + "\n")
```

## Evaluation Methods

TraiGent supports three evaluation approaches:

### 1. Default Evaluation: Semantic Similarity

Uses embedding-based comparison to measure semantic similarity between outputs.

**Advantages:**
- Compares meaning, not exact text match
- Tolerates paraphrasing and formatting differences
- Works well for natural language outputs

**Requirements:**
- Requires `OPENAI_API_KEY` environment variable for embeddings
- Uses OpenAI's embedding model (default: `text-embedding-ada-002`)

**Example:**

```python
import traigent
from langchain_openai import ChatOpenAI

@traigent.optimize(
    eval_dataset="qa_samples.jsonl",
    objectives=["accuracy", "cost"]
)
def qa_agent(question: str) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    response = llm.invoke(f"Question: {question}\nAnswer:")
    return response.content

# Uses semantic similarity by default
```

### 2. Custom Evaluators

Define your own evaluation logic for domain-specific metrics.

**Simple Custom Evaluator:**

```python
def exact_match_evaluator(output: str, expected: str) -> float:
    """Return 1.0 for exact match, 0.0 otherwise"""
    return 1.0 if output.lower().strip() == expected.lower().strip() else 0.0

@traigent.optimize(
    evaluator=exact_match_evaluator,
    eval_dataset="data.jsonl"
)
def strict_agent(query: str) -> str:
    return process_query(query)
```

**Advanced Custom Evaluator:**

```python
from typing import Dict, Any

def sentiment_evaluator(output: str, expected: str, context: Dict[str, Any] = None) -> float:
    """
    Custom evaluator for sentiment analysis
    Returns score between 0.0 and 1.0
    """
    sentiment_map = {
        "positive": ["positive", "good", "great", "excellent"],
        "negative": ["negative", "bad", "poor", "terrible"],
        "neutral": ["neutral", "okay", "average"]
    }

    output_lower = output.lower().strip()
    expected_lower = expected.lower().strip()

    # Exact match
    if output_lower == expected_lower:
        return 1.0

    # Fuzzy match within sentiment category
    if expected_lower in sentiment_map:
        if any(word in output_lower for word in sentiment_map[expected_lower]):
            return 0.8

    return 0.0

@traigent.optimize(
    evaluator=sentiment_evaluator,
    eval_dataset="sentiment_data.jsonl",
    objectives=["accuracy", "cost"]
)
def sentiment_classifier(text: str) -> str:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    response = llm.invoke(f"Classify sentiment: {text}\nSentiment:")
    return response.content
```

**Multi-Metric Evaluator:**

```python
from traigent.evaluators.base import EvaluationResult

def comprehensive_evaluator(output: str, expected: str) -> EvaluationResult:
    """
    Returns multiple metrics for comprehensive evaluation
    """
    # Calculate various metrics
    exact_match = 1.0 if output == expected else 0.0

    # Calculate semantic similarity (example)
    similarity = calculate_similarity(output, expected)

    # Calculate length penalty
    length_diff = abs(len(output) - len(expected))
    length_score = max(0.0, 1.0 - (length_diff / 100))

    return EvaluationResult(
        accuracy=similarity,
        custom_metrics={
            "exact_match": exact_match,
            "length_score": length_score,
            "combined_score": (similarity + exact_match + length_score) / 3
        }
    )
```

### 3. Mock Mode for Testing

When `TRAIGENT_MOCK_MODE=true`, TraiGent simulates realistic optimization results without making real API calls.

**Features:**
- Shows realistic accuracy improvements (60-95%)
- Demonstrates optimization value without costs
- Perfect for demos, CI/CD, and initial testing
- Generates deterministic results for reproducibility

**Usage:**

```bash
# Enable mock mode
export TRAIGENT_MOCK_MODE=true

# Run optimization
python your_optimization_script.py
```

**Example Mock Results:**

```
Trial 1/10: accuracy=0.72, cost=$0.15
Trial 2/10: accuracy=0.81, cost=$0.12
Trial 3/10: accuracy=0.88, cost=$0.09
...
Best configuration found: accuracy=0.94, cost=$0.08
```

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
