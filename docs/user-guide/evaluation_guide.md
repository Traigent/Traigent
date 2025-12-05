# 📊 TraiGent Evaluation Guide

## Understanding Evaluation in TraiGent

Evaluation is how TraiGent measures the success of different AI agent configurations. This guide explains what evaluation means, how it works, and how to use it effectively.

## Table of Contents
- [What is Evaluation?](#what-is-evaluation)
- [Understanding Accuracy](#understanding-accuracy)
- [Evaluation Methods](#evaluation-methods)
- [Creating Evaluation Datasets](#creating-evaluation-datasets)
- [Custom Evaluators](#custom-evaluators)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)

## What is Evaluation?

Evaluation in TraiGent compares your AI agent's outputs against expected results to determine which configuration performs best.

```python
@traigent.optimize(
    configuration_space={"temperature": [0.1, 0.5, 0.9]},
    objectives=["accuracy"],  # What we're measuring
    eval_dataset="data.jsonl"  # Test cases with expected outputs
)
def my_agent(input):
    # Your AI agent logic
    return output
```

## Understanding Accuracy

"Accuracy" in TraiGent can mean different things depending on your use case:

### 1. Classification Accuracy (Exact Match)
For tasks with discrete outputs (sentiment, categories, yes/no):
```python
# Accuracy = % of exact matches
Input: "I love this!"  → Output: "positive" → Expected: "positive" → ✅ 100%
Input: "I hate this!"  → Output: "neutral"  → Expected: "negative" → ❌ 0%
```

### 2. Semantic Similarity
For tasks with varied but equivalent outputs:
```python
# Accuracy = semantic similarity score
Input: "Capital of France?"
Output: "Paris is the capital"
Expected: "Paris"
→ 95% (semantically equivalent)
```

### 3. Custom Metrics
For domain-specific evaluation:
```python
# Accuracy = custom calculation
Input: Financial report
Output: Generated summary
Expected: Reference summary
→ ROUGE score, BLEU score, or business metrics
```

## Evaluation Methods

### Method 1: Default Evaluation (Semantic Similarity)

TraiGent's default evaluator uses embeddings to compare meaning:

```python
@traigent.optimize(
    eval_dataset="qa_pairs.jsonl",  # Uses default semantic evaluation
    objectives=["accuracy"]
)
def qa_agent(question):
    return answer
```

**Requirements:**
- `OPENAI_API_KEY` for embedding generation
- Works well for: Q&A, summarization, translation

**Dataset format (JSONL):**
```json
{"input": {"question": "What is AI?"}, "output": "Artificial Intelligence"}
{"input": {"question": "Who invented Python?"}, "output": "Guido van Rossum"}
```

### Method 2: Exact Match Evaluation

For classification tasks requiring exact matches:

```python
def exact_match_score(output, expected, llm_metrics=None):
    """Returns 1.0 for exact match, 0.0 otherwise."""
    return 1.0 if str(output).strip() == str(expected).strip() else 0.0

@traigent.optimize(
    eval_dataset="classification.jsonl",
    scoring_function=exact_match_score,
    objectives=["accuracy"]
)
def classifier(text):
    return category
```

### Method 3: Mock Mode (Development/Demo)

For testing without real API calls:

```bash
export TRAIGENT_MOCK_MODE=true
python your_script.py
```

Mock mode provides:
- Realistic accuracy values (60-95%)
- Progressive improvement simulation
- No API costs
- Fast iteration

Tips:
- To keep all artifacts local and writable in restricted environments, set `TRAIGENT_RESULTS_FOLDER` to a project path:
  - `export TRAIGENT_RESULTS_FOLDER="./.traigent_local"`
- The Examples Navigator pages work from `file://`, but some browsers block `fetch()` on local files. Serve via HTTP to avoid CORS:
  - `python -m http.server -d examples 8000` → open http://localhost:8000/
  - On “Getting Started”, a built-in inline fallback shows demo results even on file://

## Creating Evaluation Datasets

### Format: JSONL (JSON Lines)

Each line is a complete JSON object:

```json
{"input": {"text": "Great product!"}, "output": "positive"}
{"input": {"text": "Terrible service"}, "output": "negative"}
{"input": {"text": "It's okay"}, "output": "neutral"}
```

### Best Practices

1. **Representative Data**: Include examples from real use cases
2. **Balanced Classes**: Equal distribution across categories
3. **Edge Cases**: Include difficult or ambiguous examples
4. **Size**: 20-100 examples for initial testing, 100+ for production

### Creating Datasets Programmatically

```python
import json

# Create evaluation dataset
dataset = []

# Add examples
dataset.append({
    "input": {"question": "What is 2+2?"},
    "output": "4"
})

dataset.append({
    "input": {"question": "Capital of Japan?"},
    "output": "Tokyo"
})

# Save to JSONL
with open("eval_dataset.jsonl", "w") as f:
    for item in dataset:
        f.write(json.dumps(item) + "\n")
```

## Custom Evaluators

### When to Use Custom Evaluators

Use custom evaluators when:
- You have specific business metrics
- Default similarity doesn't fit your use case
- You need weighted scoring
- You want to evaluate multiple aspects

### Example: Multi-Aspect Evaluator

```python
def business_score(output, expected, llm_metrics=None):
    """
    Custom evaluator for business requirements.

    Returns:
        float: Score between 0.0 and 1.0
    """
    score = 0.0

    # Check correctness (40% weight)
    if output and expected:
        if output.lower() == expected.lower():
            score += 0.4

    # Check completeness (30% weight)
    if len(output) > 10:  # Minimum length requirement
        score += 0.3

    # Check format (30% weight)
    if output.startswith(("The", "A", "An")):  # Proper sentence
        score += 0.3

    return min(1.0, score)

@traigent.optimize(
    scoring_function=business_score,
    eval_dataset="business_cases.jsonl"
)
def business_agent(query):
    return response
```

### Example: ROUGE Score Evaluator

```python
from rouge_score import rouge_scorer

def rouge_score_fn(output, expected, llm_metrics=None):
    """Evaluate using ROUGE score for summarization."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(expected, output)

    # Return ROUGE-L F1 score
    return scores['rougeL'].fmeasure

@traigent.optimize(
    scoring_function=rouge_score_fn,
    eval_dataset="summaries.jsonl"
)
def summarizer(text):
    return summary
```

## Troubleshooting

### Issue: Getting 0.0% Accuracy

**Causes and Solutions:**

1. **Missing API Key for Embeddings**
   ```bash
   export OPENAI_API_KEY=your-key-here
   ```

2. **Wrong Dataset Format**
   ```python
   # ❌ Wrong
   {"question": "...", "answer": "..."}

   # ✅ Correct
   {"input": {"question": "..."}, "output": "..."}
   ```

3. **Type Mismatch**
   ```python
   # Ensure output types match
   def evaluator(output, expected):
       return 1.0 if str(output) == str(expected) else 0.0
   ```

4. **Use Mock Mode for Testing**
   ```bash
   export TRAIGENT_MOCK_MODE=true
   ```

### Issue: Inconsistent Results

**Solutions:**

1. **Set Random Seed**
   ```python
   import random
   import numpy as np
   random.seed(42)
   np.random.seed(42)
   ```

2. **Use Larger Dataset**
   - Minimum 50 examples for stable results

3. **Check for Data Leakage**
   - Ensure test data isn't in training

## Examples

### Complete Sentiment Analysis Example

```python
import traigent
import json
from pathlib import Path

# Create evaluation dataset
dataset = [
    {"input": {"text": "Amazing product!"}, "output": "positive"},
    {"input": {"text": "Terrible experience"}, "output": "negative"},
    {"input": {"text": "It's acceptable"}, "output": "neutral"},
]

# Save dataset
with open("sentiment_eval.jsonl", "w") as f:
    for item in dataset:
        f.write(json.dumps(item) + "\n")

# Define simple scoring function for exact match
def sentiment_score(output, expected, llm_metrics=None):
    """Exact match for sentiment classification."""
    return 1.0 if output.lower() == expected.lower() else 0.0

# Create optimizable function
@traigent.optimize(
    configuration_space={
        "temperature": [0.0, 0.3, 0.7],
        "model": ["gpt-3.5-turbo", "gpt-4"]
    },
    eval_dataset="sentiment_eval.jsonl",
    scoring_function=sentiment_score,
    objectives=["accuracy"],
    max_trials=10
)
def sentiment_classifier(text, temperature=0.3, model="gpt-3.5-turbo"):
    # Your LLM logic here
    # This is simplified for example
    if "amazing" in text.lower() or "great" in text.lower():
        return "positive"
    elif "terrible" in text.lower() or "bad" in text.lower():
        return "negative"
    else:
        return "neutral"

# Run optimization
import asyncio
results = asyncio.run(sentiment_classifier.optimize())

print(f"Best config: {results.best_config}")
print(f"Best accuracy: {results.best_score:.2%}")
```

### Q&A System with Semantic Evaluation

```python
import traigent
from langchain_openai import ChatOpenAI

@traigent.optimize(
    configuration_space={
        "temperature": [0.0, 0.5, 1.0],
        "max_tokens": [50, 100, 200]
    },
    eval_dataset="data/qa_dataset.jsonl",  # Uses default semantic similarity
    objectives=["accuracy", "cost"],
    max_trials=15
)
def qa_system(question, temperature=0.5, max_tokens=100):
    llm = ChatOpenAI(
        temperature=temperature,
        max_tokens=max_tokens
    )

    prompt = f"Answer concisely: {question}"
    response = llm.invoke(prompt)

    return response.content

# The default evaluator will use semantic similarity
# to compare answers, allowing for paraphrasing
```

## Summary

1. **Start Simple**: Use mock mode and default evaluation
2. **Understand Your Metrics**: Know what "accuracy" means for your use case
3. **Create Good Datasets**: Representative, balanced, sufficient size
4. **Customize When Needed**: Write evaluators for specific requirements
5. **Test Thoroughly**: Verify evaluation works before optimization

## Next Steps

- [Custom Evaluator Examples](../examples/evaluation/custom_evaluator.py)
- [Dataset Creation Tools](../examples/datasets/)
- [Troubleshooting Guide](../README.md#troubleshooting)
