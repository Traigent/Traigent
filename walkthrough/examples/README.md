# Traigent Examples

Simple, clean examples showing how to use Traigent for LLM optimization.

## Requirements

Install Traigent with integrations to run all examples:

```bash
pip install traigent[integrations]
```

This includes:

- `langchain`, `langchain-openai`, `langchain-community` - LLM framework
- `faiss-cpu` - Vector search for RAG examples
- `openai`, `anthropic` - LLM providers

## Quick Start

```bash
# Mock examples - no API keys needed
python walkthrough/examples/mock/01_simple.py

# Real examples - requires OpenAI API key
export OPENAI_API_KEY="your-key"
python walkthrough/examples/real/01_simple.py
```

## Structure

```text
examples/
  mock/          # No API keys needed, instant results
  real/          # Requires API keys, real LLM calls
  datasets/      # Pre-built evaluation datasets (20 examples each)
```

## Examples Overview

| #  | Example          | Description                                | Dataset          | Evaluation Method          |
|----|------------------|--------------------------------------------|------------------|----------------------------|
| 01 | Simple           | Basic model and temperature tuning         | simple_questions | Exact Match                |
| 02 | Zero Code Change | Seamless mode intercepts hardcoded values  | simple_questions | Exact Match                |
| 03 | Parameter Mode   | Explicit configuration control             | simple_questions | Exact Match                |
| 04 | Multi-Objective  | Balance accuracy, cost, and latency        | classification   | Exact Match                |
| 05 | RAG              | Optimize retrieval and generation together | rag_questions    | Semantic Similarity        |
| 06 | Custom Evaluator | LLM-as-Judge for code generation           | code_gen         | LLM-as-Judge (GPT-4o-mini) |
| 07 | Privacy Modes    | Local, Cloud, and Hybrid execution         | simple_questions | Exact Match                |

## Datasets

Each dataset contains **20 examples** with varying difficulty levels to ensure different model configurations show measurable differences.

### simple_questions.jsonl

General knowledge Q&A with three difficulty tiers:

- **Easy (10)**: Factual questions with single-word answers (e.g., "What is 2+2?", "Capital of France?")
- **Medium (5)**: Questions requiring brief explanations (e.g., "What causes tides?")
- **Hard (5)**: Complex questions needing nuanced reasoning (e.g., "Explain opportunity cost with an example")

**Evaluation**: Exact match against expected output (case-insensitive substring match)

### classification.jsonl

Sentiment analysis with ambiguous cases:

- **Clear (10)**: Obvious positive/negative sentiment (e.g., "Best purchase ever!", "Complete waste of money")
- **Ambiguous (10)**: Subtle or mixed sentiment requiring careful analysis (e.g., "Not bad, not great", "Frustrating setup but eventually worked")

**Evaluation**: Exact match (positive/negative/neutral)

### rag_questions.jsonl

Traigent documentation Q&A for RAG testing:

- **Simple (10)**: Direct questions with factual answers from docs
- **Complex (10)**: Questions requiring synthesis of multiple document sections

**Evaluation**: Semantic similarity - checks if key concepts from expected answer appear in response

### code_gen.jsonl

Python code generation tasks with difficulty progression:

- **Easy (5)**: Basic functions (add numbers, check even, reverse string)
- **Medium (6)**: Intermediate algorithms (palindrome check, binary search, word frequency)
- **Hard (9)**: Advanced algorithms (LRU cache, Dijkstra's algorithm, serialize binary tree)

**Note**: This dataset has no `expected_output` field - Traigent allows this for custom evaluators that don't need reference answers. The LLM judge evaluates the generated code based on the task description directly.

**Evaluation**: LLM-as-Judge using GPT-4o-mini with detailed rubric:

- Correctness (40%): Does the code solve the task correctly?
- Code Quality (30%): Is it well-structured and Pythonic?
- Documentation (30%): Does it have proper docstrings/comments?

## Evaluation Methods

### Exact Match (Default)

Compares model output against expected output. Good for:

- Factual Q&A
- Classification tasks
- Multiple choice

### Semantic Similarity

Checks if key concepts appear in the response. Good for:

- RAG applications
- Open-ended questions
- Paraphrased answers

### LLM-as-Judge (Example 06)

Uses a separate LLM to evaluate response quality with a detailed rubric.

**Judge Prompt Features**:

- Clear evaluation criteria with scoring guidelines
- Structured JSON output for reliable parsing
- Weighted scoring across multiple dimensions
- Fallback to heuristics if judge fails

```python
# Example from 06_custom_evaluator.py
def llm_code_evaluator(output: str, expected: str, **kwargs) -> float:
    """Uses GPT-4o-mini to evaluate code quality."""
    judge = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # ... detailed rubric prompt ...
    result = judge.invoke(prompt)
    return weighted_score
```

## Mock vs Real

| Mock (`mock/`)     | Real (`real/`)           |
|--------------------|--------------------------|
| No API keys        | Requires OPENAI_API_KEY  |
| Instant results    | Real LLM calls           |
| Simulated accuracy | Actual model performance |
| Great for learning | Production-ready         |
| ~50 lines each     | ~60-150 lines each       |

## Why Varying Difficulty?

The datasets include varying difficulty levels so that:

1. **Different models show different results** - GPT-4 will outperform GPT-3.5 on hard questions
2. **Temperature effects are visible** - Lower temperature helps on factual, higher on creative
3. **Optimization is meaningful** - There's room for improvement across configurations
4. **Edge cases are covered** - Ambiguous examples test model robustness
