# Code Review Agent Use Case

LLM-powered code quality analysis for Python functions.

This use case demonstrates optimizing an LLM agent to identify code quality
issues in Python source code from the Traigent SDK.

## Overview

The code review agent:
1. Receives a Python function as input
2. Analyzes it for 10 categories of code quality issues
3. Returns a structured list of identified issues

It optimizes for:
- **Detection F1** - Balance of precision and recall for issue detection
- **Cost** - API costs for the analysis

## Quick Start

```bash
# From project root
cd /path/to/Traigent

# Enable mock mode (recommended for testing)
export TRAIGENT_MOCK_LLM=true

# Run the agent optimization
python use-cases/code-review/agent/code_review_agent.py

# Or run demo mode to see sample analysis
python use-cases/code-review/agent/code_review_agent.py --demo
```

## Configuration Space

| Parameter          | Values                                        | Description                            |
|--------------------|-----------------------------------------------|----------------------------------------|
| `model`            | gpt-4.1, gpt-4o, gpt-5-mini, grok-code-fast-1 | Free LLM models via GitHub Copilot CLI |
| `prompt_strategy`  | direct, chain_of_thought, checklist           | Prompting approach                     |

### Prompting Strategies

- **direct**: Simple prompt listing issue categories to check
- **chain_of_thought**: Step-by-step analysis prompting systematic review
- **checklist**: Explicit checklist with specific criteria per category

## Issue Categories

| Category | Description |
|----------|-------------|
| `MISSING_DOCS` | Incomplete or missing docstrings |
| `IMPLICIT_ASSUMPTION` | Unvalidated inputs, implicit assumptions |
| `SIDE_EFFECT` | Global state mutation, side effects |
| `COMPLEXITY` | Complex logic, deep nesting, potential bugs |
| `BROAD_EXCEPTION` | Overly broad exception handling (`except Exception:`) |
| `PRINCIPLE_VIOLATION` | SRP violations, functions >50 lines |
| `TODO_KNOWN_ISSUE` | TODO/FIXME/HACK comments, known issues |
| `TYPE_HANDLING` | Unsafe type operations without checks |
| `API_DESIGN` | >5 parameters, unclear naming |
| `THREADING_ISSUE` | Race conditions, thread safety issues |

## Output Format

The agent returns a simple JSON array:

```json
[
  {"issue_type": "MISSING_DOCS", "description": "Function lacks docstring"},
  {"issue_type": "BROAD_EXCEPTION", "description": "except Exception: is too broad"}
]
```

If no issues are found, returns an empty array: `[]`

## Dataset

50 Python functions extracted from `traigent/` source code with manually
labeled ground truth issues. The dataset covers:

- Functions with 0-4 issues each
- All 10 issue categories represented
- Real production code examples
- Various function sizes and complexity levels

### Source Files

Functions extracted from:
- `traigent/api/functions.py` - Core API functions
- `traigent/api/constraints.py` - Constraint system
- `traigent/config/types.py` - Configuration types
- `traigent/security/session_manager.py` - Session management
- `traigent/traigent_client.py` - Client implementation

## Evaluation Metrics

The evaluator uses **type-level matching**:
- Compare predicted issue types against ground truth types
- Descriptions may vary, but categories should match

**Metrics:**
- `detection_precision`: Of issues predicted, how many are correct types?
- `detection_recall`: Of actual issues, how many types were detected?
- `detection_f1`: Harmonic mean of precision and recall

## Files

```
code-review/
├── README.md                           # This file
├── agent/
│   ├── __init__.py
│   └── code_review_agent.py           # Main agent with @traigent.optimize
├── datasets/
│   ├── __init__.py
│   └── code_issues.jsonl              # 50 labeled examples
└── eval/
    ├── __init__.py
    └── evaluator.py                   # Detection accuracy evaluator
```

## LLM Provider

Uses **GitHub Copilot CLI** for free model access:
- Requires `gh` CLI installed and authenticated
- Models: gpt-4.1, gpt-4o, gpt-5-mini, grok-code-fast-1
- All models available at 0x cost tier

### Setup

```bash
# Install GitHub CLI
brew install gh  # macOS
# or: sudo apt install gh  # Ubuntu

# Authenticate
gh auth login

# Install Copilot extension
gh extension install github/gh-copilot
```

## Expected Results

After optimization with mock mode:

```
Best Configuration:
  model: gpt-4o
  temperature: 0.0
  prompt_strategy: checklist

Best F1 Score: 0.72
```

## Development

### Run Tests

```bash
# Test evaluator
python use-cases/code-review/eval/evaluator.py

# Test agent in demo mode
python use-cases/code-review/agent/code_review_agent.py --demo
```

### Add Examples to Dataset

Edit `datasets/code_issues.jsonl` with new examples in JSONL format:

```json
{"input": {"function_code": "...", "function_name": "...", "source_file": "..."}, "output": {"issues": [...]}}
```

## Key Design Decisions

1. **Simple Output Format**: `[{"issue_type": "...", "description": "..."}]` - easy to parse and evaluate
2. **No Scoring by Model**: Agent only identifies issues, evaluator scores them
3. **Type-Level Matching**: Evaluator compares issue types, not descriptions (more robust)
4. **Strict JSON Output**: All prompts include "Return ONLY valid JSON" to prevent parse failures
5. **Multi-Objective**: Optimizes for both detection accuracy (F1) and cost

## Related Examples

- [customer-support](../customer-support/) - Similar agent optimization pattern
- [product-technical](../product-technical/) - Code generation agent
- [dspy-hotpotqa](../../examples/integrations/dspy-hotpotqa/) - DSPy prompt optimization
