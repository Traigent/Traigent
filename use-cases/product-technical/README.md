# Product & Technical Agent Use Case

> **Codegen that passes tests - then optimizes for elegance**

<p align="center">
  <a href="demo/demo.cast">
    <img src="demo/demo.svg" alt="Product Technical Demo" width="600">
  </a>
</p>

This use case demonstrates optimizing a **code generation agent** that writes Python functions based on specifications.

## Overview

The code generation agent:
1. Takes a task description and function signature
2. Generates Python code that implements the function
3. Is evaluated by running actual test cases

It optimizes for:
- **Test Pass Rate** - Functional correctness via test execution
- **Code Quality** - Readability, complexity, style
- **Solution Efficiency** - Conciseness and algorithmic efficiency

## Quick Start

```bash
# From project root
cd /path/to/Traigent

# Enable mock mode (recommended for testing)
export TRAIGENT_MOCK_MODE=true

# Run the agent optimization
python use-cases/product-technical/agent/code_agent.py
```

## Configuration Space

| Parameter | Values | Description |
|-----------|--------|-------------|
| `model` | gpt-3.5-turbo, gpt-4o-mini, gpt-4o | LLM model selection |
| `temperature` | 0.0, 0.2, 0.4 | Low for consistent code |
| `coding_style` | concise, verbose, documented | Code style preference |
| `approach` | direct, test_first | Think about tests first vs implement directly |

## Dataset

The evaluation dataset (`datasets/coding_tasks.jsonl`) contains 50+ coding tasks including:

- Basic algorithms (prime, factorial, fibonacci)
- String manipulation (reverse, palindrome, anagram)
- List operations (sort, search, merge)
- Data structures (stack validation, power set)
- Mathematical functions (GCD, LCM, Roman numerals)

### Sample Entry

```json
{
  "input": {
    "task": "Write a function that checks if a number is prime",
    "function_name": "is_prime",
    "signature": "def is_prime(n: int) -> bool"
  },
  "test_cases": [
    {"input": [2], "expected": true},
    {"input": [4], "expected": false},
    {"input": [17], "expected": true},
    {"input": [1], "expected": false}
  ],
  "reference_solution": "def is_prime(n: int) -> bool:\n    if n < 2:\n        return False\n    ..."
}
```

## Evaluation Metrics

### Test Pass Rate (50% weight)

- **Deterministic**: Actually executes generated code
- **Test Cases**: Each task has 3-8 test cases
- **Edge Cases**: Includes boundary conditions and error cases
- **Pass Rate**: `tests_passed / tests_total`

### Code Quality (30% weight)

- **Syntax Validity**: Must parse without errors
- **Complexity**: Penalizes excessive nesting (>5 levels)
- **Style**: Rewards type hints, docstrings
- **Line Length**: Penalizes lines >100 characters

### Solution Efficiency (20% weight)

- **Conciseness**: Fewer lines is better (within reason)
- **Reference Comparison**: Compares to optimal solution
- **Patterns**: Rewards use of list comprehensions, builtins

## Files

```
product-technical/
├── agent/
│   └── code_agent.py             # Code generation agent
├── datasets/
│   └── coding_tasks.jsonl        # 50+ coding tasks
├── eval/
│   └── evaluator.py              # Test runner + quality analyzer
└── README.md
```

## Expected Results

After optimization, you should see results like:

```
Best Configuration:
  model: gpt-4o-mini
  temperature: 0.2
  coding_style: concise
  approach: direct

Best Score: 0.75
```

## Key Features

### Actual Code Execution

Unlike text-based evaluation, this agent:
- Actually runs the generated code
- Executes real test cases
- Catches runtime errors and exceptions
- Compares outputs to expected values

### Quality Analysis

Static analysis includes:
- AST parsing for syntax validation
- Complexity measurement
- Pattern detection (comprehensions, builtins)
- Style checks

## Customization

### Adding Coding Tasks

Add entries to `datasets/coding_tasks.jsonl`:

```json
{
  "input": {
    "task": "Your task description",
    "function_name": "your_function",
    "signature": "def your_function(x: int) -> int"
  },
  "test_cases": [
    {"input": [1], "expected": 2},
    {"input": [5], "expected": 10}
  ],
  "reference_solution": "def your_function(x: int) -> int:\n    return x * 2"
}
```

### Modifying Evaluation

Edit `eval/evaluator.py` to adjust:
- Test execution logic
- Quality scoring weights
- Efficiency metrics

### Testing the Evaluator

```bash
python use-cases/product-technical/eval/evaluator.py
```

## Safety Note

The evaluator executes generated code in a sandboxed namespace. However, for production use, consider:
- Running in a container/sandbox
- Setting execution timeouts
- Restricting imports
