# Problem Generation Package

This package provides tools for generating large-scale, diverse problem sets for LangChain optimization with the Traigent SDK.

## Overview

The problem generation system is designed to create comprehensive test suites with:
- **30+ diverse problems** across multiple domains
- **1000+ examples per problem** with guaranteed diversity
- **Smart gap analysis** to ensure comprehensive coverage
- **Memory-aware generation** to prevent repetition
- **Batch processing** optimized for Claude API limits

## Key Components

### 1. ProblemDiversityManager
Analyzes existing problems and identifies gaps in coverage across:
- Domains (technical, medical, financial, etc.)
- Problem types (classification, generation, analysis, etc.)
- Difficulty levels (easy to expert)
- Technical capabilities tested

### 2. ExampleMemory
Maintains compact representations of generated examples to:
- Track patterns and prevent repetition
- Store summaries efficiently (6-10 tokens per example)
- Guide generation toward underrepresented patterns

### 3. DiversityAnalyzer
Measures diversity across multiple dimensions:
- Pattern entropy (variety in example types)
- Difficulty balance (distribution across levels)
- Topic coverage (breadth of subjects)
- Lexical diversity (vocabulary variety)
- Structural diversity (input/output formats)

### 4. EnhancedExampleGenerator
Generates examples with:
- Memory context to avoid repetition
- Adaptive difficulty targeting
- Diversity optimization with retries
- Batch generation (50 examples per API call)

### 5. BatchProblemGenerator
Orchestrates the entire generation process:
- Manages multiple problem generation
- Parallel processing support
- Progress tracking and reporting
- Code generation for problem modules

## Usage

### Basic Generation
```bash
# Generate default suite (30 problems, 1000 examples each)
python generate_problem_suite.py

# Test mode (2 problems, 100 examples)
python generate_problem_suite.py --test-mode
```

### Custom Configuration
```bash
# Generate 20 problems with 500 examples each
python generate_problem_suite.py --problems 20 --examples 500

# Enable parallel generation
python generate_problem_suite.py --parallel --max-concurrent 5

# Custom output directory
python generate_problem_suite.py --output-dir my_problems
```

## Generation Process

1. **Gap Analysis**: Analyzes existing problems to identify coverage gaps
2. **Problem Planning**: Suggests new problems to fill gaps
3. **Batch Generation**: Generates examples in batches of 50
4. **Diversity Optimization**: Retries batches with low diversity
5. **Code Generation**: Creates complete problem modules
6. **Report Generation**: Saves detailed reports and metrics

## Output Structure

```
langchain_problems/
├── problem_1.py              # Generated problem modules
├── problem_2.py
├── ...
├── generation_reports/       # Generation reports
│   ├── problem_plan_*.json   # Initial problem specifications
│   ├── generation_report_*.json  # Final generation report
│   └── gap_analysis.json     # Coverage analysis
└── example_data/             # Detailed example data
    ├── problem_1_examples.json
    └── ...
```

## Diversity Metrics

Each batch is evaluated on:
- **Pattern Entropy**: Shannon entropy of pattern distribution (0-1)
- **Difficulty Balance**: How well distributed across difficulty levels (0-1)
- **Topic Coverage**: Unique topics relative to example count (0-1)
- **Lexical Diversity**: Unique words / total words (0-1)
- **Structural Diversity**: Variety in input/output structures (0-1)
- **Similarity Score**: Average pairwise similarity (lower is better)

Overall diversity score combines all metrics (0-100).

## Memory System

The memory system stores compact representations:
```json
{
  "d": "m",        // Difficulty: e/m/h/v/x
  "p": "que",      // Pattern type (first 3 chars)
  "t": "a1b2c3",   // Topic hash (first 6 chars)
  "c": 3,          // Complexity feature count
  "l": "m"         // Length: s/m/l
}
```

This allows including 30+ past examples in each generation request while using minimal tokens.

## Best Practices

1. **Start with test mode** to verify setup
2. **Review gap analysis** before full generation
3. **Monitor diversity scores** during generation
4. **Check generation reports** for quality metrics
5. **Use parallel mode** for faster generation (requires more API quota)

## Troubleshooting

- **Low diversity scores**: The system will retry up to 3 times
- **Failed generations**: Check error details in generation report
- **Memory issues**: Reduce batch size or concurrent generations
- **API limits**: Adjust batch_size in GenerationConfig
