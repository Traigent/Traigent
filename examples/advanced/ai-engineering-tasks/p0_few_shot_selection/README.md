# P0-3: Few-Shot Example Selection Optimization

**Discover optimal example selection strategies to increase accuracy by 8-15% and reduce variance by 30%**

## Overview

Few-shot learning effectiveness depends critically on which examples are selected, their ordering, and formatting. Teams report 20-40% accuracy variations based solely on example selection, yet most still use random or manually curated examples. This example demonstrates how Traigent systematically discovers optimal selection strategies that significantly improve task performance.

## Problem Statement

Given a classification or generation task with a pool of labeled examples, determine the optimal strategy for:
- Example selection algorithm (semantic similarity, diversity sampling, curriculum learning)
- Number of examples and class balance
- Ordering method (similarity-based, difficulty progression, diversity-first)
- Formatting approach (I/O pairs, explanations, structured fields)

## Key Features

### Modern Selection Strategies
- **Semantic KNN**: Nearest neighbors based on embedding similarity
- **Diversity Sampling**: Maximum coverage of input space
- **MMR (Maximal Marginal Relevance)**: Balance relevance and diversity
- **Curriculum Learning**: Easy-to-hard progression
- **Contrastive Selection**: Positive and negative examples
- **Influence-Based**: Examples with highest training influence

### Advanced Ordering Methods
- **Similarity-Based**: Most/least similar first
- **Difficulty Progression**: Gradual complexity increase
- **Diversity-First**: Spread examples across feature space
- **Alternating**: Mix different ordering strategies

### Intelligent Formatting
- **I/O Pairs**: Simple input-output format
- **With Explanations**: Include reasoning steps
- **Structured Fields**: Field-by-field breakdown
- **XML-Wrapped**: Tagged sections for clarity
- **Conversational**: Dialog-style formatting

## Quick Start

### Installation
```bash
# Install dependencies
pip install traigent rich numpy scikit-learn

# Navigate to the example
cd examples/advanced/ai-engineering-tasks/p0_few_shot_selection/
```

### Run the Example
```bash
# Run with mock mode (no external API calls)
TRAIGENT_MOCK_MODE=true python main.py

# Run with real optimization (requires API keys)
python main.py
```

## Configuration Space

The optimization explores these selection strategies:

### Selection Methods
```python
selection_method = [
    "random",                  # Baseline random selection
    "semantic_knn",           # K-nearest neighbors
    "semantic_diverse",       # Diverse representative set
    "mmr",                   # Maximal Marginal Relevance
    "cluster_centroids",     # Cluster representatives
    "influence_based",       # High-influence examples
    "contrastive",          # Positive + negative examples
    "curriculum",           # Easy to hard progression
    "uncertainty_based"     # Model-uncertain examples
]
```

### Example Composition
```python
n_examples = [0, 1, 3, 5, 8]              # Number of examples
example_ratio = {
    "positive": (0.0, 1.0),               # For binary classification
    "per_class": "balanced"               # For multi-class tasks
}
```

### Ordering Strategies
```python
example_ordering = [
    "random",                 # Random order
    "similarity_desc",        # Most similar first
    "similarity_asc",         # Least similar first
    "difficulty_progression", # Easy to hard
    "diversity_first",       # Diverse then similar
    "alternating"            # Alternate strategies
]
```

### Formatting Options
```python
example_format = [
    "input_output",              # Simple I/O pairs
    "input_output_explanation",  # With reasoning
    "structured_fields",         # Field-by-field
    "conversational",           # Dialog format
    "xml_wrapped"              # XML-tagged sections
]
```

### Advanced Features
```python
dynamic_selection = [True, False]     # Query-adaptive selection
cache_selections = [True, False]      # Cache for performance
example_placement = [
    "before_instruction",             # Examples first
    "after_instruction",              # Examples after task description
    "interleaved"                     # Mixed throughout prompt
]
```

## Success Criteria

### Accuracy Improvements
- **Target**: 8-15% accuracy improvement over random selection
- **Measurement**: Task-specific metrics (accuracy, F1, BLEU)
- **Baseline**: Random selection of same number of examples

### Variance Reduction
- **Target**: 30% reduction in output variance
- **Measurement**: Consistency across multiple runs
- **Importance**: Reliable performance in production

### Selection Efficiency
- **Target**: <100ms selection latency with caching
- **Requirement**: Sub-second example selection for real-time use
- **Optimization**: Pre-computed embeddings and caching strategies

## Dataset and Task Types

### Classification Tasks
- **Sentiment Analysis**: Product reviews, movie reviews
- **Intent Recognition**: Customer support, voice assistants
- **Topic Classification**: News articles, document categorization

### Generation Tasks
- **Email Writing**: Business, personal, customer service
- **Content Generation**: Marketing copy, technical documentation
- **Code Generation**: Function implementation, bug fixes

### Reasoning Tasks
- **Math Problems**: Word problems, equation solving
- **Logic Puzzles**: Deductive reasoning, pattern recognition
- **Causal Analysis**: Root cause identification, impact assessment

### Example Types by Difficulty
- **Easy** (0.1-0.3): Clear patterns, obvious solutions
- **Medium** (0.4-0.7): Some ambiguity, multiple approaches
- **Hard** (0.8-1.0): Complex reasoning, edge cases

## Example Results

### Performance by Selection Method

| Selection Method | Accuracy | Consistency | Latency | Best For |
|------------------|----------|-------------|---------|----------|
| Random           | 0.72     | 0.65        | 1ms     | Baseline |
| Semantic KNN     | 0.81     | 0.82        | 15ms    | Similar queries |
| MMR              | 0.85     | 0.92        | 45ms    | Balanced tasks |
| Curriculum       | 0.79     | 0.87        | 25ms    | Complex reasoning |
| Traigent Optimized | **0.87** | **0.94**   | 35ms    | **All tasks** |

### Task-Specific Insights

#### Classification Tasks
- **Best Strategy**: Diverse boundary examples near decision boundaries
- **Optimal Number**: 3-5 examples for binary, 5-8 for multi-class
- **Key Finding**: Class balance more important than total number

#### Generation Tasks
- **Best Strategy**: Similar inputs with varied, high-quality outputs
- **Optimal Number**: 2-3 examples to avoid overwhelming context
- **Key Finding**: Output diversity prevents template copying

#### Reasoning Tasks
- **Best Strategy**: Curriculum learning (easy → hard progression)
- **Optimal Number**: 5-8 examples showing solution patterns
- **Key Finding**: Step-by-step explanations crucial for complex tasks

## Integration Examples

### Using Optimal Selection
```python
from p0_few_shot_selection.selection_config import create_selection_config
from p0_few_shot_selection.evaluator import select_examples

# Create optimized configuration
config = create_selection_config(
    selection_method="mmr",
    n_examples=5,
    ordering_strategy="curriculum",
    formatting="io_with_explanation"
)

# Select examples for a query
selected = select_examples(
    query="Classify sentiment: The service was okay but could be better",
    candidates=example_pool,
    config=config,
    task_type="classification"
)
```

### Custom Selection Strategy
```python
import traigent
from selection_config import SELECTION_SEARCH_SPACE

# Customize for your task
custom_space = {
    **SELECTION_SEARCH_SPACE,
    "selection_method": ["mmr", "semantic_knn"],  # Focus on proven methods
    "n_examples": [3, 5],                         # Limit for cost control
    "formatting": ["io_pairs", "with_explanation"] # Task-appropriate formats
}

@traigent.optimize(config_space=custom_space)
def your_few_shot_function(**config):
    # Your implementation
    pass
```

## Architecture

### Core Components

1. **Dataset Generation** (`dataset.py`)
   - Multi-task example pools
   - Difficulty scoring and metadata
   - Diversity analysis functions

2. **Selection Configuration** (`selection_config.py`)
   - Comprehensive search space
   - Configuration validation
   - Parameter optimization

3. **Selection & Evaluation** (`evaluator.py`)
   - Multiple selection algorithms
   - Performance evaluation metrics
   - Caching and optimization

4. **Main Optimization** (`main.py`)
   - Traigent integration
   - Multi-objective optimization
   - Results analysis and insights

### Selection Pipeline
```
Query → Embedding → Candidate Filtering → Selection Algorithm → Ordering → Formatting → Evaluation
```

## Advanced Usage Patterns

### Task-Adaptive Selection
```python
# Different strategies for different task types
def get_optimal_config(task_type, query_complexity):
    if task_type == "classification":
        if query_complexity == "simple":
            return create_selection_config(
                selection_method="semantic_knn",
                n_examples=3
            )
        else:
            return create_selection_config(
                selection_method="mmr",
                n_examples=5
            )
    elif task_type == "generation":
        return create_selection_config(
            selection_method="semantic_diverse",
            n_examples=2,
            formatting="with_explanation"
        )
```

### Dynamic Selection with Caching
```python
class CachedExampleSelector:
    def __init__(self):
        self.cache = {}
        self.embeddings = precompute_embeddings(example_pool)

    def select(self, query, config):
        cache_key = hash((query, str(config)))
        if cache_key in self.cache:
            return self.cache[cache_key]

        selected = self._select_fresh(query, config)
        self.cache[cache_key] = selected
        return selected
```

### Multi-Objective Optimization
```python
@traigent.optimize(
    objectives=[
        "accuracy",           # Primary: maximize accuracy
        "consistency",        # Primary: maximize consistency
        "-selection_latency", # Secondary: minimize latency
        "diversity_score"     # Secondary: maximize diversity
    ]
)
def optimize_selection(**config):
    # Traigent finds optimal trade-offs
    pass
```

## Performance Tuning

### For High Accuracy
- Use semantic similarity methods (KNN, MMR)
- Include 5-8 examples for complex tasks
- Add explanations for reasoning tasks
- Ensure class balance in classification

### for Speed
- Pre-compute embeddings
- Enable caching
- Limit to 3 examples maximum
- Use simple I/O formatting

### For Consistency
- Use curriculum learning ordering
- Avoid random elements
- Include diverse examples
- Test across multiple seeds

### For Cost Optimization
- Minimize number of examples
- Use simple formatting
- Cache selections aggressively
- Consider static curation for stable tasks

## Troubleshooting

### Common Issues

**Low Accuracy Improvement**:
- Check example quality and relevance
- Try different embedding models
- Increase number of examples
- Add task-specific formatting

**High Variance**:
- Remove randomization elements
- Use curriculum ordering
- Ensure sufficient example diversity
- Test example stability

**Slow Selection**:
- Pre-compute embeddings
- Enable caching
- Reduce candidate pool size
- Use approximate similarity search

**Poor Generalization**:
- Increase example diversity
- Use contrastive selection
- Test on held-out queries
- Avoid overfitting to training distribution

## Related Examples

- [P0-1: Structured Output Engineering](../p0_structured_output/) - For reliable output parsing
- [P0-2: Context Engineering & RAG](../p0_context_engineering/) - For optimal context assembly
- [P1-2: Token Budget Optimization](../p1_token_budget/) - For managing token costs with examples

## Contributing

This example demonstrates few-shot selection optimization. To extend:

1. Add new selection algorithms in `evaluator.py`
2. Implement task-specific formatting in `dataset.py`
3. Add custom similarity metrics
4. Create domain-specific example pools

## References

- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [What Makes Good In-Context Examples for GPT-3?](https://arxiv.org/abs/2101.06804)
- [Fantastically Ordered Prompts](https://arxiv.org/abs/2104.08786)
- [Active Example Selection for In-Context Learning](https://arxiv.org/abs/2211.04486)
