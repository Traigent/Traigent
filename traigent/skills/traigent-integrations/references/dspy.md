# DSPy Adapter Reference

## Overview

Traigent provides the `DSPyPromptOptimizer` adapter for integrating DSPy's automatic prompt optimization into your workflow. DSPy optimizes prompts through demonstration selection and instruction tuning, while Traigent handles model selection and parameter tuning.

## Installation

```bash
pip install traigent dspy
```

## DSPyPromptOptimizer

The adapter wraps DSPy's MIPROv2 and BootstrapFewShot optimizers.

### Import

```python
from traigent.integrations.dspy_adapter import DSPyPromptOptimizer
```

### Constructor

```python
DSPyPromptOptimizer(
    method="mipro",           # "mipro" or "bootstrap"
    teacher_model=None,       # Optional teacher model for MIPRO
    # Additional keyword arguments passed to the underlying DSPy optimizer
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `method` | `"mipro" \| "bootstrap"` | `"mipro"` | Which DSPy optimizer to use. `"mipro"` uses MIPROv2 (instruction + demo optimization). `"bootstrap"` uses BootstrapFewShot (demo-only optimization). |
| `teacher_model` | `str \| None` | `None` | Model name for the teacher in MIPRO (e.g., `"gpt-4o"`). If `None`, uses the same model as the student. |

### optimize_prompt()

```python
result = optimizer.optimize_prompt(
    module=my_dspy_module,    # DSPy module to optimize
    trainset=train_examples,  # List of dspy.Example objects
    metric=accuracy_fn,       # Metric function: (example, prediction) -> float
)
```

Returns a `PromptOptimizationResult`:

| Field | Type | Description |
|---|---|---|
| `optimized_module` | `Any` | The optimized DSPy module with tuned prompts and demonstrations. |
| `method` | `str` | The optimization method used (`"mipro"` or `"bootstrap"`). |
| `num_demos` | `int` | Number of demonstrations in the optimized prompt. |
| `trainset_size` | `int` | Size of the training set used. |
| `best_score` | `float \| None` | Best metric score achieved during optimization. |
| `metadata` | `dict[str, Any]` | Additional metadata from the optimization process. |

## MIPRO Method

MIPROv2 optimizes both the instruction text and the few-shot demonstrations:

```python
import dspy
from traigent.integrations.dspy_adapter import DSPyPromptOptimizer

# Configure DSPy
lm = dspy.LM("gpt-4o-mini")
dspy.configure(lm=lm)

# Define a DSPy module
class QAModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict("question -> answer")

    def forward(self, question):
        return self.predict(question=question)

# Define metric
def exact_match(example, prediction, trace=None):
    return example.answer.lower() == prediction.answer.lower()

# Prepare training data
trainset = [
    dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
    dspy.Example(question="Capital of France?", answer="Paris").with_inputs("question"),
    # ... more examples
]

# Optimize with MIPRO
optimizer = DSPyPromptOptimizer(
    method="mipro",
    teacher_model="gpt-4o",  # Use a stronger model as teacher
)
result = optimizer.optimize_prompt(
    module=QAModule(),
    trainset=trainset,
    metric=exact_match,
)

print(f"Best score: {result.best_score}")
print(f"Demos: {result.num_demos}")

# Use the optimized module
optimized_qa = result.optimized_module
answer = optimized_qa("What is the speed of light?")
```

## BootstrapFewShot Method

BootstrapFewShot selects the best few-shot demonstrations from the training set:

```python
optimizer = DSPyPromptOptimizer(method="bootstrap")

result = optimizer.optimize_prompt(
    module=QAModule(),
    trainset=trainset,
    metric=exact_match,
)

# The optimized module has curated demonstrations
optimized_module = result.optimized_module
print(f"Selected {result.num_demos} demonstrations")
```

## Combining DSPy with Traigent Model Optimization

Use Traigent for model/parameter optimization and DSPy for prompt optimization:

```python
import traigent
import dspy

@traigent.optimize(
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4o", "claude-3-haiku-20240307"],
        "temperature": [0.0, 0.3, 0.7],
    },
    objectives=["accuracy"],
    max_trials=9,
)
def optimized_qa(question):
    config = traigent.get_config()

    # Traigent manages model selection
    lm = dspy.LM(config["model"], temperature=config["temperature"])
    dspy.configure(lm=lm)

    # DSPy handles the prompt structure
    qa = dspy.Predict("question -> answer")
    result = qa(question=question)
    return result.answer

# Traigent finds the best model + temperature
results = optimized_qa.optimize(dataset="qa_eval.jsonl")
```

For a two-stage approach (Traigent model optimization, then DSPy prompt optimization):

```python
# Stage 1: Find the best model with Traigent
model_results = optimized_qa.optimize(dataset="qa_eval.jsonl")
best_model = model_results.best_config["model"]
best_temp = model_results.best_config["temperature"]

# Stage 2: Optimize prompts with DSPy using the best model
lm = dspy.LM(best_model, temperature=best_temp)
dspy.configure(lm=lm)

optimizer = DSPyPromptOptimizer(method="mipro")
prompt_result = optimizer.optimize_prompt(
    module=QAModule(),
    trainset=trainset,
    metric=exact_match,
)

# Use the fully optimized system
final_module = prompt_result.optimized_module
```

## Tips

- MIPRO is more powerful but slower; BootstrapFewShot is faster for demo-only optimization
- Use a stronger teacher model (e.g., `gpt-4o`) for MIPRO when the student model is smaller
- DSPy requires structured inputs/outputs; define your module's signature clearly
- Training set quality matters more than quantity for few-shot optimization
- The `optimized_module` retains all DSPy functionality and can be saved/loaded with `dspy.save`/`dspy.load`
