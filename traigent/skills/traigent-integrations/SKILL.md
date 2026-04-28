---
name: traigent-integrations
description: "Integrate Traigent with LangChain, LiteLLM, DSPy, and other AI frameworks. Use when importing langchain/litellm/dspy alongside traigent, setting up multi-provider model testing, using auto_override_frameworks, or asking about framework-specific adapter patterns."
license: Apache-2.0
metadata:
  author: Traigent
  version: "1.0"
---

# Traigent Framework Integrations

## When to Use

Use this skill when:

- Combining Traigent optimization with LangChain, LiteLLM, or DSPy
- Setting up multi-provider model testing (e.g., OpenAI + Anthropic + Google)
- Using `auto_override_frameworks` or `framework_targets` in the decorator
- Writing optimized functions that call framework-specific APIs
- Connecting Traigent results to observability tools (MLflow, Weights & Biases)

## Installation

Install Traigent with framework integration support:

```bash
# All integrations
pip install traigent[integrations]

# Or install individual frameworks alongside Traigent
pip install traigent langchain-openai langchain-anthropic
pip install traigent litellm
pip install traigent dspy
```

## LangChain Integration

Traigent integrates with LangChain by optimizing the model and parameters used inside your chain. The key pattern: get the config from Traigent, then construct your LangChain objects.

### Basic Pattern

```python
import traigent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

@traigent.optimize(
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4o"],
        "temperature": [0.0, 0.3, 0.7, 1.0],
    },
    objectives=["accuracy"],
    max_trials=10,
)
def answer_question(question):
    config = traigent.get_config()

    # Create LangChain components using Traigent config
    llm = ChatOpenAI(
        model=config["model"],
        temperature=config["temperature"],
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question accurately and concisely."),
        ("human", "{question}"),
    ])
    chain = prompt | llm
    response = chain.invoke({"question": question})
    return response.content

results = answer_question.optimize(dataset="questions.jsonl")
```

### Auto Override Frameworks

The `auto_override_frameworks` flag lets Traigent automatically intercept LangChain model instantiation to inject the optimized configuration. This is useful when you want Traigent to manage model selection across your chain without manually threading `get_config()`:

```python
@traigent.optimize(
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4o", "claude-3-haiku-20240307"],
        "temperature": [0.0, 0.5, 1.0],
    },
    objectives=["accuracy"],
    max_trials=12,
    auto_override_frameworks=True,
)
def summarize_document(text):
    # LangChain model instantiation is intercepted by Traigent
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    response = llm.invoke(text)
    return response.content
```

For finer control, use `framework_targets` to specify exactly which classes to override:

```python
@traigent.optimize(
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4o"],
        "temperature": [0.0, 0.5],
    },
    objectives=["accuracy"],
    max_trials=8,
    framework_targets=["langchain_openai.ChatOpenAI"],
)
def my_chain(input_text):
    llm = ChatOpenAI(model="gpt-4o-mini")  # Will be overridden
    return llm.invoke(input_text).content
```

See [LangChain reference](references/langchain.md) for RAG chain optimization and advanced patterns.

## LiteLLM Multi-Provider

LiteLLM provides a unified `completion()` interface across 100+ LLM providers. This makes it natural to optimize across providers with Traigent:

```python
import traigent
import litellm

@traigent.optimize(
    configuration_space={
        "model": [
            "gpt-4o-mini",          # OpenAI
            "gpt-4o",               # OpenAI
            "claude-3-haiku-20240307",  # Anthropic
            "claude-3-5-sonnet-20241022",  # Anthropic
            "gemini/gemini-1.5-flash",    # Google
        ],
        "temperature": [0.0, 0.3, 0.7],
        "max_tokens": [256, 512, 1024],
    },
    objectives=["accuracy"],
    max_trials=15,
)
def classify_text(text):
    config = traigent.get_config()

    response = litellm.completion(
        model=config["model"],
        messages=[{"role": "user", "content": f"Classify this text: {text}"}],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
    )
    return response.choices[0].message.content

results = classify_text.optimize(dataset="classification_eval.jsonl")

# Check cost across providers
for trial in results.successful_trials:
    model = trial.config["model"]
    accuracy = trial.get_metric("accuracy", 0.0)
    print(f"{model}: accuracy={accuracy:.2%}")
```

LiteLLM handles API key routing automatically based on the model prefix. Set provider API keys in environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="..."
```

See [LiteLLM reference](references/litellm.md) for the full provider list and cost tracking details.

## DSPy Integration

Traigent provides a `DSPyPromptOptimizer` adapter that wraps DSPy's MIPROv2 and BootstrapFewShot optimizers for automatic prompt engineering:

```python
from traigent.integrations.dspy_adapter import DSPyPromptOptimizer

optimizer = DSPyPromptOptimizer(method="mipro")

result = optimizer.optimize_prompt(
    module=my_dspy_module,
    trainset=train_examples,
    metric=accuracy_metric,
)

# Access the optimized module
optimized_module = result.optimized_module
print(f"Best score: {result.best_score}")
print(f"Method: {result.method}")
print(f"Demos: {result.num_demos}")
```

You can also use DSPy modules inside a Traigent-optimized function for model-level optimization:

```python
import traigent
import dspy

@traigent.optimize(
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4o"],
        "temperature": [0.0, 0.5, 1.0],
    },
    objectives=["accuracy"],
    max_trials=6,
)
def dspy_qa(question):
    config = traigent.get_config()
    lm = dspy.LM(config["model"], temperature=config["temperature"])
    dspy.configure(lm=lm)

    qa = dspy.Predict("question -> answer")
    result = qa(question=question)
    return result.answer
```

See [DSPy reference](references/dspy.md) for BootstrapFewShot patterns and advanced configuration.

## Observability Integrations

### MLflow

Log Traigent optimization results to MLflow for experiment tracking:

```python
import mlflow

results = func.optimize(dataset="data.jsonl")

with mlflow.start_run():
    mlflow.log_param("algorithm", results.algorithm)
    mlflow.log_param("best_config", results.best_config)
    mlflow.log_metric("best_score", results.best_score or 0.0)
    mlflow.log_metric("total_cost", results.total_cost or 0.0)
    mlflow.log_metric("total_trials", len(results.trials))
    mlflow.log_metric("success_rate", results.success_rate)

    for trial in results.successful_trials:
        with mlflow.start_run(nested=True, run_name=trial.trial_id):
            mlflow.log_params(trial.config)
            mlflow.log_metrics(trial.metrics)
```

### Weights & Biases

```python
import wandb

results = func.optimize(dataset="data.jsonl")

wandb.init(project="traigent-optimization")
for trial in results.trials:
    wandb.log({
        "trial_id": trial.trial_id,
        "status": str(trial.status),
        **trial.config,
        **trial.metrics,
    })
wandb.log({
    "best_score": results.best_score,
    "best_config": results.best_config,
    "total_cost": results.total_cost,
})
wandb.finish()
```

## Pattern: The Right Way

When using Traigent with any framework, always follow this order:

1. Get the config from Traigent
2. Create framework objects using that config
3. Execute with those objects

```python
# CORRECT: get config first, then create client
@traigent.optimize(
    configuration_space={"model": ["gpt-4o-mini", "gpt-4o"], "temperature": [0.0, 0.5]},
    objectives=["accuracy"],
)
def my_func(text):
    config = traigent.get_config()           # 1. Get config
    llm = ChatOpenAI(                        # 2. Create client with config
        model=config["model"],
        temperature=config["temperature"],
    )
    return llm.invoke(text).content          # 3. Execute
```

Do not create the client outside the function or before getting the config:

```python
# WRONG: client created before config is available
llm = ChatOpenAI(model="gpt-4o-mini")  # Fixed model, Traigent cannot optimize this

@traigent.optimize(...)
def my_func(text):
    return llm.invoke(text).content  # Always uses the same model
```

The exception is when using `auto_override_frameworks=True`, which intercepts client construction automatically.

## Reference Files

- [LangChain Integration Details](references/langchain.md)
- [LiteLLM Multi-Provider Guide](references/litellm.md)
- [DSPy Adapter Reference](references/dspy.md)
