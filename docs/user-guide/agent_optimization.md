# Agent Optimization Guide

This guide describes the planned Model 2 agent-specification API.

> **Current status:** Remote agent optimization in Traigent Cloud is not implemented yet. The `execution_mode="cloud"` path fails closed with: “Cloud remote execution is not available yet; use hybrid for portal-tracked optimization.” Use `execution_mode="hybrid"` for portal-visible SDK runs today; trials still execute locally.

## Overview

Agent optimization is reserved for a future remote execution path where complete agent specifications can be optimized by Traigent Cloud. When available, this approach will be ideal when:

- You want to optimize complex agent behaviors beyond simple parameter tuning
- You're comfortable with sending agent specifications to the cloud
- You need sophisticated optimization algorithms that benefit from cloud computing
- You want to optimize across multiple agent platforms (LangChain, OpenAI, etc.)

## Key Concepts

### 1. Agent Specification

An `AgentSpecification` defines your AI agent completely:

```python
from traigent.cloud.models import AgentSpecification

agent_spec = AgentSpecification(
    id="unique-agent-id",
    name="My Agent",
    agent_type="conversational",  # or "task", "tool_use"
    agent_platform="openai",      # or "langchain", "custom"
    prompt_template="Your prompt with {variables}",
    model_parameters={
        "model": "o4-mini",
        "temperature": 0.7,
        "max_tokens": 150
    },
    persona="helpful assistant",
    guidelines=["Be concise", "Be accurate"],
    custom_tools=["search", "calculator"]  # Optional
)
```

### 2. Configuration Space

Define which parameters to optimize:

```python
configuration_space = {
    # Categorical parameters
    "model": ["o4-mini", "GPT-4o"],

    # Continuous parameters
    "temperature": (0.0, 1.0),

    # Discrete parameters
    "max_tokens": [100, 200, 300, 500],

    # Nested parameters
    "top_p": (0.5, 1.0),
    "frequency_penalty": (0.0, 0.5)
}
```

### 3. Optimization Objectives

Common objectives include:

- `"accuracy"` - How well the agent performs the task
- `"response_quality"` - Quality of generated responses
- `"cost"` - Token/API usage cost
- `"latency"` - Response time
- `"consistency"` - Consistency across similar inputs

## Implementation Steps

### Step 1: Initialize Cloud Client

The following examples are roadmap/reference examples for the future remote execution API. They should not be used as production instructions until cloud remote execution is released.

```python
from traigent.cloud.client import TraigentCloudClient

client = TraigentCloudClient(
    api_key="your-api-key",  # pragma: allowlist secret
    base_url="https://api.traigent.ai"
)
```

### Step 2: Create Agent and Dataset

```python
from traigent.evaluators.base import Dataset, EvaluationExample

# Define your agent
agent_spec = AgentSpecification(
    id="qa-agent",
    name="Question Answering Agent",
    agent_type="conversational",
    agent_platform="openai",
    prompt_template="Answer the question: {question}",
    model_parameters={"model": "o4-mini", "temperature": 0.7}
)

# Create evaluation dataset
dataset = Dataset([
    EvaluationExample(
        input_data={"question": "What is AI?"},
        expected_output="Artificial Intelligence is..."
    ),
    # Add more examples
])
```

### Step 3: Start Optimization

```python
async with client:
    response = await client.optimize_agent(
        agent_spec=agent_spec,
        dataset=dataset,
        configuration_space=configuration_space,
        objectives=["accuracy", "cost"],
        max_trials=30,
        optimization_strategy={
            "exploration_ratio": 0.3,
        },
        parallel_config={"trial_concurrency": 5}
    )

    optimization_id = response.optimization_id
```

### Step 4: Monitor Progress

```python
while True:
    status = await client.get_agent_optimization_status(optimization_id)

    print(f"Progress: {status.progress * 100:.1f}%")
    print(f"Current best: {status.current_best_metrics}")

    if status.status.value == "completed":
        break

    await asyncio.sleep(10)
```

## Configuration Mapping

Traigent automatically maps optimization parameters to your agent specification:

### Using the Configuration Mapper

```python
from traigent.agents import apply_config_to_agent

# Apply optimized configuration
optimized_config = {
    "model": "GPT-4o",
    "temperature": 0.3,
    "max_tokens": 250
}

optimized_agent = apply_config_to_agent(
    original_agent_spec,
    optimized_config
)
```

### Custom Platform Mappings

For custom platforms, define your own mappings:

```python
from traigent.agents import ParameterMapping, PlatformMapping, register_platform_mapping

custom_mapping = PlatformMapping(
    platform="custom_platform",
    parameter_mappings=[
        ParameterMapping(
            source_key="temperature",
            target_key="sampling_temperature",
            transform=lambda x: x * 100  # Convert to percentage
        ),
        ParameterMapping(
            source_key="model",
            target_key="model_name",
            validation=lambda x: x in ["model1", "model2"]
        )
    ]
)

register_platform_mapping(custom_mapping)
```

## Local Agent Execution

You can also execute agents locally using the Agent Executor framework:

```python
from traigent.agents import get_executor_for_platform

# Get executor for your platform
executor = get_executor_for_platform("openai")
await executor.initialize()

# Execute agent
result = await executor.execute(
    agent_spec=agent_spec,
    input_data={"question": "What is machine learning?"},
    config_overrides={"temperature": 0.5}  # Optional
)

print(f"Output: {result.output}")
print(f"Cost: ${result.cost:.4f}")
```

### Batch Execution

Process multiple inputs efficiently:

```python
inputs = [
    {"question": "What is AI?"},
    {"question": "What is ML?"},
    {"question": "What is DL?"}
]

results = await executor.batch_execute(
    agent_spec=agent_spec,
    input_batch=inputs,
    max_concurrent=3
)

for input_data, result in zip(inputs, results):
    print(f"Q: {input_data['question']}")
    print(f"A: {result.output}\n")
```

## Custom Specification Generator Strategies

You can extend the agent specification generation with custom strategies for specialized prompt templates and platform inference:

### Custom Prompt Template Builder

```python
from traigent.agents.specification_generator import SpecificationGenerator, PromptTemplateBuilder, FunctionAnalysis
from typing import Any

class CustomPromptBuilder:
    """Custom prompt template builder for domain-specific agents."""

    def build_template(
        self,
        function_analysis: FunctionAnalysis,
        config: dict[str, Any]
    ) -> str:
        """Build custom prompt template."""
        if function_analysis.inferred_domain == "medical":
            return self._build_medical_template(function_analysis, config)
        elif function_analysis.inferred_domain == "legal":
            return self._build_legal_template(function_analysis, config)
        else:
            # Fall back to default behavior
            return self._build_default_template(function_analysis, config)

    def _build_medical_template(self, analysis: FunctionAnalysis, config: dict[str, Any]) -> str:
        """Build medical-specific prompt template."""
        return f"""You are a medical AI assistant specializing in {analysis.name}.

Context: You provide accurate, evidence-based medical information.
Always include appropriate disclaimers about consulting healthcare professionals.

Task: {analysis.docstring or analysis.name}
Input: {{input}}

Guidelines:
- Use clear, non-technical language when appropriate
- Include evidence-based recommendations
- Always mention limitations of AI medical advice
- Suggest consulting qualified healthcare providers for personalized advice

Response:"""

    def _build_default_template(self, analysis: FunctionAnalysis, config: dict[str, Any]) -> str:
        """Default template building logic."""
        return f"""You are an AI assistant specialized in {analysis.name}.

Task: {analysis.docstring or f'Process {analysis.name} requests'}
Input: {{input}}

Response:"""

# Use custom builder
custom_builder = CustomPromptBuilder()
generator = SpecificationGenerator(prompt_builder=custom_builder)
```

### Custom Platform Inference Strategy

```python
from traigent.agents.specification_generator import SpecificationGenerator, PlatformInferenceStrategy
from typing import Any

class CustomPlatformStrategy:
    """Custom platform inference strategy."""

    def infer_platform(self, config: dict[str, Any]) -> str:
        """Infer platform with custom logic."""
        # Check for custom model families
        model_config = config.get("configuration_space", {}).get("model", [])
        if isinstance(model_config, list):
            for model in model_config:
                if "claude-3" in str(model).lower():
                    return "anthropic"
                elif "medical-llm" in str(model).lower():
                    return "custom_medical"
                elif "legal-llm" in str(model).lower():
                    return "custom_legal"

        # Check for domain-specific requirements
        objectives = config.get("objectives", [])
        if "medical_accuracy" in objectives:
            return "medical_platform"
        elif "legal_compliance" in objectives:
            return "legal_platform"

        # Default fallback
        return "openai"

# Use custom strategy
custom_strategy = CustomPlatformStrategy()
generator = SpecificationGenerator(platform_strategy=custom_strategy)
```

### Combining Custom Strategies

```python
# Use both custom strategies together
generator = SpecificationGenerator(
    prompt_builder=CustomPromptBuilder(),
    platform_strategy=CustomPlatformStrategy()
)

# Generate specification with custom strategies
agent_spec = generator.from_optimized_function(
    func=my_medical_function,
    optimization_config=my_config
)
```

### Strategy Benefits

- **Extensibility**: Add new prompt templates without modifying core code
- **Specialization**: Domain-specific prompt engineering and platform selection
- **Testability**: Strategies can be tested independently
- **Maintainability**: Clear separation of concerns
- **Backward Compatibility**: Default strategies ensure existing code continues to work

## Platform-Specific Considerations

### OpenAI Platform

- Supports models: o4-mini, gpt-4, etc.
- Key parameters: temperature, max_tokens, top_p, frequency_penalty
- Cost calculation based on token usage

### LangChain Platform

- Supports various LLM backends
- Additional features: tools, memory, chains
- Requires langchain package installation

### Custom Platforms

1. Create an executor:

```python
from traigent.agents import AgentExecutor

class CustomAgentExecutor(AgentExecutor):
    async def _platform_initialize(self):
        # Initialize your platform
        pass

    async def _execute_agent(self, agent_spec, input_data, config):
        # Execute on your platform
        return {
            "output": "response",
            "tokens_used": 100,
            "cost": 0.01
        }
```

2. Register it:

```python
from traigent.agents import PlatformRegistry

PlatformRegistry.register_executor("custom", CustomAgentExecutor)
```

## Best Practices

### 1. Dataset Design

- Include diverse examples covering edge cases
- Balance positive and negative examples
- Keep examples concise but representative
- Include metadata for better analysis

### 2. Configuration Space

- Start with wider ranges, refine based on results
- Include only parameters that meaningfully affect performance
- Consider parameter interactions
- Use appropriate parameter types (categorical vs continuous)

### 3. Optimization Strategy

```python
optimization_strategy = {
    # Exploration vs Exploitation
    "exploration_ratio": 0.3,  # 30% exploration, 70% exploitation

    # Early stopping
    "early_stopping": True,
    "patience": 5,  # Stop if no improvement for 5 trials

    # Resource management
    "max_cost_budget": 10.0,  # Maximum $10 for optimization

    # Smart features
    "adaptive_sample_size": True,  # Increase sample size for promising configs
    "use_prior_knowledge": True    # Use meta-learning from similar tasks
}

# Parallel execution is configured separately:
# parallel_config={"trial_concurrency": 3}  # Run 3 trials concurrently
```

### 4. Error Handling

```python
try:
    result = await client.optimize_agent(...)
except CloudServiceError as e:
    if "quota exceeded" in str(e):
        print("Optimization quota exceeded, using fallback")
        # Use local optimization or cached results
    else:
        raise
```

## Comparison: Agent vs Interactive Optimization

| Feature            | Agent Optimization (Model 2)         | Interactive Optimization (Model 1)  |
| ------------------ | ------------------------------------ | ----------------------------------- |
| Current Status     | Not implemented                      | Requires a real remote guidance service |
| Execution Location | Future cloud service                 | Local client                        |
| Data Privacy       | Agent spec sent to cloud             | Only metadata sent                  |
| Optimization Speed | Future remote execution path         | Local execution speed               |
| Cost               | Higher (cloud compute)               | Lower (local compute)               |
| Complexity         | Simpler setup                        | More complex integration            |
| Use Case           | Standard agents, less sensitive data | Custom functions, sensitive data    |

## Advanced Features

### 1. Multi-Objective Optimization

```python
response = await client.optimize_agent(
    agent_spec=agent_spec,
    dataset=dataset,
    configuration_space=config_space,
    objectives=["accuracy", "cost", "latency"],
    multi_objective_strategy={
        "method": "pareto",  # Find Pareto optimal solutions
        "weights": {"accuracy": 0.5, "cost": 0.3, "latency": 0.2}
    }
)
```

### 2. Warm Start Optimization

```python
# Start from a known good configuration
response = await client.optimize_agent(
    agent_spec=agent_spec,
    dataset=dataset,
    configuration_space=config_space,
    optimization_strategy={
        "warm_start": True,
        "initial_configs": [
            {"model": "GPT-4o", "temperature": 0.3},
            {"model": "o4-mini", "temperature": 0.7}
        ]
    }
)
```

### 3. Constraint-Based Optimization

```python
response = await client.optimize_agent(
    agent_spec=agent_spec,
    dataset=dataset,
    configuration_space=config_space,
    constraints={
        "max_cost_per_call": 0.01,      # Max $0.01 per API call
        "max_latency": 2.0,              # Max 2 second response time
        "min_accuracy": 0.8              # Minimum 80% accuracy
    }
)
```

## Troubleshooting

### Common Issues

1. **"Platform not supported"**

   - Check available platforms: `get_supported_platforms()`
   - Register custom platform if needed

2. **"Configuration validation failed"**

   - Use `validate_config_compatibility()` before optimization
   - Check parameter ranges and types

3. **"Optimization timeout"**

   - Increase max_trials or adjust early_stopping
   - Check if objectives are achievable

4. **"Cost budget exceeded"**
   - Reduce dataset size or max_trials
   - Use more efficient models in config space

### Debug Mode

Enable detailed logging:

```python
import logging
from traigent.utils.logging import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)

# Now run optimization with detailed logs
```

## Next Steps

- See the [examples catalog](../../docs/examples/START_HERE.md) for runnable examples
- Review [Execution Modes](../guides/execution-modes.md) for local/cloud/hybrid context
- Read [Architecture Overview](../architecture/ARCHITECTURE.md) for system design details
