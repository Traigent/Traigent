# Traigent Plugin Architecture

## Overview

Traigent uses a plugin architecture to enable a tiered feature model. The base package (`traigent`) provides core optimization functionality, while optional plugins enable advanced features.

## Feature Tiers

### Base Package (Always Available)

The core `traigent` package includes:

- **Optimization Algorithms**: TPE, Random, Grid search
- **Injection Modes**: Context, Parameter, Attribute
- **Config Space**: Range, Choices, Uniform
- **Stop Conditions**: MaxTrials, MaxTotalExamples
- **Evaluators**: LocalEvaluator
- **Basic Metrics**: Accuracy, Latency, Cost

### ML Bundle (`pip install traigent[ml]`)

Recommended for production ML workloads:

- **Advanced Algorithms**: CMA-ES, NSGA-II (multi-objective)
- **Parallel Execution**: Trial concurrency, example concurrency
- **BatchInvoker/StreamingInvoker**: Advanced invocation patterns
- **Enhanced Metrics**: Harmonic aggregation, Chebyshev scalarization

### Cloud Bundle (`pip install traigent[cloud]`)

For cloud-native deployments:

- **Cloud Execution Mode**: Remote optimization
- **Session Management**: Backend sync, trial persistence
- **Advanced Security**: JWT authentication, encryption

### Enterprise Bundle (`pip install traigent[enterprise]`)

All features combined for enterprise deployments.

## Plugin System

### Plugin Discovery

Plugins register via Python entry points:

```toml
# In plugin's pyproject.toml
[project.entry-points."traigent.plugins"]
my-plugin = "my_plugin:MyPlugin"
```

### Creating a Plugin

```python
from traigent.plugins import FeaturePlugin, FEATURE_TRACING

class TracingPlugin(FeaturePlugin):
    @property
    def name(self) -> str:
        return "traigent-tracing"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def description(self) -> str:
        return "OpenTelemetry tracing for Traigent"

    def provides_features(self) -> list[str]:
        return [FEATURE_TRACING]

    def initialize(self) -> None:
        # Setup tracing
        pass
```

### Checking Feature Availability

```python
from traigent.plugins import has_feature, FEATURE_PARALLEL

if has_feature(FEATURE_PARALLEL):
    # Use parallel features
    pass
else:
    # Graceful fallback
    pass
```

### Feature Flags

Available feature flags:

| Constant | Feature | Default Plugin |
|----------|---------|----------------|
| `FEATURE_PARALLEL` | Parallel execution | traigent-parallel |
| `FEATURE_MULTI_OBJECTIVE` | Multi-objective optimization | traigent-mo |
| `FEATURE_SEAMLESS` | Seamless injection | traigent-seamless |
| `FEATURE_CLOUD` | Cloud execution mode | traigent-cloud |
| `FEATURE_ADVANCED_ALGORITHMS` | CMA-ES, NSGA-II | traigent-algorithms |
| `FEATURE_TRACING` | OpenTelemetry tracing | traigent-tracing |
| `FEATURE_ANALYTICS` | Advanced analytics | traigent-analytics |
| `FEATURE_INTEGRATIONS` | Framework integrations | traigent-integrations |

## Graceful Degradation

When a feature is not available, Traigent raises `FeatureNotAvailableError` with helpful installation instructions:

```python
from traigent.utils.exceptions import FeatureNotAvailableError

# Raised automatically when feature not available
FeatureNotAvailableError(
    "Parallel execution",
    plugin_name="traigent-parallel",
    install_hint="pip install traigent[ml]"
)
```

## Migration Guide

### Breaking Changes

1. **Default Algorithm**: Changed from `"bayesian"` to `"tpe"`
   - TPE is always available with Optuna
   - Use `algorithm="bayesian"` explicitly if needed

2. **auto_override_frameworks**: Default changed to `False`
   - Requires `traigent-integrations` plugin for framework overrides
   - Set `auto_override_frameworks=True` explicitly when using LangChain/OpenAI

### Deprecation Timeline

- **v0.9.x**: Current behavior with plugin infrastructure
- **v1.0.0**: Full plugin extraction (breaking)
- **v1.1.0**: Remove deprecated stub modules

## Plugin Development

See [plugins/traigent-tracing](../plugins/traigent-tracing/) for a complete example plugin.

### Plugin Structure

```
traigent-my-plugin/
├── pyproject.toml          # Entry point registration
├── README.md               # Plugin documentation
└── traigent_my_plugin/
    ├── __init__.py         # Plugin class and exports
    └── ...                 # Plugin implementation
```

### Testing Plugins

```python
from traigent.plugins import get_plugin_registry

registry = get_plugin_registry()
assert registry.has_feature(FEATURE_MY_FEATURE)
```
