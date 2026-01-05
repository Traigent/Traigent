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

---

## Lazy Import Architecture

### Modules with Lazy Cloud Imports

The following modules use lazy imports to allow the base package to function without cloud dependencies installed. Each module gracefully degrades when `traigent.cloud` is unavailable.

| Module | Lazy Import Pattern | Fallback Behavior |
| ------ | ------------------- | ------------------ |
| `traigent.core.orchestrator` | `ModuleNotFoundError` with `.name` check | Disables backend sync, local-only mode |
| `traigent.core.backend_session_manager` | `TYPE_CHECKING` for type hints | No runtime dependency on cloud |
| `traigent.core.tracing` | Plugin import with embedded fallback | Full tracing via embedded implementation |
| `traigent.utils.local_analytics` | Try/except for `MIN_TOKEN_LENGTH` | Local fallback constant |
| `traigent.agents.executor` | `TYPE_CHECKING` for `AgentSpecification` | Type hints only |
| `traigent.agents.config_mapper` | `TYPE_CHECKING` for cloud models | Type hints only |
| `traigent.agents.specification_generator` | `TYPE_CHECKING` for cloud models | Type hints only |
| `traigent.agents.platforms` | Runtime try/except for cloud auth | Platform-specific auth disabled |
| `traigent.optimizers.interactive_optimizer` | Runtime try/except for cloud models | Local optimization only |
| `traigent.traigent_client` | Runtime try/except for cloud clients | Local integration only |

### Import Patterns Used

#### 1. TYPE_CHECKING Pattern (Preferred for Type Hints)

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from traigent.cloud.models import SomeCloudType

def func(param: SomeCloudType) -> None:  # Works due to PEP 563
    ...
```

#### 2. ModuleNotFoundError with `.name` Check (For Runtime Imports)

```python
try:
    from traigent.cloud.backend_client import BackendIntegratedClient
    _CLOUD_AVAILABLE = True
except ModuleNotFoundError as err:
    # CRITICAL: Check .name to distinguish between:
    # - "traigent.cloud not installed" (graceful degradation)
    # - "broken dependency like boto3" (hard error, re-raise)
    if err.name and err.name.startswith("traigent.cloud"):
        _CLOUD_AVAILABLE = False
    else:
        raise  # Re-raise for broken transitive dependencies
```

#### 3. Plugin Stub with Embedded Fallback

```python
# traigent/core/tracing.py
try:
    from traigent_tracing import get_tracer, optimization_session_span, ...
    _PLUGIN_AVAILABLE = True
except ImportError:
    _PLUGIN_AVAILABLE = False
    # Embedded fallback implementation (~400 lines)
    def get_tracer(): ...
```

---

## Tracing Stub/Plugin Architecture

### Design Decision: Embedded Fallback

The tracing module (`traigent/core/tracing.py`) includes a full embedded fallback implementation rather than minimal no-op stubs. This ensures:

1. **Zero-install tracing**: Users can enable tracing without installing extra packages
2. **Backward compatibility**: Existing code continues to work
3. **Gradual adoption**: Users can upgrade to the plugin for enhanced features

### Contract Between Stub and Plugin

Both the embedded fallback and `traigent-tracing` plugin MUST export identical symbols:

```python
__all__ = [
    "TRACING_ENABLED",
    "OTEL_AVAILABLE",
    "SecureIdGenerator",
    "get_tracer",
    "set_test_context",
    "clear_test_context",
    "get_test_context",
    "optimization_session_span",
    "trial_span",
    "record_trial_result",
    "record_optimization_complete",
    "example_evaluation_span",
    "record_example_result",
]
```

### Preventing Drift

To prevent the stub and plugin from diverging:

1. **Shared test suite**: Both implementations run against `tests/unit/core/test_tracing.py`
2. **Type checking**: Both use identical type signatures
3. **CI validation**: Tests run with and without the plugin installed

### Future: Shared Protocol (Recommended)

Consider extracting a shared `TracingProtocol` to enforce API parity:

```python
# traigent/protocols/tracing.py
from typing import Protocol, ContextManager, Any

class TracingProtocol(Protocol):
    def get_tracer(self) -> Any | None: ...
    def optimization_session_span(
        self, function_name: str, ...
    ) -> ContextManager[Any | None]: ...
    # ... etc
```

---

## Security Model for Plugin Discovery

### Entry Point Discovery Threat Model

The plugin system uses Python entry points for automatic discovery. This section documents security considerations.

#### Trust Boundary

**Entry points are loaded from installed packages.** If an attacker can install a malicious package, they have already compromised the Python environment. The plugin system does NOT add new attack surface beyond pip/package installation.

#### Mitigations Implemented

1. **Class Type Validation**

   ```python
   if inspect.isclass(plugin_class) and issubclass(plugin_class, TraigentPlugin):
       # Only load valid TraigentPlugin subclasses
   ```

2. **Safe Module Path Validation** (for `load_plugin_from_module`)

   ```python
   # Only allow known prefixes
   allowed_prefixes = ["traigent_plugins", "traigent.plugins.contrib", "custom_plugins"]

   # Block dangerous module names
   unsafe_patterns = ["os", "sys", "subprocess", "importlib", ...]
   ```

3. **Safe Directory Validation** (for `load_plugins_from_directory`)

   ```python
   # Only load from approved directories
   allowed_dirs = [
       Path.home() / ".traigent" / "plugins",
       Path.cwd() / "traigent_plugins",
       Path(__file__).parent / "contrib",
   ]
   ```

4. **Version Compatibility Check**

   ```python
   # Plugins must declare compatible Traigent version
   if not _is_version_compatible(plugin.traigent_version, current_version):
       raise PluginVersionError(...)
   ```

#### Assumptions

- Users trust packages installed via pip
- The `traigent.plugins` entry point group is not shared with untrusted packages
- Plugin authors follow semantic versioning for `traigent_version`

#### Recommendations for Plugin Authors

1. **Pin Traigent Version**: Use `traigent_version = "0.9.0"` to declare minimum required version
2. **Minimal Dependencies**: Only import what you need in `initialize()`
3. **Safe Cleanup**: Handle errors gracefully in `cleanup()` methods
4. **No Dynamic Code**: Avoid `eval()`, `exec()`, or dynamic imports

#### Out of Scope Threats

The plugin system does NOT protect against:

- Malicious packages installed via pip (supply chain attacks)
- Plugins with known vulnerabilities
- Denial of service via slow `initialize()` methods
- Information disclosure via plugin logging

---

## Thread Safety

### Plugin Discovery Synchronization

The registry uses a combination of locks and events for thread-safe discovery:

```python
class PluginRegistry:
    def __init__(self):
        self._discovery_lock = threading.RLock()          # Serializes discovery
        self._discovery_complete = threading.Event()       # Signals completion
        self._discovery_thread_id: int | None = None       # Tracks discovery thread
```

#### Behavior

1. **First caller** acquires the lock and performs discovery
2. **Concurrent callers** wait on `_discovery_complete` event
3. **Re-entrant calls** (same thread) bypass waiting via thread ID check
4. All callers see complete registry state after discovery

#### Why This Matters

Plugins may call `has_feature()` during their `initialize()` method:

```python
class MyPlugin(FeaturePlugin):
    def initialize(self):
        # This would deadlock without re-entrancy support
        if registry.has_feature(FEATURE_TRACING):
            self.enable_tracing()
```
