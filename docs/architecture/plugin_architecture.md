# Traigent Plugin Architecture Documentation

## Overview

The Traigent integration system has been redesigned with a **hybrid plugin + configuration architecture** that provides:

- **Type Safety**: Plugin interfaces with compile-time checking
- **Flexibility**: YAML configuration overrides without code changes
- **Consistency**: Enforced structure across all integrations
- **Extensibility**: Easy addition of new frameworks and vendors
- **Version Management**: Support for multiple versions of the same package

## Architecture Components

### 1. Base Plugin System (`base_plugin.py`)

The foundation of the architecture is the `IntegrationPlugin` abstract base class that all integration plugins must extend:

```python
class IntegrationPlugin(ABC):
    def __init__(self, config_path: Optional[Path] = None)

    @abstractmethod
    def _get_metadata() -> PluginMetadata

    @abstractmethod
    def _get_default_mappings() -> Dict[str, str]

    @abstractmethod
    def _get_validation_rules() -> Dict[str, ValidationRule]

    @abstractmethod
    def get_target_classes() -> List[str]

    @abstractmethod
    def get_target_methods() -> Dict[str, List[str]]
```

### 2. Plugin Registry (`plugin_registry.py`)

A singleton registry manages all plugins throughout the application lifecycle:

```python
registry = PluginRegistry()

# Register a plugin
registry.register(plugin)

# Get plugin for a class
plugin = registry.get_plugin_for_class("openai.OpenAI")

# Get plugins for a package
plugins = registry.get_plugins_for_package("langchain")
```

### 3. Framework Override Manager (`framework_override.py`)

The manager uses plugins to intercept and override framework calls:

```python
manager = FrameworkOverrideManager()

# Activate overrides for specific frameworks
manager.activate_overrides(["openai", "anthropic"])

# Use as context manager
with manager.override_context(["langchain"]):
    # Framework calls are intercepted here
    pass
```

## Creating a New Plugin

### Step 1: Create the Plugin Class

```python
from traigent.integrations.base_plugin import IntegrationPlugin, PluginMetadata, ValidationRule

class MyFrameworkPlugin(IntegrationPlugin):
    def _get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="myframework",
            version="1.0.0",
            supported_packages=["myframework", "myframework-extras"],
            description="MyFramework integration plugin"
        )

    def _get_default_mappings(self) -> Dict[str, str]:
        return {
            "model": "model_name",
            "temperature": "temp",
            "max_tokens": "max_length"
        }

    def _get_validation_rules(self) -> Dict[str, ValidationRule]:
        return {
            "temperature": ValidationRule(min_value=0.0, max_value=1.0),
            "max_tokens": ValidationRule(min_value=1, max_value=4096)
        }

    def get_target_classes(self) -> List[str]:
        return ["myframework.Client", "myframework.AsyncClient"]

    def get_target_methods(self) -> Dict[str, List[str]]:
        return {
            "myframework.Client": ["generate", "complete"],
            "myframework.AsyncClient": ["agenerate", "acomplete"]
        }
```

### Step 2: Register the Plugin

```python
# Automatic registration (place in llms/ directory)
# The plugin will be discovered automatically

# Manual registration
from traigent.integrations.plugin_registry import get_registry

registry = get_registry()
plugin = MyFrameworkPlugin()
registry.register(plugin)
```

### Step 3: Optional Configuration File

Create a YAML configuration file to customize the plugin without code changes:

```yaml
# config/plugins/myframework.yaml
metadata:
  description: "Custom MyFramework settings"

mappings:
  # Override or extend parameter mappings
  custom_param: "framework_specific_param"
  api_key: "auth_token"

validation:
  temperature:
    min_value: 0.1
    max_value: 0.9
  custom_param:
    required: true
    allowed_values: ["option1", "option2", "option3"]
```

## Parameter Mapping Flow

1. **User Code**: Calls framework with original parameters
2. **Interception**: Plugin intercepts the call
3. **Configuration Loading**: TraigentConfig provides optimization parameters
4. **Mapping**: Plugin maps Traigent parameters to framework parameters
5. **Validation**: Plugin validates parameters against rules
6. **Override**: Modified parameters passed to original framework

```python
# User code
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello"}]
)

# Behind the scenes with TraigentConfig(model="gpt-4", temperature=0.5)
# 1. OpenAIPlugin intercepts the call
# 2. Maps: model="gpt-4", temperature=0.5
# 3. Validates: model is in allowed list, temperature in range
# 4. Calls original with: model="gpt-4", temperature=0.5, messages=[...]
```

## Built-in Plugins

### OpenAI Plugin (`openai_plugin.py`)

- **Packages**: `openai`
- **Models**: GPT-4, GPT-3.5, Davinci, etc.
- **Special Features**: Function calling, streaming, vision

### LangChain Plugin (`langchain_plugin.py`)

- **Packages**: `langchain`, `langchain-*` sub-packages
- **Special Features**: Dynamic discovery, RAG parameters, chain configuration
- **Auto-Discovery**: Scans for installed LangChain packages

### Anthropic Plugin (`anthropic_plugin.py`)

- **Packages**: `anthropic`
- **Models**: Claude 3, Claude 2, Claude Instant
- **Special Features**: System prompts, tool use

## Advanced Features

### Parameter Normalizer Utility

The `ParameterNormalizer` class in `utils/parameter_normalizer.py` provides cross-framework parameter name conversion:

```python
from traigent.integrations.utils import Framework, normalize_params, get_normalizer

# Direct conversion between frameworks
params = {"model_name": "gpt-4", "streaming": True, "max_tokens": 100}
normalized = normalize_params(params, "langchain", "openai")
# Result: {"model": "gpt-4", "stream": True, "max_tokens": 100}

# Or use the normalizer instance for more control
normalizer = get_normalizer()

# Convert to canonical Traigent format first
canonical = normalizer.to_canonical(params, Framework.LANGCHAIN)

# Then convert to target framework
openai_params = normalizer.from_canonical(canonical, Framework.OPENAI)
```

**Supported Frameworks** (10 total):

- `TRAIGENT` - Canonical parameter names
- `OPENAI`, `ANTHROPIC`, `LANGCHAIN`, `LLAMAINDEX`
- `GEMINI`, `BEDROCK`, `AZURE_OPENAI`
- `COHERE`, `HUGGINGFACE`

**Parameters with Auto-Conversion**:

| Canonical | OpenAI | LangChain | Gemini | Bedrock | HuggingFace |
|-----------|--------|-----------|--------|---------|-------------|
| `model` | `model` | `model_name` | `model_name` | `model_id` | `model_id` |
| `max_tokens` | `max_tokens` | `max_tokens` | `max_output_tokens` | `max_tokens` | `max_new_tokens` |
| `stop` | `stop` | `stop` | `stop_sequences` | `stop_sequences` | `stop_sequences` |
| `stream` | `stream` | `streaming` | `stream` | `stream` | `stream` |
| `top_p` | `top_p` | `top_p` | `top_p` | `top_p` | `top_p` |
| `top_k` | `top_k` | `top_k` | `top_k` | `top_k` | `top_k` |

### Dynamic Discovery

The LangChain plugin demonstrates dynamic discovery of components:

```python
def _discover_langchain_components(self):
    packages_to_scan = [
        ("langchain_openai", ["ChatOpenAI", "OpenAI"]),
        ("langchain_anthropic", ["ChatAnthropic"]),
        # ... more packages
    ]

    for package_name, class_names in packages_to_scan:
        try:
            module = importlib.import_module(package_name)
            # Register discovered classes
        except ImportError:
            continue
```

### Priority System

Plugins have priority levels to resolve conflicts:

```python
class IntegrationPriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3
```

Higher priority plugins override lower priority ones for the same classes.

### Version Management

Support multiple versions through:

1. **Versioned Plugins**: `OpenAIV1Plugin`, `OpenAIV2Plugin`
2. **Configuration Files**: `openai_v1.yaml`, `openai_v2.yaml`
3. **Runtime Selection**: Based on detected package version

### Custom Validation

Plugins can implement custom validation methods:

```python
def _get_validation_rules(self):
    return {
        "seed": ValidationRule(custom_validator="_validate_seed")
    }

def _validate_seed(self, param_name: str, value: Any) -> List[str]:
    errors = []
    if not isinstance(value, int):
        errors.append(f"{param_name} must be an integer")
    return errors
```

## Testing Plugins

```python
import pytest
from traigent.integrations.base_plugin import IntegrationPlugin

class TestMyPlugin:
    def test_parameter_mappings(self, my_plugin):
        mappings = my_plugin.get_parameter_mappings()
        assert "model" in mappings
        assert mappings["model"] == "model_name"

    def test_validation(self, my_plugin, config):
        # Valid config
        assert my_plugin.validate_config(valid_config) is True

        # Invalid config
        with pytest.raises(Exception):
            my_plugin.validate_config(invalid_config)

    def test_overrides(self, my_plugin, config):
        kwargs = {"existing": "value"}
        overridden = my_plugin.apply_overrides(kwargs, config)
        assert "model_name" in overridden
```

## Migration Guide

### From Old System to Plugin System

**Old System** (Duplicate mappings in multiple files):
```python
# In framework_override.py
PARAMETER_MAPPINGS = {
    "openai.OpenAI": {"model": "model", ...}
}

# In integrations/llms/openai.py
OPENAI_MAPPINGS = {"model": "model", ...}
```

**New System** (Single plugin owns mappings):
```python
# In integrations/llms/openai_plugin.py
class OpenAIPlugin(IntegrationPlugin):
    def _get_default_mappings(self):
        return {"model": "model", ...}
```

### Benefits of Migration

1. **Single Source of Truth**: Each integration owns its mappings
2. **Type Safety**: Abstract base class enforces structure
3. **Flexibility**: YAML overrides without code changes
4. **Consistency**: All integrations follow same pattern
5. **Discoverability**: Registry provides central access
6. **Testability**: Plugins are easily unit tested
7. **Maintainability**: Clear separation of concerns

## Best Practices

1. **Keep Plugins Focused**: One plugin per vendor/framework
2. **Use Configuration Files**: For environment-specific settings
3. **Implement Validation**: Catch errors early with validation rules
4. **Document Mappings**: Clear comments for parameter mappings
5. **Test Thoroughly**: Unit tests for all plugin functionality
6. **Version Carefully**: Plan for package version changes
7. **Priority Wisely**: Set appropriate priority levels

## Future Enhancements

1. **Plugin Marketplace**: Community-contributed plugins
2. **Auto-Generation**: Generate plugins from API specifications
3. **Hot Reload**: Reload plugins without restart
4. **Performance Metrics**: Track plugin overhead
5. **Conflict Resolution**: Automatic resolution strategies
6. **Plugin Composition**: Combine multiple plugins
7. **Schema Validation**: JSON Schema for configuration files
