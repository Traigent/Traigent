# Enhanced Features vs Regular Mode in Framework Override

## Overview

The FrameworkOverrideManager has two modes:
1. **Regular Mode**: Uses hardcoded parameter mappings for known frameworks
2. **Enhanced Mode**: Adds dynamic discovery, validation, and version compatibility

## Regular Mode (use_enhanced_features=False)

In regular mode, the system only uses the hardcoded mappings defined in `PARAMETER_MAPPINGS`:

```python
PARAMETER_MAPPINGS = {
    "openai.OpenAI": {
        "model": "model",
        "temperature": "temperature",
        "max_tokens": "max_tokens",
        # ... etc
    },
    # ... other frameworks
}
```

### How it works:
1. When a framework is initialized (e.g., `OpenAI()`), we intercept the constructor
2. We check if we have a Traigent config with parameters
3. We map Traigent parameters to framework parameters using the hardcoded mapping
4. We inject the mapped parameters into the constructor

### Example:
```python
# Traigent config has: {"model": "gpt-4", "temperature": 0.5}
# OpenAI expects: model="gpt-4", temperature=0.5
# Mapping is straightforward: model -> model, temperature -> temperature
```

### Limitations:
- Only works with frameworks we've explicitly added to PARAMETER_MAPPINGS
- Can't handle parameter name variations (e.g., `max_tokens` vs `max_new_tokens`)
- No version awareness (parameter names might change between versions)
- No validation of parameter types or values

## Enhanced Mode (use_enhanced_features=True) - Default

Enhanced mode adds three powerful components:

### 1. ParameterDiscovery

**Purpose**: Dynamically discover parameter names from any framework without hardcoding

**How it works**:
```python
# Instead of hardcoding, we inspect the class
params = discovery.discover_init_parameters(OpenAI)
# Returns: {"api_key": <Parameter>, "model": <Parameter>, ...}

# We can also discover method parameters
method_params = discovery.discover_method_parameters(client, "chat.completions.create")
```

**Auto-discovery features**:
- Uses Python's `inspect` module to analyze constructors and methods
- Creates universal mappings for common parameter variations:
  ```python
  universal_mapping = {
      "model": ["model", "model_name", "model_id", "engine"],
      "max_tokens": ["max_tokens", "max_length", "max_new_tokens", "max_tokens_to_sample"],
      # ... etc
  }
  ```
- Finds similar parameters using fuzzy matching (e.g., "temperature" matches "temp")

### 2. ParameterValidator

**Purpose**: Ensure parameters are valid before injection

**Features**:
- Type checking: Validates parameter types match expected signatures
- Value constraints: Checks min/max values (e.g., temperature 0-2)
- Type conversion: Attempts to convert compatible types
- Sanitization: Removes incompatible parameters

**Example**:
```python
# Validates that temperature is a float between 0 and 2
# Converts "0.5" (string) to 0.5 (float) if needed
# Warns if value is out of range
```

### 3. VersionCompatibilityManager

**Purpose**: Handle parameter changes across SDK versions

**Features**:
- Version-specific mappings:
  ```python
  # OpenAI v1.0: uses "max_tokens"
  # OpenAI v2.0 (hypothetical): might use "max_completion_tokens"
  ```
- Deprecation warnings for old parameters
- Parameter migration between versions
- Automatic version detection

**Example scenario**:
```python
# User has OpenAI v0.28 (old version)
# Old version uses "engine" parameter
# New version uses "model" parameter
# Version manager automatically maps "model" -> "engine" for compatibility
```

## Key Differences

| Feature | Regular Mode | Enhanced Mode |
|---------|--------------|---------------|
| **Parameter Discovery** | Hardcoded only | Dynamic + hardcoded |
| **New Frameworks** | Must manually add | Auto-discovers parameters |
| **Parameter Variations** | Exact match only | Fuzzy matching + variations |
| **Type Safety** | None | Type validation & conversion |
| **Version Handling** | None | Version-aware mappings |
| **Error Recovery** | Basic | Multiple fallback strategies |

## When Enhanced Features Shine

### 1. Unknown Framework Support
```python
# New framework not in PARAMETER_MAPPINGS
class NewLLMClient:
    def __init__(self, model_name, temp, max_length):
        pass

# Enhanced mode can:
# - Discover parameters: model_name, temp, max_length
# - Map intelligently: model -> model_name, temperature -> temp, max_tokens -> max_length
# - Work without any hardcoded mapping!
```

### 2. Parameter Name Variations
```python
# Different frameworks use different names for the same concept:
# OpenAI: max_tokens
# Anthropic: max_tokens_to_sample
# HuggingFace: max_new_tokens
# Some custom framework: max_length

# Enhanced mode handles all these automatically
```

### 3. Version Compatibility
```python
# User upgrades OpenAI SDK from v0.28 to v1.0
# Parameter names changed
# Enhanced mode detects version and applies correct mappings
```

## Fallback Strategies in Enhanced Mode

The `_create_override_wrapper` has two strategies:

1. **Direct Mapping**: Use discovered or hardcoded mappings
2. **Exact Name Fallback**: If no mapping found, try exact parameter names

```python
# Strategy 1: Use mapping
# Traigent: "model" -> Framework: "model_name" (via mapping)

# Strategy 2: Fallback
# Traigent: "custom_param" -> Framework: "custom_param" (no mapping, use exact name)
```

This ensures maximum compatibility even with unknown parameters.

## Performance Considerations

- **Regular Mode**: Faster, no discovery overhead
- **Enhanced Mode**: Slight overhead for discovery (happens once per class)
- Both modes cache discovered mappings for reuse

## Recommendation

Use Enhanced Mode (default) because:
1. It falls back to regular mode behavior when discovery fails
2. The overhead is minimal (milliseconds)
3. It provides much better compatibility and error handling
4. It future-proofs your code against SDK changes
