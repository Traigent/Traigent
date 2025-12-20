# Story 2.8: Auto-Discovery with Sensible Defaults

Status: done

## Story

As an **ML Engineer**,
I want the system to provide sensible default ranges when auto-discovering parameters,
so that I can start optimizing immediately without manual configuration.

## Acceptance Criteria

1. **AC1: Temperature Default Range**
   - **Given** an OpenAIGenerator with `temperature` parameter
   - **When** auto-discovery runs
   - **Then** the default range is [0.0, 2.0] (OpenAI's valid range)

2. **AC2: Top-K Default Range**
   - **Given** a retriever with `top_k` parameter
   - **When** auto-discovery runs
   - **Then** the default range is [1, 100] with default value matching the component's current setting
   - **Note:** Must align with existing `TVAR_SEMANTICS["top_k"]` in introspection.py (range: 1-100)

3. **AC3: Model Choices from Provider Catalog**
   - **Given** a known model parameter (e.g., `generator.model`)
   - **When** auto-discovery runs with provider catalog available
   - **Then** the choices include common models from that provider's hardcoded catalog

4. **AC4: Offline Fallback for Model Parameter**
   - **Given** a model parameter when provider catalog is unavailable (offline)
   - **When** auto-discovery runs
   - **Then** the parameter uses the component's current value as the single choice
   - **And** a warning is logged suggesting manual specification of model choices

## Tasks / Subtasks

- [x] Task 1: Define default range catalog for common parameters (AC: #1, #2)
  - [x] 1.1 Used existing `TVAR_SEMANTICS` (already has temperature, top_k, etc.)
  - [x] 1.2 Verified temperature: [0.0, 2.0] already present
  - [x] 1.3 Verified top_k: [1, 100] already present
  - [x] 1.4 Verified top_p: [0.0, 1.0] already present
  - [x] 1.5 Verified max_tokens: [1, 4096] already present

- [x] Task 2: Define provider model catalogs (AC: #3)
  - [x] 2.1 Created `MODEL_CATALOGS` mapping by provider
  - [x] 2.2 Added OpenAI catalog: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo, o1, o1-mini
  - [x] 2.3 Added Anthropic catalog: claude-3-5-sonnet-latest, claude-3-5-haiku-latest, claude-3-opus-latest
  - [x] 2.4 Added Azure OpenAI, Cohere, and Google catalogs
  - [x] 2.5 Documented catalogs with update guidance

- [x] Task 3: Enhance introspection to apply defaults (AC: #1, #2, #3)
  - [x] 3.1 `TVAR_SEMANTICS` already detects known parameter names
  - [x] 3.2 `_infer_tvar_semantics()` applies default ranges from catalog
  - [x] 3.3 Created `PROVIDER_DETECTION` mapping from component class names to providers
  - [x] 3.4 Added `_get_model_choices()` to apply model choices from provider catalog

- [x] Task 4: Implement offline fallback (AC: #4)
  - [x] 4.1 `_get_model_choices()` uses current value as single choice when provider unknown
  - [x] 4.2 Logs warning with suggestion to specify choices manually
  - [x] 4.3 Sets `non_tunable_reason` to explain fallback

- [x] Task 5: Add comprehensive tests (AC: #1, #2, #3, #4)
  - [x] 5.1 Test temperature gets [0.0, 2.0] range via TVAR_SEMANTICS
  - [x] 5.2 Test top_k gets [1, 100] range via TVAR_SEMANTICS
  - [x] 5.3 Test OpenAI model parameter gets OpenAI choices
  - [x] 5.4 Test unknown provider falls back to current value
  - [x] 5.5 Test warning is logged for fallback

## Dev Notes

### Architecture Context

This is **Story 2.8** in Epic 2 (Configuration Space & TVL). It enhances auto-discovery to provide practical default ranges, reducing manual configuration.

**Current State (from Epic 1):**
- Introspection extracts parameters with `is_tunable=True`
- `default_range` is extracted from type hints when available
- Many parameters lack explicit ranges

**Story 2.8 adds:**
- Hardcoded default ranges for common LLM parameters
- Provider-specific model catalogs
- Intelligent fallback for unknown parameters

### Default Range Catalog

> **⚠️ IMPORTANT:** This story MUST use the existing `TVAR_SEMANTICS` dictionary
> in `introspection.py` rather than creating a duplicate `_DEFAULT_RANGES`.
> The existing semantics already define ranges for common parameters.

**Existing TVAR_SEMANTICS (from introspection.py:42-52):**

```python
TVAR_SEMANTICS = {
    "temperature": {"range": (0.0, 2.0), "scale": "continuous"},
    "top_p": {"range": (0.0, 1.0), "scale": "continuous"},
    "top_k": {"range": (1, 100), "scale": "discrete"},  # Note: 1-100, not 1-50
    "max_tokens": {"range": (1, 4096), "scale": "discrete"},
    "presence_penalty": {"range": (-2.0, 2.0), "scale": "continuous"},
    "frequency_penalty": {"range": (-2.0, 2.0), "scale": "continuous"},
    "score_threshold": {"range": (0.0, 1.0), "scale": "continuous"},
    "similarity_threshold": {"range": (0.0, 1.0), "scale": "continuous"},
}
```

**Story 2.8 extends this with model catalogs:**

```python
# EXTEND existing TVAR_SEMANTICS with model catalogs
# In traigent/integrations/haystack/introspection.py

_MODEL_CATALOGS: dict[str, list[str]] = {
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
    ],
    "anthropic": [
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "claude-3-5-sonnet-20241022",
    ],
    "azure_openai": [
        # Azure uses deployment names, but we can suggest common ones
        "gpt-4o",
        "gpt-4",
        "gpt-35-turbo",
    ],
}

_PROVIDER_DETECTION: dict[str, str] = {
    "OpenAIGenerator": "openai",
    "OpenAIChatGenerator": "openai",
    "AnthropicGenerator": "anthropic",
    "AnthropicChatGenerator": "anthropic",
    "AzureOpenAIGenerator": "azure_openai",
    "AzureOpenAIChatGenerator": "azure_openai",
}
```

### Enhanced Introspection Flow

```python
def _extract_tvar(
    param_name: str,
    param_value: Any,
    type_hint: Any,
    component_class_name: str,
) -> DiscoveredTVAR:
    """Extract TVAR with intelligent defaults."""

    # Start with basic extraction
    tvar = DiscoveredTVAR(
        name=param_name,
        value=param_value,
        python_type=_get_python_type(type_hint),
    )

    # Apply default range if known parameter
    if param_name in _DEFAULT_RANGES:
        defaults = _DEFAULT_RANGES[param_name]
        tvar.default_range = defaults["range"]
        if defaults.get("type") == "int":
            tvar.range_type = "discrete"

    # Apply model choices if it's a model parameter
    if param_name == "model":
        provider = _PROVIDER_DETECTION.get(component_class_name)
        if provider and provider in _MODEL_CATALOGS:
            tvar.literal_choices = _MODEL_CATALOGS[provider]
        else:
            # Fallback: use current value
            tvar.literal_choices = [param_value] if param_value else None
            tvar.non_tunable_reason = (
                "Unknown provider - specify model choices manually"
            )
            logger.warning(
                f"Could not determine model catalog for {component_class_name}. "
                f"Using current value '{param_value}' as only choice. "
                f"Use set_choices() to specify available models."
            )

    return tvar
```

### Testing Strategy

```python
def test_temperature_gets_default_range():
    """Test that temperature parameter gets [0.0, 2.0] range."""
    # Create mock pipeline with OpenAIGenerator
    pipeline = _create_mock_pipeline_with_generator()
    spec = from_pipeline(pipeline)

    temp_tvar = spec.get_tvar("generator", "temperature")
    assert temp_tvar.default_range == (0.0, 2.0)

def test_top_k_gets_default_range():
    """Test that top_k parameter gets [1, 50] range."""
    pipeline = _create_mock_pipeline_with_retriever()
    spec = from_pipeline(pipeline)

    top_k_tvar = spec.get_tvar("retriever", "top_k")
    assert top_k_tvar.default_range == (1, 50)
    assert top_k_tvar.range_type == "discrete"

def test_openai_model_gets_catalog_choices():
    """Test that OpenAI model parameter gets OpenAI model choices."""
    pipeline = _create_mock_pipeline_with_openai_generator()
    spec = from_pipeline(pipeline)

    model_tvar = spec.get_tvar("generator", "model")
    assert "gpt-4o" in model_tvar.literal_choices
    assert "gpt-4o-mini" in model_tvar.literal_choices

def test_unknown_provider_falls_back_with_warning(caplog):
    """Test that unknown provider falls back to current value with warning."""
    pipeline = _create_mock_pipeline_with_custom_generator()
    spec = from_pipeline(pipeline)

    model_tvar = spec.get_tvar("generator", "model")
    # Should have current value as only choice
    assert model_tvar.literal_choices == ["current-model-value"]
    # Should log warning
    assert "specify model choices manually" in caplog.text
```

### File Structure

```
traigent/integrations/haystack/
├── introspection.py     # MODIFY: Add _DEFAULT_RANGES, _MODEL_CATALOGS, enhance extraction
├── defaults.py          # NEW (optional): Extract defaults to separate module
└── __init__.py          # No changes needed

tests/integrations/
└── test_introspection.py  # MODIFY: Add TestDefaultRanges class
```

### Dependencies

- **Story 1.1-1.7 (done)**: Provides base introspection
- **External**: None (hardcoded catalogs, no API calls)

### NFR Considerations

- **NFR-4 (≤10 lines integration)**: Sensible defaults enable immediate use:
  ```python
  spec = from_pipeline(pipeline)  # Defaults are applied
  space = ExplorationSpace.from_pipeline_spec(spec)
  # temperature already has [0.0, 2.0], top_k has [1, 50]
  config = space.sample()  # Works immediately
  ```

### Maintenance Notes

- Model catalogs should be updated when new models are released
- Consider moving catalogs to a config file for easier updates
- Log when catalog was last updated in documentation

### References

- [Source: _bmad-output/epics.md - Epic 2, Story 2.8 - FR-207]
- [Pattern: traigent/integrations/haystack/introspection.py]
- [Dependency: Epic 1 introspection]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A - No issues encountered

### Completion Notes List

- Leveraged existing `TVAR_SEMANTICS` dictionary which already contains default ranges for common parameters (temperature, top_k, top_p, max_tokens, etc.)
- Added `MODEL_CATALOGS` dictionary with 5 providers: openai, anthropic, azure_openai, cohere, google
- Added `PROVIDER_DETECTION` dictionary mapping 13 component class names to providers
- Added `_get_model_choices()` helper function that:
  - Returns model catalog choices for known providers
  - Includes current model value if not in catalog
  - Falls back to single-value list for unknown providers with warning
- Added 17 comprehensive tests in two test classes:
  - `TestModelCatalogs`: 9 tests for catalog and provider detection
  - `TestDefaultRanges`: 8 tests for TVAR semantics
- All 121 introspection tests pass

### File List

- `traigent/integrations/haystack/introspection.py` - Added MODEL_CATALOGS, PROVIDER_DETECTION, _get_model_choices()
- `tests/integrations/test_haystack_introspection.py` - Added TestModelCatalogs and TestDefaultRanges classes

## Change Log

| Date       | Change                                     | Author                                   |
|------------|--------------------------------------------|------------------------------------------|
| 2025-12-20 | Story created for auto-discovery defaults  | Claude Opus 4.5 (create-story workflow)  |
| 2025-12-20 | Story implemented with all tests passing   | Claude Opus 4.5                          |
