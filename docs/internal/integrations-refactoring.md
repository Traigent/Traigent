# Refactoring Plan: Phase 2 & 3

## Phase 2: Parameter Mapping Consolidation (~400 lines reduction)

### Objectives
- Remove hardcoded parameter mappings from `framework_override.py`.
- Centralize mapping logic in `ParameterNormalizer` and `LLMPlugin`.

### Tasks
| Task ID | Description |
| :--- | :--- |
| 2.1 | Extend `ParameterNormalizer` with `get_canonical_parameters()` and `get_framework_parameter()`. |
| 2.2 | Create `LLMPlugin` base class that auto-generates mappings from normalizer. |
| 2.3 | Migrate all 10 LLM plugins to inherit from `LLMPlugin`. |
| 2.4 | Delete hardcoded `PARAMETER_MAPPINGS` from `framework_override.py`. |

### Proposed Plugin Assignment
- **Gemini**: openai, anthropic, langchain, azure_openai (4 plugins)
- **Codex**: llamaindex, bedrock, gemini, cohere, huggingface (5 plugins)

---

## Phase 3: Module Split (~0 net lines, but organized)

### Objectives
- Decompose the monolithic `framework_override.py` into focused modules.

### Structure
`traigent/integrations/override/`
- `manager.py`: `FrameworkOverrideManager` (~350 lines)
- `context.py`: Context managers (~100 lines)
- `wrappers.py`: Sync/async wrappers (~100 lines)
- `targets.py`: Framework definitions (~150 lines)
- `discovery.py`: Dynamic discovery (~100 lines)

`framework_override.py` -> Facade only (~50 lines) keeping backward compatibility imports.

---

## Key Questions for Review

1. **LLMPlugin approach: Inheritance vs composition?**
2. **Plugin assignment: Is the split balanced?**
3. **Phase order: Should Phase 3 come first?**
4. **Backwards compatibility: How long to keep LegacyBaseOverrideManager?**
