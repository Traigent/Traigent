# HF Override Fix — Worker Result (Issue #1570)

**Branch:** fix/hf-override-1570  
**Base:** origin/develop @ fbe22e54  
**Date:** 2026-06-29

---

## Round 2 — Codex NO-GO Review Fixes (commit 64c5ee1d)

Two blocking findings from the independent Codex gpt-5.5 review were addressed:

### Finding 1: Generation params injected into `InferenceClient.__init__` → TypeError

**Root cause:** The class-level `PARAMETER_MAPPINGS` for `huggingface_hub.InferenceClient`
and `AsyncInferenceClient` included generation kwargs (`temperature`, `max_tokens`→`max_new_tokens`,
`top_p`, `top_k`, `stop`, `stream`). `FrameworkOverrideManager._create_override_constructor`
iterates over `PARAMETER_MAPPINGS` and injects ALL mapped params into `__init__`. Since
`InferenceClient.__init__` only accepts constructor kwargs (`model`, `token`, `timeout`,
`base_url`, `headers`, `cookies`, `api_key`), injecting generation params caused:

```
TypeError: InferenceClient.__init__() got an unexpected keyword argument 'temperature'
```

**Fix applied:**

`mappings.py` — PARAMETER_MAPPINGS for both InferenceClient classes:
- **Before:** `{model, temperature, max_tokens→max_new_tokens, top_p, top_k, stop, stream}`
- **After:** `{model}` — constructor-valid params only

`mappings.py` — new `METHOD_PARAMETER_TRANSLATIONS` dict:
- Stores method-level param name translations for generation params absent from class-level PARAMETER_MAPPINGS
- `huggingface_hub.InferenceClient.text_generation: {max_tokens: max_new_tokens}`
- `huggingface_hub.AsyncInferenceClient.text_generation: {max_tokens: max_new_tokens}`
- `chat_completion: {}` (max_tokens excluded from METHOD_MAPPINGS for this method; no translation needed)

`framework_override.py` — `_create_override_method` refactored:
- Builds an **effective per-method translation** by merging (priority order):
  1. `METHOD_PARAMETER_TRANSLATIONS` for this class+method (method-specific, e.g. max_tokens→max_new_tokens)
  2. Class-level `PARAMETER_MAPPINGS` (constructor params, e.g. model)
  3. Identity fallback for all remaining `supported_params` (e.g. temperature→temperature)
- Loop now iterates over `supported_params` (not `parameter_mapping.items()`) so all
  method-accepted generation params are injected with correct translation even when absent
  from the class-level PARAMETER_MAPPINGS

`framework_override.py` — new `_init_method_parameter_translations()` method and
`_method_parameter_translations` instance attribute (deep-copied from `METHOD_PARAMETER_TRANSLATIONS`).

### Finding 2: False "fallback/plugin-precedence" documentation at two remaining sites

**Before (mappings.py module docstring):**
> "This module contains the static mappings used as FALLBACK when no plugin is registered.
> Plugin mappings (via LLMPlugin._get_default_mappings) take precedence over these static mappings."

**Before (framework_override.py `_init_method_mappings` docstring ~line 100):**
> "These serve as fallback when no plugin is registered for a framework."

Both were false — the static maps are the single active source; there is no plugin-precedence
layer on top of them in `FrameworkOverrideManager`.

**Fix:** Both docstrings updated to accurately describe the role of the static maps.

---

## Changed Files (Round 2)

| File | Change |
|------|--------|
| `traigent/integrations/mappings.py` | Module docstring fixed; PARAMETER_MAPPINGS for InferenceClient/AsyncInferenceClient stripped to `{model}`; new `METHOD_PARAMETER_TRANSLATIONS` dict + `get_method_parameter_translation()` helper |
| `traigent/integrations/framework_override.py` | `_init_method_mappings` docstring fixed; import `METHOD_PARAMETER_TRANSLATIONS`; new `_init_method_parameter_translations()`; `_create_override_method` refactored (both sync + async inner functions) to use effective per-method mapping |
| `tests/unit/integrations/test_hf_override_consistency.py` | `test_hf_parameter_mappings_have_core_params` updated (now asserts generation params ABSENT from class-level, asserts METHOD_PARAMETER_TRANSLATIONS has max_tokens→max_new_tokens); new `test_hf_constructor_no_generation_param_injection` (Codex repro); new `test_hf_manager_uses_all_target_classes` (strengthened coverage assertion) |

---

## Validation Output (Round 2)

### Codex repro directly confirmed:

```
SUCCESS: InferenceClient created without TypeError
Client model: gpt2
Override restored.
```

### HF consistency tests (10 tests, up from 8):

```
tests/unit/integrations/test_hf_override_consistency.py ..........
10 passed in 1.63s
```

### Full integrations suite:

```
1597 passed in 5.73s
```

### Ruff:

```
All checks passed!
```

---

## Round 1 Summary (original fix, commit on branch)

See git log for prior commit. Root cause was that `override_huggingface()` and
`override_all_platforms()` targeted `transformers.*` instead of `huggingface_hub.*`,
and PARAMETER_MAPPINGS/METHOD_MAPPINGS had no `huggingface_hub.*` entries at all.

---

## Residual Risks (post Round 2)

1. **`transformers.*` entries remain in PARAMETER_MAPPINGS.** They are no longer enabled
   by `override_huggingface()` / `override_all_platforms()`. Users explicitly calling
   `enable_framework_overrides(["transformers.pipeline"])` still get constructor-level
   override. This is an acceptable legacy artifact; removal is a separate cleanup PR.

2. **mypy pre-existing debt.** The `--no-verify` flag is authorized by the captain for
   the pre-existing mypy failures in files outside the write scope (verified identical
   on base fbe22e54).

---

## PR-Ready

**Code changes:** Yes — all tests pass (1597), ruff clean, Codex repro confirmed no TypeError.  
**Commit:** 64c5ee1d — `fix(integrations): scope HF generation params to methods, not InferenceClient.__init__ (#1570 review fixes)`
