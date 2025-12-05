# Call Graph Analysis Report

**Generated from**: 223 modules
**Total Functions**: 3400
**Total Call Relationships**: 19404

## Most Called Functions (Hot Paths)

These functions are called most frequently and are critical paths in the codebase.

| Rank | Function | Calls | Module |
|------|----------|-------|--------|
| 1 | `len` | 1074 | unknown |
| 2 | `isinstance` | 851 | unknown |
| 3 | `info` | 430 | logger |
| 4 | `debug` | 408 | logger |
| 5 | `warning` | 361 | logger |
| 6 | `str` | 330 | unknown |
| 7 | `getattr` | 287 | unknown |
| 8 | `ValueError` | 283 | unknown |
| 9 | `hasattr` | 246 | unknown |
| 10 | `time` | 229 | time |
| 11 | `float` | 228 | unknown |
| 12 | `sum` | 201 | unknown |
| 13 | `error` | 196 | logger |
| 14 | `max` | 195 | unknown |
| 15 | `cast` | 177 | unknown |
| 16 | `print` | 174 | console |
| 17 | `list` | 167 | unknown |
| 18 | `int` | 164 | unknown |
| 19 | `now` | 164 | datetime |
| 20 | `min` | 136 | unknown |
| 21 | `echo` | 125 | click |
| 22 | `range` | 104 | unknown |
| 23 | `dict` | 93 | unknown |
| 24 | `ValidationError` | 92 | unknown |
| 25 | `getenv` | 86 | os |
| 26 | `set` | 84 | unknown |
| 27 | `type` | 81 | unknown |
| 28 | `join` | 80 | unknown |
| 29 | `append` | 79 | unknown |
| 30 | `isoformat` | 77 | unknown |

## Entry Points

Functions that are not called by other functions in the codebase (potential API surface):

### traigent.agents
- `register_platform_mapping`
- `apply_configuration`
- `get_supported_platforms`
- `get_platform_mapping`
- `validate_configuration_compatibility`
- `apply_config_to_agent`
- `validate_config_compatibility`
- `register_platform_mapping`
- `get_supported_platforms`
- `initialize`

### traigent.analytics
- `add_data_point`
- `detect_anomalies`
- `set_threshold`
- `detect_anomalies`
- `record_performance`
- `detect_performance_regression`
- `add_optimization_result`
- `detect_regressions`
- `add_monitoring_rule`
- `add_notification_callback`

## Leaf Functions

Functions that don't call other functions (442 total).
These are typically simple utilities or terminal operations.

- `__init__`
- `_platform_initialize`
- `_execute_agent`
- `_validate_platform_spec`
- `_get_platform_capabilities`
- `_validate_platform_config`
- `_import_langchain_components`
- `_get_platform_capabilities`
- `__missing__`
- `_get_platform_capabilities`
- `_extract_api_kwargs`
- `__init__`
- `build_template`
- `infer_platform`
- `__init__`
- `_generate_reasoning_instructions`
- `__init__`
- `_calculate_severity`
- `__init__`
- `set_threshold`

## Recommendations

### High-Traffic Functions to Optimize

Consider profiling and optimizing these frequently-called functions:

1. **len** (1074 calls) - High impact on performance
1. **isinstance** (851 calls) - High impact on performance
1. **info** (430 calls) - High impact on performance
1. **debug** (408 calls) - High impact on performance
1. **warning** (361 calls) - High impact on performance

### Potential Refactoring Targets

Functions with many callers may benefit from interface stabilization:

- `str` (330 callers)
- `getattr` (287 callers)
- `ValueError` (283 callers)
- `hasattr` (246 callers)
- `time` (229 callers)