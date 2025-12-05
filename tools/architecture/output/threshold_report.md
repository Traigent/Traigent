# Architecture Threshold Check Results

## Configuration

- Max Complexity: 50 (warn: 30)
- Max Methods/Class: 80 (warn: 50)
- Max Lines/File: 2500 (warn: 1500)
- Max Fan-In: 100

## Summary

- **Errors**: 1
- **Warnings**: 7

## ❌ Errors (Must Fix)

| Category | Location | Value | Threshold |
|----------|----------|-------|-----------|
| FAN_IN | `traigent.utils.logging` | 103 | 100 |

## ⚠️ Warnings

| Category | Location | Value | Threshold |
|----------|----------|-------|-----------|
| FILE_SIZE | `analytics/intelligence.py` | 1861 | 1500 |
| FILE_SIZE | `analytics/predictive.py` | 1520 | 1500 |
| FILE_SIZE | `cloud/auth.py` | 2208 | 1500 |
| FILE_SIZE | `cloud/client.py` | 1671 | 1500 |
| FILE_SIZE | `core/optimized_function.py` | 1672 | 1500 |
| FILE_SIZE | `core/orchestrator.py` | 1831 | 1500 |
| FILE_SIZE | `evaluators/base.py` | 2068 | 1500 |
