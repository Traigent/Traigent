# Manual Validation

These checks are intentionally kept out of the default `tests/` suite.

They are local/manual harnesses for backend-driven validation and typically
require a localhost service or hands-on verification.

Run them explicitly with:

```bash
RUN_MANUAL_VALIDATION=1 pytest manual_validation -o addopts=''
```

This keeps the main pytest surface clean while preserving the harnesses for
targeted debugging and release validation.
