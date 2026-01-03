# VS Code Restart Guide After Environment Changes

This guide helps when VS Code extensions lose the Python environment after
recreating `.venv/` or changing dependencies.

## Symptoms

- isort client: couldn't create connection to server.
- Mypy Type Checker client: couldn't create connection to server.

## Quick Fix

1. Reload the VS Code window:
   - Command Palette -> "Developer: Reload Window"
2. If that fails, fully close and reopen VS Code.

## Verify the Interpreter

1. Check the status bar for the selected Python interpreter.
2. If it's not `.venv/bin/python`, run "Python: Select Interpreter" and pick it.

## Sanity Check

Run a quick test from the terminal:

```bash
source .venv/bin/activate
TRAIGENT_MOCK_LLM=true pytest tests/unit/core/test_stop_conditions.py -v
```

## If Issues Persist

- Restart the Python language server ("Python: Restart Language Server").
- Disable/re-enable the Python and Pylance extensions.
- Recreate the virtual environment and reload the window again.
