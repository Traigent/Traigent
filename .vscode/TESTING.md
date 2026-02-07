# VSCode Testing Configuration

## Parallel Test Execution

This project is configured to use **pytest-xdist** for parallel test execution.

### System Specs
- **RAM**: 196GB
- **CPU Cores**: 24
- **Parallel Workers**: `auto` (pytest-xdist automatically uses all available cores)

### Configuration Files

1. **`pytest.ini`** - Default pytest configuration
   - Uses `-n auto --dist loadgroup` for parallel execution
   - Automatically used by command-line `pytest` and `make test`

2. **`.vscode/settings.json`** - VSCode Python test runner
   - Uses same parallel configuration (`-n auto --dist loadgroup`)
   - Applied when running tests via VSCode UI (Testing panel, Run/Debug)

### Running Tests

#### From VSCode UI
1. Open Testing panel (beaker icon in sidebar)
2. Click "Run All Tests" - **runs in parallel automatically**
3. Individual test files run in parallel too

#### From Terminal
```bash
# Parallel (default) - uses all 24 cores
make test-unit

# Sequential (for debugging) - single-threaded
make test-unit-serial

# Or directly with pytest
pytest tests/unit/ -n auto  # parallel
pytest tests/unit/          # sequential (no -n flag)
```

### Load Distribution Strategy

We use `--dist loadgroup` which:
- Groups tests by **test file** (not individual tests)
- Distributes groups across workers for balanced load
- Tests in same file run together (good for fixtures/setup)

Alternative strategies:
- `--dist loadscope` - Group by test scope/module
- `--dist loadfile` - One file per worker at a time
- `--dist worksteal` - Dynamic work stealing (fastest for uneven test durations)

### Excluded Tests (Always Skipped in VSCode)

These tests are automatically ignored to prevent hangs/crashes:

- `tests/integration/` - Integration tests (slow)
- `tests/e2e/` - End-to-end tests (very slow)
- `tests/performance/` - Performance benchmarks
- `tests/chaos/` - Chaos/fault injection tests
- `tests/optimizer_validation/` - Optimizer validation suite
- `tests/unit/bridges/` - JS bridge tests (can crash terminals)
- `tests/unit/core/test_orchestrator.py` - Known to hang at ~33%
- `tests/unit/core/test_constraints_enforced.py` - Hangs
- `tests/unit/evaluators/test_litellm_integration.py` - Crashes with pydantic 2.12.x

To run these, use command line with explicit paths.

### Performance Tips

1. **First run is slower** - pytest-xdist spawns workers and distributes tests
2. **Subsequent runs are faster** - workers reuse pytest cache
3. **For single test debugging** - Use "Run Test" on specific test (still parallel by file)
4. **For true single-threaded** - Use `make test-unit-serial` or `pytest tests/unit/some_test.py` (no `-n`)

### Customizing Worker Count

If you want to limit parallelism (e.g., debugging race conditions):

```bash
# Use 4 workers instead of auto (24)
pytest tests/unit/ -n 4

# Or set in pytest.ini temporarily:
# addopts = -n 4
```

### Troubleshooting

**Problem**: Tests hang or never finish
- **Solution**: Check if `test_orchestrator.py` or other excluded tests are running
- **Solution**: Try `make test-unit-serial` to run sequentially

**Problem**: Random test failures that don't happen sequentially
- **Solution**: Likely a test isolation issue (shared state, global variables)
- **Solution**: Run with `-n 1` to confirm, then fix the test

**Problem**: "Too many open files" error
- **Solution**: Increase ulimit: `ulimit -n 65536`

**Problem**: Out of memory (unlikely with 196GB!)
- **Solution**: Reduce worker count: `pytest -n 12` instead of `auto`

## See Also

- [pytest-xdist documentation](https://pytest-xdist.readthedocs.io/)
- `MEMORY.md` - Known test issues and workarounds
