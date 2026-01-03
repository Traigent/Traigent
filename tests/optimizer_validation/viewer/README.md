# Optimizer Validation Test Viewer

A lightweight web UI for browsing, running, and debugging optimizer validation tests.

## Quick Start

1. **Install dependency** (one-time):
   ```bash
   pip install pytest-json-report
   ```

2. **Start Jaeger** (optional, for traces):
   ```bash
   make jaeger-start
   ```

3. **Start the viewer**:
   ```bash
   python tests/optimizer_validation/viewer/serve.py
   ```

4. **Open in browser**:
   ```
   http://localhost:8765
   ```

## Features

- **Browse tests** - Navigate 250+ tests organized by category (dimensions, failures, interactions)
- **View details** - See test scenario specs, parameters, and markers
- **Run tests** - Execute individual tests, categories, or all tests with one click
- **View results** - See pass/fail status, duration, and error messages
- **Trace integration** - Link directly to Jaeger traces for debugging

## UI Overview

```
┌──────────────────────────────────────────────────────────────────┐
│ Optimizer Validation Tests              [Refresh] [Run All Tests]│
├────────────────────┬─────────────────────────────────────────────┤
│ Categories         │ Test Details                                │
│ ▼ dimensions (154) │ test_injection_mode_basic[context]          │
│   ▶ injection (16) │                                             │
│   ▶ execution (17) │ Module: test_injection_modes                │
│   ...              │ Class: TestInjectionModeMatrix              │
│ ▼ failures (73)    │                                             │
│   ...              │ [Run Test] [View Trace in Jaeger]           │
│ ▶ interactions(23) │                                             │
│                    │ Last Result: ✅ PASSED (0.15s)               │
│ [Search...]        │                                             │
└────────────────────┴─────────────────────────────────────────────┘
```

## API Endpoints

The server exposes a REST API:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/tests` | GET | List all tests with metadata |
| `/api/results` | GET | Get latest test results |
| `/api/run` | POST | Run tests (body: `{"target": "test_id"}`) |
| `/api/run/status/:id` | GET | Check job status |

## Configuration

- **Port**: Set via `VIEWER_PORT` env var or `--port` flag (default: 8765)
- **Jaeger**: Expected at `http://localhost:16686` (default docker setup)

## Directory Structure

```
viewer/
├── index.html      # Single-page app (HTML/CSS/JS)
├── serve.py        # Python HTTP server + API
├── README.md       # This file
└── _results/       # Test result JSON files (auto-created)
```

## Tips

- **Search**: Use the search box to filter tests by name
- **Expand/Collapse**: Click category headers to expand/collapse
- **Run category**: Click the "Run" button next to a category
- **Trace viewing**: Make sure Jaeger is running before clicking "View Trace"

## Troubleshooting

**"Failed to load tests"**
- Make sure you're running from the project root directory
- Check that pytest is installed and working

**Tests not running**
- Ensure `TRAIGENT_MOCK_LLM` is available
- Check console output from serve.py for errors

**Traces not showing in Jaeger**
- Start Jaeger: `make jaeger-start`
- Traces are created when tests run with `TRAIGENT_TRACE_ENABLED=true`
