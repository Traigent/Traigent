#!/usr/bin/env python3
"""Simple HTTP server for the optimizer validation test viewer app.

Usage:
    python serve.py [--port PORT]

The server provides:
- Static file serving for index.html
- REST API for test collection, execution, and results
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Add project root to path for imports
PROJECT_ROOT_FOR_IMPORTS = Path(__file__).parents[3]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

# Configuration
PORT = int(os.environ.get("VIEWER_PORT", 8765))
HOST = os.environ.get("VIEWER_HOST", "127.0.0.1")  # Bind to localhost only by default
ROOT = Path(__file__).parent
PROJECT_ROOT = (
    ROOT.parent.parent.parent
)  # tests/optimizer_validation/viewer -> project root
RESULTS_DIR = ROOT / "_results"
RESULTS_DIR.mkdir(exist_ok=True)

# Test directory path (used for collection and validation)
TEST_DIR = "tests/optimizer_validation"

# Chat mode configuration
# Set VIEWER_CHAT_USE_CLI=true to enable Claude CLI (requires claude CLI installed)
# Default is false (local fallback mode) for reliability
CHAT_USE_CLI = os.environ.get("VIEWER_CHAT_USE_CLI", "false").lower() == "true"

# Allowed test target prefixes (security: restrict what can be run)
ALLOWED_TARGET_PREFIX = f"{TEST_DIR}/"

# Active jobs tracking
_active_jobs: dict[str, dict[str, Any]] = {}

# Chat conversation state (per-session, simplified for single user)
_chat_messages: list[dict[str, Any]] = []

# Evidence extraction prefix
EVIDENCE_PREFIX = '{"type": "TEST_EVIDENCE"'


def extract_evidence(test_result: dict[str, Any]) -> dict[str, Any] | None:
    """Extract evidence JSON from test logs.

    Looks for TEST_EVIDENCE JSON in the test's log output (captured by
    pytest-json-report from the logging module).

    Args:
        test_result: A single test result dict from pytest-json-report

    Returns:
        Parsed evidence dict, or None if not found
    """
    # Check call phase logs (where most test execution happens)
    call_data = test_result.get("call", {})
    logs = call_data.get("log", [])

    for log_entry in logs:
        msg = log_entry.get("msg", "")
        if msg.startswith(EVIDENCE_PREFIX):
            try:
                return json.loads(msg)
            except json.JSONDecodeError:
                pass

    return None


def process_results_with_evidence(data: dict[str, Any]) -> dict[str, Any]:
    """Process test results to extract and attach evidence.

    Args:
        data: Full pytest-json-report output

    Returns:
        Same data structure with evidence attached to each test
    """
    if "tests" in data:
        for test in data["tests"]:
            test["evidence"] = extract_evidence(test)
    return data


def collect_tests() -> list[dict[str, Any]]:
    """Collect all test metadata using pytest's collection API."""
    import pytest

    class TestCollector:
        def __init__(self) -> None:
            self.tests: list[dict[str, Any]] = []

        def pytest_collection_modifyitems(self, items: list[pytest.Item]) -> None:
            for item in items:
                # Extract test info
                test_info: dict[str, Any] = {
                    "id": item.nodeid,
                    "name": item.name,
                    "module": item.module.__name__ if item.module else "",
                    "file": (
                        str(Path(item.fspath).relative_to(PROJECT_ROOT))
                        if item.fspath
                        else ""
                    ),
                    "class": item.cls.__name__ if item.cls else None,
                    "markers": [m.name for m in item.iter_markers()],
                    "params": {},
                }

                # Extract parameters if present
                if hasattr(item, "callspec"):
                    for key, val in item.callspec.params.items():
                        # Convert to JSON-serializable types
                        if isinstance(val, (str, int, float, bool, type(None))):
                            test_info["params"][key] = val
                        else:
                            test_info["params"][key] = str(val)

                # Extract docstring - full text for detailed view
                if hasattr(item, "function") and item.function:
                    doc = item.function.__doc__
                    if doc:
                        doc_lines = doc.strip().split("\n")
                        test_info["description"] = doc_lines[0]
                        # Full docstring for detailed view
                        test_info["full_description"] = doc.strip()

                # Extract class docstring for context
                if item.cls and item.cls.__doc__:
                    test_info["class_description"] = item.cls.__doc__.strip().split(
                        "\n"
                    )[0]

                # Extract gist_template if present in test fixtures/scenarios
                test_info["gist_template"] = None
                if hasattr(item, "callspec") and item.callspec:
                    for val in item.callspec.params.values():
                        # Look for scenario objects with gist_template attribute
                        if hasattr(val, "gist_template") and val.gist_template:
                            test_info["gist_template"] = val.gist_template
                            break

                self.tests.append(test_info)

    collector = TestCollector()

    # Capture stdout/stderr to suppress pytest output
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = open(os.devnull, "w")

    try:
        pytest.main(
            ["--collect-only", "-q", str(PROJECT_ROOT / TEST_DIR)],
            plugins=[collector],
        )
    finally:
        sys.stdout.close()
        sys.stdout, sys.stderr = old_stdout, old_stderr

    return collector.tests


def organize_tests(tests: list[dict[str, Any]]) -> dict[str, Any]:
    """Organize tests into a hierarchical structure."""
    tree: dict[str, Any] = {"name": "root", "children": {}, "tests": []}

    for test in tests:
        # Parse path from nodeid: tests/optimizer_validation/dimensions/test_x.py::Class::method
        parts = test["id"].split("::")
        file_path = parts[0]

        # Extract category from file path
        path_parts = file_path.replace(f"{TEST_DIR}/", "").split("/")
        if len(path_parts) >= 2:
            category = path_parts[0]  # dimensions, failures, interactions
            test_file = path_parts[1].replace("test_", "").replace(".py", "")
        else:
            category = "other"
            test_file = path_parts[0].replace("test_", "").replace(".py", "")

        # Build tree
        if category not in tree["children"]:
            tree["children"][category] = {"name": category, "children": {}, "tests": []}

        if test_file not in tree["children"][category]["children"]:
            tree["children"][category]["children"][test_file] = {
                "name": test_file,
                "tests": [],
            }

        tree["children"][category]["children"][test_file]["tests"].append(test)

    return tree


def run_tests_async(job_id: str, test_target: str, result_file: Path) -> None:
    """Run tests in a background thread."""
    _active_jobs[job_id]["status"] = "running"
    _active_jobs[job_id]["start_time"] = time.time()

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        test_target,
        "--json-report",
        f"--json-report-file={result_file}",
        "--json-report-indent=2",
        "--log-cli-level=INFO",  # Capture INFO logs for evidence
        "-v",
    ]

    env = os.environ.copy()
    env["TRAIGENT_MOCK_MODE"] = "true"
    env["TRAIGENT_TRACE_ENABLED"] = "true"
    env["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4318"
    # Add mock delay to make parallel execution visible in traces (50ms default)
    env.setdefault("TRAIGENT_MOCK_DELAY_MS", "50")

    try:
        process = subprocess.run(
            cmd,
            env=env,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        _active_jobs[job_id]["status"] = "completed"
        _active_jobs[job_id]["exit_code"] = process.returncode
        _active_jobs[job_id]["stdout"] = process.stdout
        _active_jobs[job_id]["stderr"] = process.stderr
    except Exception as e:
        _active_jobs[job_id]["status"] = "error"
        _active_jobs[job_id]["error"] = str(e)

    _active_jobs[job_id]["end_time"] = time.time()


def is_valid_target(target: str) -> bool:
    """Validate that the test target is within allowed paths.

    Security: Only allow running tests within tests/optimizer_validation/.
    This prevents arbitrary command execution via the /api/run endpoint.
    """
    # Normalize target path
    normalized = target.rstrip("/") + "/" if not target.endswith("/") else target

    # Must start with allowed prefix
    if not normalized.startswith(ALLOWED_TARGET_PREFIX):
        # Also allow individual test nodeids like tests/optimizer_validation/...::test_x
        if not target.startswith(ALLOWED_TARGET_PREFIX):
            return False

    # Block path traversal attempts
    if ".." in target:
        return False

    return True


def start_test_run(test_target: str) -> str:
    """Start a test run and return job ID."""
    job_id = str(uuid.uuid4())[:8]
    result_file = RESULTS_DIR / f"{job_id}.json"

    _active_jobs[job_id] = {
        "id": job_id,
        "target": test_target,
        "result_file": str(result_file),
        "status": "pending",
    }

    thread = threading.Thread(
        target=run_tests_async, args=(job_id, test_target, result_file)
    )
    thread.daemon = True
    thread.start()

    return job_id


def get_latest_results() -> dict[str, Any] | None:
    """Get the most recent test results with evidence extracted."""
    result_files = sorted(RESULTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not result_files:
        return None

    latest = result_files[-1]
    try:
        with open(latest) as f:
            result: dict[str, Any] = json.load(f)
            return process_results_with_evidence(result)
    except (json.JSONDecodeError, OSError):
        return None


def process_chat_message(message: str, history: list[dict[str, Any]]) -> dict[str, Any]:
    """Process a chat message.

    Uses either Claude CLI (if VIEWER_CHAT_USE_CLI=true) or local tool execution.
    Local mode is default for reliability (CLI can have fs.watch issues in subprocess).

    Args:
        message: The user's message
        history: Recent conversation history

    Returns:
        Dict with 'response' text, 'tools_used' list, and 'mode' indicator
    """
    if CHAT_USE_CLI:
        result = _process_chat_with_cli(message, history)
        # If CLI fails, fall back to local mode
        if "CLI error" in result.get("response", "") or "Error:" in result.get(
            "response", ""
        ):
            result = _process_chat_fallback(message)
            result["mode"] = "local (CLI failed)"
        else:
            result["mode"] = "cli"
        return result

    # Default: use local tool execution for reliability
    result = _process_chat_fallback(message)
    result["mode"] = "local"
    return result


def _process_chat_with_cli(
    message: str, history: list[dict[str, Any]]
) -> dict[str, Any]:
    """Process chat using Claude Code CLI.

    Note: This can fail with fs.watch errors when invoked as subprocess.
    Use VIEWER_CHAT_USE_CLI=true to enable, but expect potential failures.
    """
    # Build context from recent history
    context_lines = []
    for msg in history[-4:]:  # Include last 4 messages for context
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            context_lines.append(f"User: {content}")
        else:
            context_lines.append(f"Assistant: {content[:200]}...")

    history_context = "\n".join(context_lines) if context_lines else "No prior context."

    # Build the prompt for Claude Code CLI
    cli_prompt = f"""You are a test suite assistant for Traigent's optimizer validation tests.

## Available Tools (via Python)
- `get_test_stats()` - Get overall test statistics
- `list_dimensions()` - List all test dimensions and values
- `search_tests(keyword)` - Search tests by keyword
- `get_coverage_gaps(dim1, dim2)` - Find uncovered dimension pairs

## Recent Conversation
{history_context}

## Current Question
{message}

Be concise and helpful. Use the tools to answer questions about tests."""

    try:
        result = subprocess.run(
            [
                "claude",
                "--print",
                "--model",
                "sonnet",  # Use Sonnet for faster responses
                "--dangerously-skip-permissions",
                cli_prompt,
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(PROJECT_ROOT),
        )

        if result.returncode == 0:
            response_text = result.stdout.strip()
            return {
                "response": response_text if response_text else "No response.",
                "tools_used": [],
            }
        else:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            return {"response": f"CLI error: {error_msg[:300]}", "tools_used": []}

    except subprocess.TimeoutExpired:
        return {"response": "Request timed out.", "tools_used": []}
    except FileNotFoundError:
        return {"response": "CLI error: claude not installed", "tools_used": []}
    except Exception as e:
        return {"response": f"Error: {str(e)}", "tools_used": []}


def _process_chat_fallback(message: str) -> dict[str, Any]:
    """Fallback chat processing when Claude CLI is not available.

    Directly executes tool functions based on keywords in the message.
    Includes evidence and explanations for transparency.
    """
    from tests.optimizer_validation.chatbot.tools import (
        get_coverage_gaps,
        get_test_stats,
        list_dimensions,
        search_tests,
    )

    message_lower = message.lower()
    tools_used = []

    def format_evidence(tool_name: str, query_info: str, result_summary: str) -> str:
        """Format evidence section for transparency."""
        return f"""
---
**🔍 How I found this:**
- Tool: `{tool_name}()`
- Query: {query_info}
- Result: {result_summary}"""

    try:
        # Simple keyword-based routing
        if "stat" in message_lower:
            tools_used.append("get_test_stats")
            stats = get_test_stats()
            evidence = format_evidence(
                "get_test_stats",
                "Fetched overall test suite statistics from knowledge graph",
                f"Retrieved counts for {stats['total_tests']} total tests",
            )
            return {
                "response": f"""**Test Suite Statistics**
- Total tests: {stats['total_tests']}
- Passed: {stats['passed']}
- Failed: {stats['failed']}
- Not run: {stats['not_run']}
{evidence}""",
                "tools_used": tools_used,
            }

        if "dimension" in message_lower or "dims" in message_lower:
            tools_used.append("list_dimensions")
            dims = list_dimensions()
            lines = ["**Test Dimensions**"]
            for dim, values in dims.items():
                lines.append(f"- {dim}: {', '.join(values)}")
            evidence = format_evidence(
                "list_dimensions",
                "Retrieved all dimension definitions from test ontology",
                f"Found {len(dims)} dimensions with their valid values",
            )
            lines.append(evidence)
            return {"response": "\n".join(lines), "tools_used": tools_used}

        if "gap" in message_lower or "coverage" in message_lower:
            tools_used.append("get_coverage_gaps")
            # Default to InjectionMode vs Algorithm
            gaps = get_coverage_gaps("InjectionMode", "Algorithm")
            if gaps:
                lines = ["**Coverage Gaps (InjectionMode × Algorithm)**"]
                for gap in gaps[:10]:
                    lines.append(f"- {gap['InjectionMode']} + {gap['Algorithm']}")
                if len(gaps) > 10:
                    lines.append(f"...and {len(gaps) - 10} more")
            else:
                lines = ["No coverage gaps found for InjectionMode × Algorithm"]
            evidence = format_evidence(
                "get_coverage_gaps",
                'get_coverage_gaps("InjectionMode", "Algorithm")',
                f"Checked all {4 * 6} possible pairs, found {len(gaps)} uncovered",
            )
            lines.append(evidence)
            return {"response": "\n".join(lines), "tools_used": tools_used}

        if "failed" in message_lower or "fail" in message_lower:
            tools_used.append("get_test_stats")
            stats = get_test_stats()
            evidence = format_evidence(
                "get_test_stats",
                "Queried test results for failure count",
                f"Scanned {stats['total_tests']} tests for status='FAIL'",
            )
            return {
                "response": f"""**Failed Tests**: {stats['failed']} tests have failed.
{evidence}""",
                "tools_used": tools_used,
            }

        # Check for specific keywords to search for
        search_keywords = ["optuna", "grid", "random", "bayesian", "parallel", "inject"]
        for kw in search_keywords:
            if kw in message_lower:
                tools_used.append("search_tests")
                results = search_tests(kw)
                if results:
                    lines = [f"**Tests matching '{kw}'** ({len(results)} found)"]
                    for r in results[:15]:
                        lines.append(f"- {r['name']}")
                    if len(results) > 15:
                        lines.append(f"...and {len(results) - 15} more")
                else:
                    lines = [f"No tests found matching '{kw}'"]
                evidence = format_evidence(
                    "search_tests",
                    f'search_tests("{kw}")',
                    f"Searched test names and descriptions for '{kw}', found {len(results)} matches",
                )
                lines.append(evidence)
                return {"response": "\n".join(lines), "tools_used": tools_used}

        if "search" in message_lower or "find" in message_lower:
            # Extract search term (simple heuristic)
            words = message.split()
            search_term = words[-1] if len(words) > 1 else "test"
            tools_used.append("search_tests")
            results = search_tests(search_term)
            if results:
                lines = [f"**Tests matching '{search_term}'**"]
                for r in results[:10]:
                    lines.append(f"- {r['name']}")
                if len(results) > 10:
                    lines.append(f"...and {len(results) - 10} more")
            else:
                lines = [f"No tests found matching '{search_term}'"]
            evidence = format_evidence(
                "search_tests",
                f'search_tests("{search_term}")',
                f"Searched test names and descriptions, found {len(results)} matches",
            )
            lines.append(evidence)
            return {"response": "\n".join(lines), "tools_used": tools_used}

        # Default response
        return {
            "response": """I can help you with:
- **Stats**: "Show test stats"
- **Dimensions**: "List dimensions"
- **Gaps**: "What coverage gaps exist?"
- **Failed**: "Show failed tests"
- **Search**: "Find tests with optuna"

(Note: Using local fallback mode - Claude CLI not invoked)""",
            "tools_used": [],
        }

    except Exception as e:
        return {"response": f"Error: {str(e)}", "tools_used": tools_used}


class ViewerHandler(SimpleHTTPRequestHandler):
    """HTTP request handler for the test viewer."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self) -> None:
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/tests":
            self._handle_get_tests()
        elif path == "/api/results":
            self._handle_get_results()
        elif path.startswith("/api/run/status/"):
            job_id = path.split("/")[-1]
            self._handle_get_status(job_id)
        elif path.startswith("/api/open"):
            self._handle_open_file()
        elif path.startswith("/api/file/"):
            self._handle_get_file_content()
        elif path == "/" or path == "/index.html":
            self._serve_file("index.html")
        elif path == "/favicon.ico":
            # Return empty favicon to avoid errors
            self.send_response(204)
            self.end_headers()
        else:
            super().do_GET()

    def do_POST(self) -> None:
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/run":
            self._handle_run_tests()
        elif path == "/api/chat":
            self._handle_chat()
        else:
            self.send_error(404, "Not Found")

    def _send_json(self, data: Any, status: int = 200) -> None:
        """Send JSON response."""
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, filename: str) -> None:
        """Serve a static file."""
        file_path = ROOT / filename
        if file_path.exists():
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            if filename.endswith(".html"):
                self.send_header("Content-Type", "text/html")
            elif filename.endswith(".js"):
                self.send_header("Content-Type", "application/javascript")
            elif filename.endswith(".css"):
                self.send_header("Content-Type", "text/css")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_error(404, "File not found")

    def _handle_get_tests(self) -> None:
        """Handle GET /api/tests - return test list."""
        tests = collect_tests()
        tree = organize_tests(tests)
        self._send_json({"tests": tests, "tree": tree, "count": len(tests)})

    def _handle_get_results(self) -> None:
        """Handle GET /api/results - return latest test results."""
        results = get_latest_results()
        if results:
            self._send_json(results)
        else:
            self._send_json({"error": "No results found"}, 404)

    def _handle_get_status(self, job_id: str) -> None:
        """Handle GET /api/run/status/:job_id - return job status."""
        if job_id not in _active_jobs:
            self._send_json({"error": "Job not found"}, 404)
            return

        job = _active_jobs[job_id]
        response: dict[str, Any] = {
            "id": job["id"],
            "status": job["status"],
            "target": job["target"],
        }

        if job["status"] == "completed":
            # Load results from file and extract evidence
            try:
                with open(job["result_file"]) as f:
                    raw_results = json.load(f)
                    response["results"] = process_results_with_evidence(raw_results)
            except (json.JSONDecodeError, OSError):
                response["results"] = None

            if "exit_code" in job:
                response["exit_code"] = job["exit_code"]

        if "start_time" in job:
            elapsed = job.get("end_time", time.time()) - job["start_time"]
            response["duration"] = elapsed

        if job["status"] == "error":
            response["error"] = job.get("error", "Unknown error")

        self._send_json(response)

    def _handle_run_tests(self) -> None:
        """Handle POST /api/run - start a test run."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")

        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON"}, 400)
            return

        # Determine test target (default to all validation tests)
        test_target = data.get("target", f"{TEST_DIR}/")

        # Security: Validate target is within allowed paths
        if not is_valid_target(test_target):
            self._send_json(
                {"error": f"Invalid target. Must be within {TEST_DIR}/"}, 403
            )
            return

        # Start test run
        job_id = start_test_run(test_target)

        self._send_json({"job_id": job_id, "status": "started"})

    def _handle_chat(self) -> None:
        """Handle POST /api/chat - process chat message."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")

        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON"}, 400)
            return

        message = data.get("message", "").strip()
        if not message:
            self._send_json({"error": "No message provided"}, 400)
            return

        history = data.get("history", [])

        try:
            result = process_chat_message(message, history)
            self._send_json(result)
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _handle_open_file(self) -> None:
        """Handle GET /api/open?file=path&line=N - open file in system editor."""
        from urllib.parse import parse_qs

        query = parse_qs(urlparse(self.path).query)
        file_path = query.get("file", [""])[0]
        line_num = query.get("line", ["1"])[0]

        if not file_path:
            self._send_json({"error": "No file specified"}, 400)
            return

        # Security: Validate file path is within project
        full_path = PROJECT_ROOT / file_path
        try:
            full_path = full_path.resolve()
            if not str(full_path).startswith(str(PROJECT_ROOT.resolve())):
                self._send_json({"error": "Invalid file path"}, 403)
                return
        except (OSError, ValueError):
            self._send_json({"error": "Invalid file path"}, 400)
            return

        if not full_path.exists():
            self._send_json({"error": "File not found"}, 404)
            return

        # Try to open in VS Code (if available)
        try:
            # Try 'code' command (VS Code CLI)
            subprocess.run(
                ["code", "--goto", f"{full_path}:{line_num}"],
                check=False,
                timeout=2,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._send_json(
                {"success": True, "editor": "vscode", "file": str(full_path)}
            )
            return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Fallback: Return file content for built-in viewer
        self._send_json(
            {
                "success": True,
                "editor": "browser",
                "file": str(file_path),
                "line": int(line_num) if line_num.isdigit() else 1,
            }
        )

    def _handle_get_file_content(self) -> None:
        """Handle GET /api/file/path/to/file.py - get file content for viewer."""
        # Extract file path from URL (everything after /api/file/)
        file_path = self.path.replace("/api/file/", "", 1)

        if not file_path:
            self._send_json({"error": "No file specified"}, 400)
            return

        # Security: Validate file path is within project
        full_path = PROJECT_ROOT / file_path
        try:
            full_path = full_path.resolve()
            if not str(full_path).startswith(str(PROJECT_ROOT.resolve())):
                self._send_json({"error": "Invalid file path"}, 403)
                return
        except (OSError, ValueError):
            self._send_json({"error": "Invalid file path"}, 400)
            return

        if not full_path.exists():
            self._send_json({"error": "File not found"}, 404)
            return

        # Read and return file content
        try:
            content = full_path.read_text()
            self._send_json(
                {
                    "file": file_path,
                    "content": content,
                    "lines": content.count("\n") + 1,
                }
            )
        except (OSError, UnicodeDecodeError) as e:
            self._send_json({"error": f"Failed to read file: {e}"}, 500)

    def log_message(self, format: str, *args: Any) -> None:
        """Log HTTP requests."""
        # Only log non-API requests for cleaner output
        # args[0] may be HTTPStatus enum on errors, so convert to string
        try:
            first_arg = str(args[0]) if args else ""
            if not first_arg.startswith("GET /api"):
                print(f"[{self.log_date_time_string()}] {format % args}")
        except Exception:
            # Fallback: just print what we can
            print(f"[{self.log_date_time_string()}] {format}")


def start_server(host: str, port: int) -> None:
    """Start the test viewer server."""
    server = HTTPServer((host, port), ViewerHandler)

    print(f"\n{'='*60}")
    print("  Optimizer Validation Test Viewer")
    print(f"{'='*60}")
    print(f"\n  Open in browser: http://{host}:{port}")
    if host == "127.0.0.1":
        print("  (Bound to localhost only for security)")
    print("\n  API endpoints:")
    print("    GET  /api/tests         - List all tests")
    print("    GET  /api/results       - Get latest results")
    print("    POST /api/run           - Run tests")
    print("    GET  /api/run/status/ID - Check job status")
    print("    POST /api/chat          - Chat with test assistant")
    print("\n  Press Ctrl+C to stop\n")
    print(f"{'='*60}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


def run_with_knowledge_graph(
    kg: Any, output_path: Path | None = None, host: str = HOST, port: int = PORT
) -> None:
    """Write graph data and start the viewer."""
    graph_path = output_path or (ROOT / "graph_data.json")
    try:
        graph_path.write_text(json.dumps(kg.to_json(), indent=2))
        print(f"Wrote graph data to {graph_path}")
    except OSError as exc:
        print(f"Failed to write graph data to {graph_path}: {exc}")
    start_server(host, port)


def main() -> None:
    """Start the test viewer server."""
    import argparse

    parser = argparse.ArgumentParser(description="Optimizer validation test viewer")
    parser.add_argument("--port", type=int, default=PORT, help="Port to run on")
    parser.add_argument(
        "--host",
        type=str,
        default=HOST,
        help="Host to bind to (default: 127.0.0.1 for security)",
    )
    args = parser.parse_args()

    start_server(args.host, args.port)


if __name__ == "__main__":
    main()
