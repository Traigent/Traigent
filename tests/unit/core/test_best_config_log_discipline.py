"""Structural guard: no value-bearing log call or raise on the best-config paths.

The leak class (customer text reaching SDK logs / exception text) kept recurring one call
site at a time. This scans the best-config modules with ``ast`` and fails on any

* logger call (``logger.debug/info/warning/error/exception/critical/log``) that passes
  anything but a constant message, or whose message is an f-string / ``%`` / ``.format``;
* ``raise X(...)`` whose message is an f-string, ``%``-format, ``.format`` or ``+`` concat
  involving a variable;

outside ``best_config_diagnostics.py`` (the one place diagnostics are worded). Existing
sites that are safe are allowlisted below, each with the reason it is safe. A new site
fails here: route it through ``best_config_diagnostics`` (``log`` / ``refusal``) or, if it
truly carries only SDK-owned constants, add it to ``ALLOWLIST`` with that reason.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "traigent"
FILES = (
    *sorted((ROOT / "core").glob("best_config_*.py")),
    ROOT / "core" / "config_state_manager.py",
    ROOT / "cloud" / "backend_client.py",
    ROOT / "utils" / "optimization_result_persistence.py",
)
EXEMPT = {"best_config_diagnostics.py"}
LOG_METHODS = {"debug", "info", "warning", "error", "exception", "critical", "log"}

# (file name, enclosing function, kind) -> (number of sites allowed, reason).
# A site is allowed only if every interpolated value is one of: a count, an exception
# *type* name, a code identifier (function name, GNFR-PRIV-003), a fixed label derived from
# the source enum (``_source_label``), an SDK-owned constant, or text the SDK wrote itself.
_COUNT = "interpolates only counts / type names / code identifiers / SDK constants"
_LABEL = (
    "interpolates only _source_label() (fixed enum-derived label) and fixed reasons"
)
_DECODER = (
    "decoder ValueError text: the only best-config caller (load_optimization_results) no "
    "longer passes artifact_name (label is '<unnamed>') and keeps the text off str/args by "
    "wrapping it in diag.refusal(reason=...) details; other callers (persistence.py) name "
    "their own artifacts"
)
_LEGACY_BACKEND = (
    "PRE-EXISTING, NOT PROVEN SAFE: generic backend-client path outside the best-config "
    "fetch/publish/head paths (session ids, run ids, exception text). Out of scope for "
    "F1j; any change to these sites should move them to best_config_diagnostics and "
    "remove the entry. Reported as an open follow-up."
)
ALLOWLIST: dict[tuple[str, str, str], tuple[int, str]] = {
    ("best_config_runtime.py", "_validate_and_filter_config_keys", "log"): (1, _COUNT),
    ("best_config_runtime.py", "_normalize_json_value", "raise"): (1, _COUNT),
    ("config_state_manager.py", "_read_spec_file", "raise"): (
        1,
        _LABEL + "; the path lives on details only",
    ),
    ("config_state_manager.py", "_load_snapshot_from_file", "log"): (
        2,
        _LABEL + "; refusal text via diag.fixed_reason",
    ),
    ("config_state_manager.py", "_refuse_legacy_file", "log"): (
        1,
        "logs the already-built message: _LABEL text with a literal reason",
    ),
    ("config_state_manager.py", "_load_legacy_snapshot", "log"): (2, _LABEL),
    ("config_state_manager.py", "_verify_top_level_identity", "raise"): (
        1,
        "key is one of the SDK's fixed top-level field names",
    ),
    ("config_state_manager.py", "reset_optimization", "log"): (1, _COUNT),
    ("config_state_manager.py", "set_config", "log"): (1, _COUNT),
    ("config_state_manager.py", "apply_best_config", "log"): (1, _COUNT),
    ("config_state_manager.py", "maybe_auto_load_config", "log"): (1, _COUNT),
    ("config_state_manager.py", "_read_legacy_payload", "log"): (
        6,
        _LABEL + "; JSON decoder .msg is a fixed library phrase; exceptions by type",
    ),
    ("config_state_manager.py", "export_config", "raise"): (
        1,
        "echoes the caller's own 'format' argument (slim/full choice), not artifact content",
    ),
    ("config_state_manager.py", "export_config", "log"): (1, _COUNT),
    ("config_state_manager.py", "_promotion_precondition_kwargs", "log"): (
        1,
        "environment names are the caller's own run configuration, not artifact content",
    ),
    ("backend_client.py", "_raise_for_transient_backend_response", "raise"): (
        1,
        "operation literal + safe_http_error_summary (status code and fixed summary only)",
    ),
    ("backend_client.py", "_run_sync_backend_request_with_retry", "raise"): (
        3,
        "operation literal + pre-sanitized transient text or exception type name",
    ),
    ("backend_client.py", "_get_sync_auth_headers", "log"): (1, _COUNT),
    ("backend_client.py", "_get_sync_auth_headers", "raise"): (1, _COUNT),
    ("backend_client.py", "_protected_request", "raise"): (1, _COUNT),
    ("backend_client.py", "fetch_agent_head_generation_sync", "raise"): (
        1,
        "safe_http_error_summary only",
    ),
    ("backend_client.py", "publish_best_config_sync", "raise"): (
        1,
        "safe_http_error_summary only",
    ),
    ("backend_client.py", "fetch_best_config_sync", "raise"): (
        1,
        "safe_http_error_summary only",
    ),
    ("backend_client.py", "upload_example_features", "log"): (5, _LEGACY_BACKEND),
    ("backend_client.py", "__init__", "log"): (2, _LEGACY_BACKEND),
    ("backend_client.py", "_normalize_interaction_policy_payload", "raise"): (
        2,
        "missing-field names of the backend's fixed policy schema",
    ),
    ("backend_client.py", "get_interaction_policy", "raise"): (
        1,
        "safe_http_error_summary only",
    ),
    ("backend_client.py", "get_interaction_policy", "log"): (1, _LEGACY_BACKEND),
    ("backend_client.py", "_check_rate_limit", "raise"): (1, "a computed wait time"),
    ("backend_client.py", "_register_security_session", "log"): (1, _LEGACY_BACKEND),
    ("backend_client.py", "_revoke_security_session", "log"): (1, _LEGACY_BACKEND),
    ("backend_client.py", "validate_session_token", "log"): (1, _LEGACY_BACKEND),
    ("backend_client.py", "__aenter__", "log"): (1, _LEGACY_BACKEND),
    ("backend_client.py", "_build_http_session", "log"): (1, _LEGACY_BACKEND),
    ("backend_client.py", "_build_http_session", "raise"): (1, _LEGACY_BACKEND),
    ("backend_client.py", "_reset_http_session", "log"): (1, _LEGACY_BACKEND),
    ("backend_client.py", "_build_feature_upload_url", "log"): (1, _LEGACY_BACKEND),
    ("backend_client.py", "_stale_head_error_from_response", "log"): (
        1,
        _LEGACY_BACKEND,
    ),
    ("backend_client.py", "submit_result", "log"): (2, _LEGACY_BACKEND),
    ("optimization_result_persistence.py", "decode_trial_error", "log"): (1, _COUNT),
    ("optimization_result_persistence.py", "decode_trial_score", "log"): (1, _COUNT),
    ("optimization_result_persistence.py", "_decode_status", "log"): (1, _COUNT),
    ("optimization_result_persistence.py", "_read_field", "log"): (
        1,
        "dataclass field name + type name (SDK-owned)",
    ),
    ("optimization_result_persistence.py", "_read_field", "raise"): (
        1,
        "dataclass field name + type name (SDK-owned); encode side, no artifact content",
    ),
    ("optimization_result_persistence.py", "_encode_timestamp", "raise"): (
        1,
        "encode side: the caller's own in-memory value, raised before anything is written",
    ),
    ("optimization_result_persistence.py", "_decode_timestamp", "raise"): (1, _DECODER),
    ("optimization_result_persistence.py", "_decode_legacy", "raise"): (1, _DECODER),
    ("optimization_result_persistence.py", "_decode_legacy", "log"): (
        1,
        "legacy format literal + SDK dataclass field names; no artifact name",
    ),
    ("optimization_result_persistence.py", "_introduced_in", "raise"): (
        1,
        "SDK-internal consistency error naming SDK-owned field names",
    ),
    ("optimization_result_persistence.py", "_decode_versioned", "raise"): (1, _DECODER),
    ("optimization_result_persistence.py", "_decode_versioned", "log"): (
        1,
        "interpolates the artifact label: only reached via decode_result; the "
        "best-config caller passes no artifact_name ('<unnamed>'); persistence.py "
        "callers name their own artifacts",
    ),
    ("optimization_result_persistence.py", "verify_envelope_version", "raise"): (
        2,
        _DECODER,
    ),
    ("optimization_result_persistence.py", "decode_result", "raise"): (4, _DECODER),
}


def _is_const(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant)


def _variable_message(node: ast.AST) -> bool:
    """True when ``node`` builds text from something other than constants."""
    if isinstance(node, ast.JoinedStr):
        return any(isinstance(v, ast.FormattedValue) for v in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mod, ast.Add)):
        return not (_is_const(node.left) and _is_const(node.right))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "format"
    ):
        return True
    return False


def _is_log_call(node: ast.Call) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr in LOG_METHODS
        and isinstance(func.value, ast.Name)
        and func.value.id in {"logger", "log", "_logger", "logging"}
    )


class _Scan(ast.NodeVisitor):
    def __init__(self, name: str) -> None:
        self.name = name
        self.stack: list[str] = []
        self.found: list[tuple[str, str, str, int]] = []

    def _fn(self, node: ast.AST) -> None:
        self.stack.append(getattr(node, "name", "?"))
        self.generic_visit(node)
        self.stack.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = _fn  # noqa: N815

    def visit_Call(self, node: ast.Call) -> None:
        if _is_log_call(node):
            args = list(node.args)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "log":
                args = args[1:]  # drop the level
            extra = args[1:] or [kw for kw in node.keywords if kw.arg != "exc_info"]
            first = args[0] if args else None
            if (first is not None and _variable_message(first)) or extra:
                self.found.append(
                    (self.name, ".".join(self.stack) or "<module>", "log", node.lineno)
                )
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        exc = node.exc
        if isinstance(exc, ast.Call) and any(_variable_message(a) for a in exc.args):
            self.found.append(
                (self.name, ".".join(self.stack) or "<module>", "raise", node.lineno)
            )
        self.generic_visit(node)


def _sites() -> list[tuple[str, str, str, int]]:
    out: list[tuple[str, str, str, int]] = []
    for path in FILES:
        if path.name in EXEMPT:
            continue
        scan = _Scan(path.name)
        scan.visit(ast.parse(path.read_text(encoding="utf-8")))
        out.extend(scan.found)
    return out


def test_files_exist() -> None:
    assert all(p.exists() for p in FILES)
    assert len(FILES) >= 5


def test_no_new_value_bearing_log_or_raise() -> None:
    counts: dict[tuple[str, str, str], list[int]] = {}
    for name, fn, kind, line in _sites():
        counts.setdefault((name, fn, kind), []).append(line)
    unexpected = []
    for key, lines in sorted(counts.items()):
        allowed = ALLOWLIST.get(key, (0, ""))[0]
        if len(lines) > allowed:
            unexpected.append(
                f"{key[0]} {key[2]} in {key[1]}: lines {lines} (allowed {allowed})"
            )
    assert not unexpected, (
        "value-bearing log/raise on a best-config path (route through "
        "best_config_diagnostics.log/refusal, or allowlist with a reason):\n"
        + "\n".join(unexpected)
    )


def test_allowlist_is_tight_and_reasoned() -> None:
    counts: dict[tuple[str, str, str], int] = {}
    for name, fn, kind, _ in _sites():
        counts[(name, fn, kind)] = counts.get((name, fn, kind), 0) + 1
    loose = [
        key
        for key, (n, reason) in ALLOWLIST.items()
        if counts.get(key, 0) != n or not reason.strip()
    ]
    assert not loose, f"stale / loose / reasonless allowlist entries: {loose}"


def test_guard_catches_a_new_leaky_site() -> None:
    tree = ast.parse(
        "def f(path):\n"
        "    logger.warning('bad %s', path)\n"
        "    raise ValueError(f'bad {path}')\n"
        "    logger.info('fixed')\n"
    )
    scan = _Scan("x.py")
    scan.visit(tree)
    assert sorted(k for _, _, k, _ in scan.found) == ["log", "raise"]
