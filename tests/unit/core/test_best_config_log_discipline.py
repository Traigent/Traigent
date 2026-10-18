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
import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3] / "traigent"
FILES = (
    *sorted((ROOT / "core").glob("best_config_*.py")),
    ROOT / "core" / "config_state_manager.py",
    ROOT / "cloud" / "backend_client.py",
    ROOT / "utils" / "optimization_result_persistence.py",
    ROOT / "labels" / "transport_gate.py",
)
EXEMPT = {"best_config_diagnostics.py"}
LOG_METHODS = {"debug", "info", "warning", "error", "exception", "critical", "log"}

# (file name, enclosing function, kind, sha256[:16] of the site's exact expression text)
# -> (number of identical sites, reason). Binding each entry to the hash of the exact
# expression (and of the assignments that build a prepared message variable) means a safe
# interpolation cannot be swapped for an unsafe one while the count stays the same: the
# digest changes and the site is unexpected. A site is allowed only if every interpolated
# value is one of: a count, an exception *type* name, a code identifier (function name,
# GNFR-PRIV-003), a fixed label derived from the source enum (``_source_label``), an
# SDK-owned constant, or text the SDK wrote itself. Regenerate a digest only after
# re-reading the new expression; do not copy it blindly.
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
ALLOWLIST: dict[tuple[str, str, str, str], tuple[int, str]] = {
    (
        "backend_client.py",
        "BackendIntegratedClient.__aenter__",
        "log",
        "430208ba8ae23af1",
    ): (
        1,
        _LEGACY_BACKEND,
    ),
    (
        "backend_client.py",
        "BackendIntegratedClient._build_feature_upload_url",
        "log",
        "9d951cd22b5cb5e5",
    ): (
        1,
        _LEGACY_BACKEND,
    ),
    (
        "backend_client.py",
        "BackendIntegratedClient._build_http_session",
        "log",
        "b5132b1593040e79",
    ): (
        1,
        _LEGACY_BACKEND,
    ),
    (
        "backend_client.py",
        "BackendIntegratedClient._build_http_session",
        "raise",
        "8f916c9046c70d0f",
    ): (
        1,
        _LEGACY_BACKEND,
    ),
    (
        "backend_client.py",
        "BackendIntegratedClient._check_rate_limit",
        "raise",
        "76c8c10f5574f6d6",
    ): (
        1,
        "a computed wait time",
    ),
    (
        "backend_client.py",
        "BackendIntegratedClient._get_sync_auth_headers",
        "log",
        "951944a31db78294",
    ): (
        1,
        _COUNT,
    ),
    (
        "backend_client.py",
        "BackendIntegratedClient._get_sync_auth_headers",
        "raise",
        "6be35cafdb5e84f5",
    ): (
        1,
        _COUNT,
    ),
    (
        "backend_client.py",
        "BackendIntegratedClient._normalize_interaction_policy_payload",
        "raise",
        "24d1c6f00285c2f4",
    ): (
        1,
        "missing-field names of the backend's fixed policy schema",
    ),
    (
        "backend_client.py",
        "BackendIntegratedClient._normalize_interaction_policy_payload",
        "raise",
        "ffcb05f152b3df8a",
    ): (
        1,
        "missing-field names of the backend's fixed policy schema",
    ),
    (
        "backend_client.py",
        "BackendIntegratedClient._protected_request",
        "raise",
        "d9d06232d3947c89",
    ): (
        1,
        _COUNT,
    ),
    (
        "backend_client.py",
        "BackendIntegratedClient._register_security_session",
        "log",
        "89a621867608b974",
    ): (
        1,
        _LEGACY_BACKEND,
    ),
    (
        "backend_client.py",
        "BackendIntegratedClient._reset_http_session",
        "log",
        "840e8ecfd0d065b3",
    ): (
        1,
        _LEGACY_BACKEND,
    ),
    (
        "backend_client.py",
        "BackendIntegratedClient._revoke_security_session",
        "log",
        "54c3713d1a4dbc57",
    ): (
        1,
        _LEGACY_BACKEND,
    ),
    (
        "backend_client.py",
        "BackendIntegratedClient._stale_head_error_from_response",
        "log",
        "ee39547a44750122",
    ): (
        1,
        _LEGACY_BACKEND,
    ),
    (
        "backend_client.py",
        "BackendIntegratedClient.fetch_agent_head_generation_sync",
        "raise",
        "8209411c7dd78e61",
    ): (
        1,
        "safe_http_error_summary only",
    ),
    (
        "backend_client.py",
        "BackendIntegratedClient.fetch_best_config_sync",
        "raise",
        "e7f0bf7ea4a2be08",
    ): (
        1,
        "safe_http_error_summary only",
    ),
    (
        "backend_client.py",
        "BackendIntegratedClient.get_interaction_policy",
        "log",
        "4c96e9ad1ff2de51",
    ): (
        1,
        _LEGACY_BACKEND,
    ),
    (
        "backend_client.py",
        "BackendIntegratedClient.get_interaction_policy",
        "raise",
        "7aca6e937c1cd55d",
    ): (
        1,
        "safe_http_error_summary only",
    ),
    (
        "backend_client.py",
        "BackendIntegratedClient.publish_best_config_sync",
        "raise",
        "b059b7be5a3ecf39",
    ): (
        1,
        "safe_http_error_summary only",
    ),
    (
        "backend_client.py",
        "BackendIntegratedClient.submit_result",
        "log",
        "e81c0ab0dc7c6acc",
    ): (
        1,
        _LEGACY_BACKEND,
    ),
    (
        "backend_client.py",
        "BackendIntegratedClient.upload_example_features",
        "log",
        "66762c532b1f6fea",
    ): (
        1,
        _LEGACY_BACKEND,
    ),
    (
        "backend_client.py",
        "BackendIntegratedClient.upload_example_features",
        "log",
        "684478ff85c030ae",
    ): (
        1,
        _LEGACY_BACKEND,
    ),
    (
        "backend_client.py",
        "BackendIntegratedClient.upload_example_features",
        "log",
        "b61870d25225ca50",
    ): (
        1,
        _LEGACY_BACKEND,
    ),
    (
        "backend_client.py",
        "BackendIntegratedClient.upload_example_features",
        "log",
        "bd71d7af37378c22",
    ): (
        1,
        _LEGACY_BACKEND,
    ),
    (
        "backend_client.py",
        "BackendIntegratedClient.upload_example_features",
        "log",
        "f984be89a4d94951",
    ): (
        1,
        _LEGACY_BACKEND,
    ),
    (
        "backend_client.py",
        "BackendIntegratedClient.validate_session_token",
        "log",
        "d9f6ca85645982d3",
    ): (
        1,
        _LEGACY_BACKEND,
    ),
    (
        "backend_client.py",
        "PurposeKeyGrantFetchError.__init__",
        "exc_class",
        "8660996a06a105a3",
    ): (
        1,
        "status code (int) and error_code that _envelope_error_code restricts to a plain token (_SAFE_ERROR_CODE_RE)",
    ),
    (
        "backend_client.py",
        "_raise_for_transient_backend_response",
        "raise",
        "5b157c2d6f67eef2",
    ): (
        1,
        "operation literal + safe_http_error_summary (status code and fixed summary only)",
    ),
    (
        "backend_client.py",
        "_run_sync_backend_request_with_retry",
        "raise",
        "60b7196a634e59f4",
    ): (
        1,
        "operation literal + pre-sanitized transient text or exception type name",
    ),
    (
        "backend_client.py",
        "_run_sync_backend_request_with_retry",
        "raise",
        "892a75e340f4b806",
    ): (
        1,
        "operation literal + pre-sanitized transient text or exception type name",
    ),
    (
        "backend_client.py",
        "_run_sync_backend_request_with_retry",
        "raise",
        "dbc1ab4451f92868",
    ): (
        1,
        "operation literal + pre-sanitized transient text or exception type name",
    ),
    ("best_config_runtime.py", "_normalize_json_value", "raise", "b59a8129cf386185"): (
        1,
        _COUNT,
    ),
    (
        "best_config_runtime.py",
        "_validate_and_filter_config_keys",
        "log",
        "3f5d12b9f5919481",
    ): (
        1,
        _COUNT,
    ),
    (
        "config_state_manager.py",
        "ConfigStateManager._load_legacy_snapshot",
        "log",
        "b1fbc64b907b761d",
    ): (
        1,
        _LABEL,
    ),
    (
        "config_state_manager.py",
        "ConfigStateManager._load_legacy_snapshot",
        "log",
        "cea3e679939c85b5",
    ): (
        1,
        _LABEL,
    ),
    (
        "config_state_manager.py",
        "ConfigStateManager._load_snapshot_from_file",
        "log",
        "306e50e6919ef056",
    ): (
        1,
        "interpolates only _source_label() (fixed enum-derived label) and fixed reasons; refusal text via diag.fixed_reason",
    ),
    (
        "config_state_manager.py",
        "ConfigStateManager._load_snapshot_from_file",
        "log",
        "c93a48b67cfaffc1",
    ): (
        1,
        "interpolates only _source_label() (fixed enum-derived label) and fixed reasons; refusal text via diag.fixed_reason",
    ),
    (
        "config_state_manager.py",
        "ConfigStateManager._promotion_precondition_kwargs",
        "log",
        "9b0a7718b6396383",
    ): (
        1,
        "environment names are the caller's own run configuration, not artifact content",
    ),
    (
        "config_state_manager.py",
        "ConfigStateManager._read_legacy_payload",
        "log",
        "25567ea31b9886c8",
    ): (
        1,
        "interpolates only _source_label() (fixed enum-derived label) and fixed reasons; JSON decoder .msg is a fixed library phrase; exceptions by type",
    ),
    (
        "config_state_manager.py",
        "ConfigStateManager._read_legacy_payload",
        "log",
        "28b6de4dbecb85f7",
    ): (
        1,
        "interpolates only _source_label() (fixed enum-derived label) and fixed reasons; JSON decoder .msg is a fixed library phrase; exceptions by type",
    ),
    (
        "config_state_manager.py",
        "ConfigStateManager._read_legacy_payload",
        "log",
        "493a8d9ba51427b3",
    ): (
        1,
        "interpolates only _source_label() (fixed enum-derived label) and fixed reasons; JSON decoder .msg is a fixed library phrase; exceptions by type",
    ),
    (
        "config_state_manager.py",
        "ConfigStateManager._read_legacy_payload",
        "log",
        "abd85b78d1f1cbbf",
    ): (
        1,
        "interpolates only _source_label() (fixed enum-derived label) and fixed reasons; JSON decoder .msg is a fixed library phrase; exceptions by type",
    ),
    (
        "config_state_manager.py",
        "ConfigStateManager._read_legacy_payload",
        "log",
        "d23c9a9333e0d04c",
    ): (
        1,
        "interpolates only _source_label() (fixed enum-derived label) and fixed reasons; JSON decoder .msg is a fixed library phrase; exceptions by type",
    ),
    (
        "config_state_manager.py",
        "ConfigStateManager._read_legacy_payload",
        "log",
        "de9ebe4d05039de2",
    ): (
        1,
        "interpolates only _source_label() (fixed enum-derived label) and fixed reasons; JSON decoder .msg is a fixed library phrase; exceptions by type",
    ),
    (
        "config_state_manager.py",
        "ConfigStateManager._read_spec_file",
        "raise",
        "e9f0fd0c5a65be0b",
    ): (
        1,
        "interpolates only _source_label() (fixed enum-derived label) and fixed reasons; the path lives on details only",
    ),
    (
        "config_state_manager.py",
        "ConfigStateManager._refuse_legacy_file",
        "log",
        "5491ed5b0b45275f",
    ): (
        1,
        "logs the already-built message: _LABEL text with a literal reason",
    ),
    (
        "config_state_manager.py",
        "ConfigStateManager._refuse_legacy_file",
        "raise",
        "b647783b06ba48e3",
    ): (
        1,
        "message is _LABEL text (fixed enum-derived label) plus a literal reason; the path lives on details only",
    ),
    (
        "config_state_manager.py",
        "ConfigStateManager._resolve_mode_snapshot",
        "log",
        "3619248b5b4b60ab",
    ): (
        1,
        "interpolates only source.value (the closed best-config source enum) via diag.log with a fixed category",
    ),
    (
        "config_state_manager.py",
        "ConfigStateManager._resolve_mode_snapshot",
        "log",
        "596663837dba5a91",
    ): (
        1,
        "interpolates only source.value (the closed best-config source enum) via diag.log with a fixed category",
    ),
    (
        "config_state_manager.py",
        "ConfigStateManager._verify_top_level_identity",
        "raise",
        "e374dbea7f7ea7ce",
    ): (
        1,
        "key is one of the SDK's fixed top-level field names",
    ),
    (
        "config_state_manager.py",
        "ConfigStateManager.apply_best_config",
        "log",
        "e73375ab325457d5",
    ): (
        1,
        _COUNT,
    ),
    (
        "config_state_manager.py",
        "ConfigStateManager.export_config",
        "log",
        "d38a31596f58d8ce",
    ): (
        1,
        _COUNT,
    ),
    (
        "config_state_manager.py",
        "ConfigStateManager.export_config",
        "raise",
        "91f853c9fe0f5aa8",
    ): (
        1,
        "echoes the caller's own 'format' argument (slim/full choice), not artifact content",
    ),
    (
        "config_state_manager.py",
        "ConfigStateManager.maybe_auto_load_config",
        "log",
        "66e853dcebdc2f1e",
    ): (
        1,
        _COUNT,
    ),
    (
        "config_state_manager.py",
        "ConfigStateManager.reset_optimization",
        "log",
        "6eae13aaa89e5503",
    ): (
        1,
        _COUNT,
    ),
    (
        "config_state_manager.py",
        "ConfigStateManager.set_config",
        "log",
        "83bb83574354c0ca",
    ): (
        1,
        _COUNT,
    ),
    (
        "optimization_result_persistence.py",
        "_decode_legacy",
        "log",
        "525f6bdf25886530",
    ): (
        1,
        "legacy format literal + SDK dataclass field names; no artifact name",
    ),
    (
        "optimization_result_persistence.py",
        "_decode_legacy",
        "raise",
        "5d1bc6e25c5fa25d",
    ): (
        1,
        _DECODER,
    ),
    (
        "optimization_result_persistence.py",
        "_decode_status",
        "log",
        "cadf9043f85d2cd8",
    ): (
        1,
        _COUNT,
    ),
    (
        "optimization_result_persistence.py",
        "_decode_timestamp",
        "raise",
        "5fe7e0b228f39f82",
    ): (
        1,
        _DECODER,
    ),
    (
        "optimization_result_persistence.py",
        "_decode_versioned",
        "log",
        "e01c45b109519da2",
    ): (
        1,
        "interpolates the artifact label: only reached via decode_result; the best-config caller passes no artifact_name ('<unnamed>'); persistence.py callers name their own artifacts",
    ),
    (
        "optimization_result_persistence.py",
        "_decode_versioned",
        "raise",
        "83de48c3e6071b70",
    ): (
        1,
        _DECODER,
    ),
    (
        "optimization_result_persistence.py",
        "_encode_timestamp",
        "raise",
        "eabcfa82bec244e0",
    ): (
        1,
        "encode side: the caller's own in-memory value, raised before anything is written",
    ),
    (
        "optimization_result_persistence.py",
        "_introduced_in",
        "raise",
        "aa8985ff8d298e6f",
    ): (
        1,
        "SDK-internal consistency error naming SDK-owned field names",
    ),
    ("optimization_result_persistence.py", "_read_field", "log", "3e05b4376b977acb"): (
        1,
        "dataclass field name + type name (SDK-owned)",
    ),
    (
        "optimization_result_persistence.py",
        "_read_field",
        "raise",
        "453e6b7a9986b0c7",
    ): (
        1,
        "dataclass field name + type name (SDK-owned); encode side, no artifact content",
    ),
    (
        "optimization_result_persistence.py",
        "decode_result",
        "raise",
        "0578d5df7a64095a",
    ): (
        1,
        _DECODER,
    ),
    (
        "optimization_result_persistence.py",
        "decode_result",
        "raise",
        "126a2d1e119cabbf",
    ): (
        1,
        _DECODER,
    ),
    (
        "optimization_result_persistence.py",
        "decode_result",
        "raise",
        "21404594fb6d3e6d",
    ): (
        1,
        _DECODER,
    ),
    (
        "optimization_result_persistence.py",
        "decode_result",
        "raise",
        "881773e24affc973",
    ): (
        1,
        _DECODER,
    ),
    (
        "optimization_result_persistence.py",
        "decode_trial_error",
        "log",
        "f6855889ebbc9a3b",
    ): (
        1,
        _COUNT,
    ),
    (
        "optimization_result_persistence.py",
        "decode_trial_score",
        "log",
        "2e0fce893c336eb6",
    ): (
        1,
        _COUNT,
    ),
    (
        "optimization_result_persistence.py",
        "verify_envelope_version",
        "raise",
        "1114c29897b1ff84",
    ): (
        1,
        _DECODER,
    ),
    (
        "optimization_result_persistence.py",
        "verify_envelope_version",
        "raise",
        "ee54c085717d16c2",
    ): (
        1,
        _DECODER,
    ),
    (
        "transport_gate.py",
        "TransportGateRefusal.__init__",
        "exc_class",
        "2ce69a653d075128",
    ): (
        1,
        "route is a table template (never the request path) and the dict holds only violation-class counts",
    ),
    ("transport_gate.py", "inspect_outgoing", "log", "6155e24cf2fbfce4"): (
        1,
        "debug line of mode, route template and violation-class counts; no request path, header or body",
    ),
}


def _is_const(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant)


def _text_args(call: ast.Call) -> list[ast.AST]:
    return [*call.args, *(kw.value for kw in call.keywords if kw.arg is None)]


def _formats_value(node: ast.AST) -> bool:
    """True when ``node`` builds text from non-constants or stringifies a value."""
    if isinstance(node, ast.JoinedStr):
        return any(isinstance(v, ast.FormattedValue) for v in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mod, ast.Add)):
        return not (_is_const(node.left) and _is_const(node.right))
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "format":
            return True
        if isinstance(func, ast.Name) and func.id in {"str", "repr", "format"}:
            return True
    return False


def _variable_message(node: ast.AST) -> bool:
    """A message that is anything but a plain constant string."""
    if isinstance(node, ast.JoinedStr):
        return _formats_value(node)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mod, ast.Add)):
        return not (_is_const(node.left) and _is_const(node.right))
    return not _is_const(node)


_LOGGER_NAMES = {"logger", "log", "_logger", "logging", "LOGGER", "_LOGGER"}


def _is_logger_factory(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
        and getattr(node.func, "id", getattr(node.func, "attr", ""))
        in {"getLogger", "get_logger"}
    )


class _Aliases(ast.NodeVisitor):
    """Names bound to a logger, and names bound to a logger's bound log method."""

    def __init__(self) -> None:
        self.loggers: set[str] = set(_LOGGER_NAMES)
        self.bound: dict[str, str] = {}

    def _loggerish(self, expr: ast.AST) -> bool:
        if isinstance(expr, ast.Name):
            return expr.id in self.loggers
        if isinstance(expr, ast.Attribute):
            return expr.attr in self.loggers
        return _is_logger_factory(expr)

    def visit_Assign(self, node: ast.Assign) -> None:
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        value = node.value
        if self._loggerish(value):
            self.loggers.update(names)
        elif (
            isinstance(value, ast.Attribute)
            and value.attr in LOG_METHODS
            and self._loggerish(value.value)
        ):
            for name in names:
                self.bound[name] = value.attr
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        annotation = ast.unparse(node.annotation) if node.annotation else ""
        if node.arg in _LOGGER_NAMES or "Logger" in annotation:
            self.loggers.add(node.arg)


def _source_key(node: ast.AST, extra: list[ast.AST]) -> str:
    """Exact expression text of a site (and of the assignments feeding its message)."""
    parts = [ast.unparse(node), *(ast.unparse(e) for e in extra)]
    return "\n".join(parts)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class _Scan(ast.NodeVisitor):
    def __init__(self, name: str, aliases: _Aliases | None = None) -> None:
        self.name = name
        self.aliases = aliases or _Aliases()
        self.stack: list[str] = []
        self.funcs: list[ast.AST] = []
        # (file, function, kind, digest of exact expression text, line)
        self.found: list[tuple[str, str, str, str, int]] = []

    def _where(self) -> str:
        return ".".join(self.stack) or "<module>"

    def _fn(self, node: ast.AST) -> None:
        self.stack.append(getattr(node, "name", "?"))
        self.funcs.append(node)
        self.generic_visit(node)
        self.funcs.pop()
        self.stack.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = _fn  # noqa: N815

    def _feeders(self, message: ast.AST) -> list[ast.AST]:
        """Assignments to a prepared message variable, in the enclosing function."""
        if not isinstance(message, ast.Name) or not self.funcs:
            return []
        out: list[ast.AST] = []
        for sub in ast.walk(self.funcs[-1]):
            if isinstance(sub, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == message.id for t in sub.targets
            ):
                out.append(sub)
            elif (
                isinstance(sub, (ast.AugAssign, ast.AnnAssign))
                and isinstance(sub.target, ast.Name)
                and sub.target.id == message.id
            ):
                out.append(sub)
        return out

    def _add(self, kind: str, node: ast.AST, extra: list[ast.AST], line: int) -> None:
        self.found.append(
            (self.name, self._where(), kind, digest(_source_key(node, extra)), line)
        )

    def _sink(self, node: ast.Call) -> tuple[str, list[ast.AST]] | None:
        """Classify a call as a diagnostic sink: ``(method, message-bearing args)``."""
        func = node.func
        args = list(node.args)
        if isinstance(func, ast.Attribute):
            if func.attr in LOG_METHODS and self.aliases._loggerish(func.value):
                if func.attr == "log":
                    args = args[1:]
                return func.attr, args
            if (
                func.attr == "log"
                and isinstance(func.value, ast.Name)
                and func.value.id == "diag"
            ):
                msg = args[2:3] or [
                    kw.value for kw in node.keywords if kw.arg == "message"
                ]
                return "diag.log", msg
            if (
                func.attr == "warn"
                and isinstance(func.value, ast.Name)
                and func.value.id == "warnings"
            ):
                return "warn", args[:1]
        if isinstance(func, ast.Name):
            if func.id in self.aliases.bound:
                method = self.aliases.bound[func.id]
                return method, args[1:] if method == "log" else args
            if func.id == "print":
                return "print", args
        return None

    def visit_Call(self, node: ast.Call) -> None:
        sink = self._sink(node)
        if sink is not None:
            method, msgs = sink
            first = msgs[0] if msgs else None
            if method in {"print"}:
                bad = any(_variable_message(m) for m in msgs)
            elif method == "diag.log":
                bad = first is not None and _variable_message(first)
            else:
                extra = msgs[1:] or [
                    kw
                    for kw in node.keywords
                    if kw.arg not in {"exc_info", "stacklevel"}
                ]
                bad = (first is not None and _variable_message(first)) or bool(extra)
            exc_kw = next((kw for kw in node.keywords if kw.arg == "exc_info"), None)
            traceback_bearing = method == "exception" or (
                exc_kw is not None
                and not (
                    isinstance(exc_kw.value, ast.Constant)
                    and exc_kw.value.value in (False, None)
                )
            )
            if bad or traceback_bearing:
                feeders = self._feeders(first) if first is not None else []
                self._add("log", node, feeders, node.lineno)
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        exc = node.exc
        if isinstance(exc, ast.Call):
            feeders = [f for a in exc.args for f in self._feeders(a)]
            builds_text = any(_formats_value(a) for a in exc.args) or any(
                _formats_value(getattr(f, "value", None) or f) for f in feeders
            )
            if builds_text:
                self._add("raise", node, feeders, node.lineno)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        bases = [ast.unparse(b) for b in node.bases]
        if any(b.endswith(("Error", "Exception", "Refusal")) for b in bases):
            for item in node.body:
                if (
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name in {"__init__", "__str__", "__repr__"}
                    and any(_formats_value(n) for n in ast.walk(item))
                ):
                    self.found.append(
                        (
                            self.name,
                            f"{'.'.join([*self.stack, node.name])}.{item.name}",
                            "exc_class",
                            digest(ast.unparse(item)),
                            item.lineno,
                        )
                    )
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()


def scan_source(name: str, source: str) -> list[tuple[str, str, str, str, int]]:
    tree = ast.parse(source)
    aliases = _Aliases()
    aliases.visit(tree)
    scan = _Scan(name, aliases)
    scan.visit(tree)
    return scan.found


def _sites() -> list[tuple[str, str, str, str, int]]:
    out: list[tuple[str, str, str, str, int]] = []
    for path in FILES:
        if path.name in EXEMPT:
            continue
        out.extend(scan_source(path.name, path.read_text(encoding="utf-8")))
    return out


def _site_counts() -> dict[tuple[str, str, str, str], int]:
    counts: dict[tuple[str, str, str, str], int] = {}
    for name, fn, kind, dig, _ in _sites():
        key = (name, fn, kind, dig)
        counts[key] = counts.get(key, 0) + 1
    return counts


def test_files_exist() -> None:
    assert all(p.exists() for p in FILES)
    assert len(FILES) >= 6
    assert "transport_gate.py" in {p.name for p in FILES}


def test_no_new_or_changed_value_bearing_site() -> None:
    """Every flagged site is allowlisted by the hash of its exact expression text."""
    unexpected = []
    for key, n in sorted(_site_counts().items()):
        allowed = ALLOWLIST.get(key, (0, ""))[0]
        if n > allowed:
            unexpected.append(f"{key} x{n} (allowed {allowed})")
    assert not unexpected, (
        "value-bearing / traceback-bearing diagnostic on a best-config path, or a "
        "previously allowed expression changed (route through "
        "best_config_diagnostics.log/refusal, or re-review and update ALLOWLIST with "
        "the new digest and a reason):\n" + "\n".join(unexpected)
    )


def test_allowlist_is_tight_and_reasoned() -> None:
    counts = _site_counts()
    loose = [
        key
        for key, (n, reason) in ALLOWLIST.items()
        if counts.get(key, 0) != n or not reason.strip()
    ]
    assert not loose, f"stale / loose / reasonless allowlist entries: {loose}"


def _kinds(source: str) -> list[str]:
    return sorted(k for _, _, k, _, _ in scan_source("x.py", source))


@pytest.mark.parametrize(
    "source, expected",
    [
        (
            "def f(p):\n logger.warning('bad %s', p)\n raise ValueError(f'bad {p}')\n",
            ["log", "raise"],
        ),
        ("def f(p):\n msg = f'bad {p}'\n logger.warning(msg)\n", ["log"]),
        ("def f(p):\n logger.warning(str(p))\n", ["log"]),
        ("def f(p):\n lg = logging.getLogger('x')\n lg.warning(p)\n", ["log"]),
        ("def f(p):\n alias = logger\n alias.info(f'{p}')\n", ["log"]),
        ("def f(self, p):\n self.logger.error(f'{p}')\n", ["log"]),
        ("def f(p):\n w = logger.warning\n w(f'{p}')\n", ["log"]),
        ("def f(p):\n warnings.warn(f'{p}')\n", ["log"]),
        ("def f(p):\n print(p)\n", ["log"]),
        ("def f(p):\n diag.log(logger, 30, f'{p}', category=c)\n", ["log"]),
        ("def f(p):\n logger.exception('fixed')\n", ["log"]),
        ("def f(p):\n logger.warning('fixed', exc_info=True)\n", ["log"]),
        ("def f(p):\n msg = 'bad ' + p\n raise ValueError(msg)\n", ["raise"]),
        ("def f(p):\n raise ValueError(str(p))\n", ["raise"]),
        (
            "class E(ValueError):\n def __init__(self, v):\n  super().__init__(f'bad {v}')\n",
            ["exc_class"],
        ),
        (
            "class E(Exception):\n def __str__(self):\n  return 'bad %s' % self.v\n",
            ["exc_class"],
        ),
    ],
)
def test_guard_catches_each_bypass(source: str, expected: list[str]) -> None:
    assert _kinds(source) == expected


@pytest.mark.parametrize(
    "source",
    [
        "def f():\n logger.warning('fixed')\n",
        "def f():\n logger.warning('fixed', exc_info=False)\n",
        "def f():\n raise ValueError('fixed')\n",
        "def f(e):\n diag.log(logger, 30, 'fixed', category=c, exc=e)\n",
        "class E(ValueError):\n def __init__(self):\n  super().__init__('fixed')\n",
    ],
)
def test_guard_accepts_fixed_literals(source: str) -> None:
    assert _kinds(source) == []


def test_swapping_a_safe_interpolation_for_an_unsafe_one_changes_the_key() -> None:
    safe = "def f(n):\n logger.info(f'count {n}')\n"
    unsafe = "def f(n):\n logger.info(f'count {n.path}')\n"
    (key_safe,) = [(n, f, k, d) for n, f, k, d, _ in scan_source("x.py", safe)]
    (key_unsafe,) = [(n, f, k, d) for n, f, k, d, _ in scan_source("x.py", unsafe)]
    assert key_safe[:3] == key_unsafe[:3] and key_safe != key_unsafe
    # an allowlist entry for the safe expression does not admit the unsafe one
    assert {key_safe: (1, "safe")}.get(key_unsafe) is None
