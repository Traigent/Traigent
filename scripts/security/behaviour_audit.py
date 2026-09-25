#!/usr/bin/env python3
"""Runtime behaviour recorder for the Traigent Python SDK (PEP 578 audit hooks).

Purpose: tell a customer's security team, *before* a deployment, what the SDK
does that a firewall, proxy, antivirus or EDR product may notice — outbound
connections, child processes, file writes, in-memory code generation, native
library loads — measured, not guessed.

How it works
------------
``run`` starts a throwaway HTTP stub on 127.0.0.1 (standing in for the Traigent
backend, so backend calls are *recorded, not sent*), then launches the chosen
workload in a FRESH Python subprocess so import-time behaviour is captured. The
child installs ``sys.addaudithook`` before importing anything from the SDK and
records, de-duplicated with the first-seen call site:

* ``socket.getaddrinfo`` / ``socket.connect`` / ``socket.sendto`` /
  ``socket.bind`` (host, port), plus ``http.client.connect`` and
  ``urllib.Request`` when the stdlib HTTP stack is used;
* ``subprocess.Popen`` / ``os.system`` / ``os.exec*`` / ``os.spawn*`` /
  ``os.posix_spawn`` / ``os.fork`` (argv[0] only);
* ``open()`` / ``os.open`` in a WRITE mode, ``os.mkdir``, ``os.rename``,
  ``tempfile.mkstemp`` / ``mkdtemp`` and executable-bit ``os.chmod``, with the
  path reduced to a class (cwd/.traigent, cwd, home/.traigent, home, temp dir,
  site-packages, /dev, other);
* ``compile()`` of code that is NOT an import (count + origin), ``exec``/
  ``eval`` of code objects, and assignments to ``function.__code__``;
* ``ctypes.dlopen`` and ``webbrowser.open``.

Imports of native extensions are deliberately not recorded (noise).

Audit hooks do NOT see inside child processes. Every process launch is
recorded (argv[0]) so it can be audited separately; the report says so.

By default every non-loopback DNS lookup / connect is recorded and then
BLOCKED (the hook raises ``OSError``), so running this tool never sends a
packet off the machine. ``--allow-egress`` lets connections through to measure
real-world behaviour.

Usage::

    python scripts/security/behaviour_audit.py list
    python scripts/security/behaviour_audit.py run optimize_mock \\
        --json out.json --markdown out.md
    python scripts/security/behaviour_audit.py run optimize_mock --pins

This file has no third-party imports; the child side must stay importable
before the SDK is.
"""

from __future__ import annotations

import argparse
import atexit
import ipaddress
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

_THIS_FILE = os.path.abspath(__file__)
_SCRIPTS_DIR = os.path.dirname(_THIS_FILE)

#: Env vars that stop the known third-party downloads. Kept in one place so
#: the test, the manifest and ``--pins`` agree.
NETWORK_PINS: dict[str, str] = {
    "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    "LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS": "True",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}

#: Well-formed but worthless key (traigent/cloud/api_key_manager.py
#: validate_format: ``tg_`` + 61 word characters). It only ever reaches the
#: loopback stub or a blocked lookup.
DUMMY_API_KEY = "tg_" + "behaviour_audit_dummy_key".ljust(61, "0")

#: Env vars that could make a workload reach a real service; cleared in the child.
_SCRUBBED_ENV_PREFIXES = (
    "OPENAI_",
    "ANTHROPIC_",
    "GEMINI_",
    "GOOGLE_",
    "AZURE_",
    "AWS_",
    "MISTRAL_",
    "COHERE_",
    "GROQ_",
    "TOGETHER",
    "HF_",
    "HUGGING",
    "TRANSFORMERS_",
    "LITELLM_",
    "LANGFUSE_",
    "LANGCHAIN_",
    "LANGSMITH_",
    "TRAIGENT_",
    "OTEL_",
    "PYTEST_",
    "COV_CORE_",
)

_NETWORK_EVENTS = {
    "socket.getaddrinfo",
    "socket.connect",
    "socket.sendto",
    "socket.bind",
    "http.client.connect",
    "urllib.Request",
}
_PROCESS_EVENTS = {
    "subprocess.Popen",
    "os.system",
    "os.exec",
    "os.spawn",
    "os.posix_spawn",
    "os.fork",
    "os.forkpty",
    "pty.spawn",
    "os.startfile",
}
_FILE_EVENTS = {
    "open",
    "os.mkdir",
    "os.rename",
    "tempfile.mkstemp",
    "tempfile.mkdtemp",
    "os.chmod",
}
_CODE_EVENTS = {"compile", "exec", "object.__setattr__"}
_NATIVE_EVENTS = {"ctypes.dlopen", "webbrowser.open"}
_WATCHED = (
    _NETWORK_EVENTS | _PROCESS_EVENTS | _FILE_EVENTS | _CODE_EVENTS | _NATIVE_EVENTS
)

_WRITE_FLAGS = (
    os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | getattr(os, "O_TRUNC", 0)
)
_LOOPBACK_NAMES = {"localhost", "localhost.localdomain", "ip6-localhost"}


# ---------------------------------------------------------------------------
# Child side: the recorder
# ---------------------------------------------------------------------------


class Recorder:
    """Collects de-duplicated audit events for one process."""

    def __init__(self, *, block_egress: bool, sdk_root: str, workload_file: str):
        self.block_egress = block_egress
        self.sdk_root = os.path.realpath(sdk_root) + os.sep
        self.repo_root = os.path.dirname(self.sdk_root.rstrip(os.sep)) + os.sep
        self.workload_file = os.path.realpath(workload_file)
        self.cwd = os.path.realpath(os.getcwd())
        self.home = os.path.realpath(os.path.expanduser("~"))
        self.tmp = os.path.realpath(tempfile.gettempdir())
        self.stdlib = os.path.realpath(os.path.dirname(os.__file__)) + os.sep
        self.records: dict[tuple[str, str], dict[str, Any]] = {}
        self.blocked: list[str] = []
        self._lock = threading.Lock()
        self._local = threading.local()
        self.started = time.monotonic()
        self.enabled = True

    # -- classification helpers ------------------------------------------

    def _frame_label(self, filename: str) -> str | None:
        """Return a package label for a frame filename, or None to skip it."""
        if filename.startswith("<frozen") or filename == _THIS_FILE:
            return None
        real = os.path.realpath(filename) if not filename.startswith("<") else ""
        if real.startswith(self.sdk_root):
            return "traigent"
        if real == self.workload_file:
            return "workload"
        for marker in ("site-packages", "dist-packages"):
            token = os.sep + marker + os.sep
            if token in real:
                rest = real.split(token, 1)[1]
                top = rest.split(os.sep, 1)[0]
                return top[:-3] if top.endswith(".py") else top
        if real.startswith(self.stdlib) or not real:
            return None
        return "other:" + os.path.basename(real)

    def _call_site(self) -> tuple[str, list[str], bool]:
        """(traigent module:line or first non-stdlib frame, package chain, is_import)."""
        frame = sys._getframe(1)
        while frame is not None and frame.f_code.co_filename == _THIS_FILE:
            frame = frame.f_back
        if frame is None:
            return "?", [], False
        innermost = frame.f_code.co_filename
        self._local.innermost = innermost
        is_import = innermost.startswith("<frozen importlib")
        site = ""
        chain: list[str] = []
        first_other = ""
        while frame is not None:
            fname = frame.f_code.co_filename
            label = self._frame_label(fname)
            if label is not None:
                if label not in chain:
                    chain.append(label)
                if label == "traigent" and not site:
                    rel = os.path.relpath(os.path.realpath(fname), self.repo_root)
                    site = f"{rel}:{frame.f_lineno}"
                if not first_other:
                    first_other = f"{label}:{os.path.basename(fname)}:{frame.f_lineno}"
            frame = frame.f_back
        return site or first_other or "stdlib/interpreter", chain[:6], is_import

    def classify_path(self, path: Any) -> str | None:
        if isinstance(path, int):
            return None
        try:
            p = os.fsdecode(path)
        except TypeError:
            return None
        real = os.path.realpath(os.path.abspath(p))
        if real.startswith("/dev/") or real.startswith("/proc/"):
            return "/dev,/proc"
        if "site-packages" in real or "dist-packages" in real:
            return "site-packages"
        # home before cwd: the driver places HOME inside the working dir.
        if real.startswith(self.home + os.sep):
            top = os.path.relpath(real, self.home).split(os.sep)[0]
            return f"home/{top}"
        if real.startswith(self.cwd + os.sep) or real == self.cwd:
            top = os.path.relpath(real, self.cwd).split(os.sep)[0]
            return f"cwd/{top}" if top != "." else "cwd"
        if real.startswith(self.tmp + os.sep):
            return "temp dir"
        if real.startswith(self.repo_root):
            return "sdk source tree"
        return "other"

    @staticmethod
    def _is_loopback(host: Any) -> bool:
        if host in (None, "", b""):
            return True
        if isinstance(host, bytes):
            host = host.decode("ascii", "replace")
        host = str(host).strip("[]").lower()
        if host in _LOOPBACK_NAMES:
            return True
        try:
            ip = ipaddress.ip_address(host.split("%", 1)[0])
        except ValueError:
            return False
        return ip.is_loopback or ip.is_unspecified

    # -- recording --------------------------------------------------------

    def _add(self, category: str, detail: str, site: str, chain: list[str]) -> None:
        key = (category, detail)
        with self._lock:
            rec = self.records.get(key)
            if rec is None:
                self.records[key] = {
                    "category": category,
                    "detail": detail,
                    "first_site": site,
                    "via": chain,
                    "count": 1,
                    "first_seen_s": round(time.monotonic() - self.started, 3),
                }
            else:
                rec["count"] += 1

    def hook(self, event: str, args: tuple[Any, ...]) -> None:
        if event not in _WATCHED or not self.enabled:
            return
        if getattr(self._local, "busy", False):
            return
        self._local.busy = True
        try:
            self._handle(event, args)
        finally:
            self._local.busy = False

    def _block(self, what: str) -> None:
        if self.block_egress:
            self.blocked.append(what)
            raise OSError(101, f"egress blocked by behaviour_audit: {what}")

    def _handle(self, event: str, args: tuple[Any, ...]) -> None:
        if event in _NETWORK_EVENTS:
            self._handle_network(event, args)
        elif event in _PROCESS_EVENTS:
            argv0 = _argv0(event, args)
            site, chain, _ = self._call_site()
            self._add("process", f"{event}: {argv0}", site, chain)
        elif event in _FILE_EVENTS:
            self._handle_file(event, args)
        elif event in _CODE_EVENTS:
            self._handle_code(event, args)
        elif event == "ctypes.dlopen":
            site, chain, _ = self._call_site()
            self._add("native", f"ctypes.dlopen: {args[0]}", site, chain)
        elif event == "webbrowser.open":
            site, chain, _ = self._call_site()
            self._add("process", f"webbrowser.open: {args[0]}", site, chain)

    def _handle_network(self, event: str, args: tuple[Any, ...]) -> None:
        site, chain, _ = self._call_site()
        if event == "socket.getaddrinfo":
            host, port = args[0], args[1]
            if isinstance(host, bytes):
                host = host.decode("ascii", "replace")
            self._add("dns", f"{host}:{port}", site, chain)
            if not self._is_loopback(host):
                self._block(f"getaddrinfo {host}:{port}")
            return
        if event in ("socket.connect", "socket.sendto", "socket.bind"):
            address = args[1]
            if isinstance(address, tuple) and len(address) >= 2:
                host, port = address[0], address[1]
                verb = event.split(".", 1)[1]
                self._add(verb, f"{host}:{port}", site, chain)
                if verb != "bind" and not self._is_loopback(host):
                    self._block(f"{verb} {host}:{port}")
            else:  # AF_UNIX path or abstract socket
                self._add(event.split(".", 1)[1], f"unix:{address!r}", site, chain)
            return
        if event == "http.client.connect":
            self._add("http", f"http.client {args[1]}:{args[2]}", site, chain)
            return
        if event == "urllib.Request":
            url = str(args[0])
            self._add(
                "http", f"urllib {args[3] or 'GET'} {_url_origin(url)}", site, chain
            )

    def _handle_file(self, event: str, args: tuple[Any, ...]) -> None:
        if event == "open":
            path, mode, flags = args[0], args[1], args[2]
            writes = False
            if isinstance(mode, str) and any(c in mode for c in "wax+"):
                writes = True
            elif isinstance(flags, int) and flags & _WRITE_FLAGS:
                writes = True
            if not writes:
                return
            cls = self.classify_path(path)
            if cls is None or cls == "/dev,/proc":
                return
            site, chain, is_import = self._call_site()
            if is_import:  # bytecode cache written by the import system
                self._add("file-write", f"{cls}: *.pyc (import cache)", "import", [])
                return
            self._add("file-write", f"{cls}: {_path_hint(path)}", site, chain)
            return
        if event in ("tempfile.mkstemp", "tempfile.mkdtemp", "os.mkdir"):
            cls = self.classify_path(args[0])
            if cls is None:
                return
            site, chain, is_import = self._call_site()
            if is_import:
                return
            self._add("file-write", f"{event} in {cls}", site, chain)
            return
        if event == "os.rename":
            cls = self.classify_path(args[1])
            if cls is None:
                return
            site, chain, is_import = self._call_site()
            if is_import:
                self._add("file-write", f"{cls}: *.pyc (import cache)", "import", [])
                return
            self._add("file-write", f"os.rename into {cls}", site, chain)
            return
        if event == "os.chmod":
            mode = args[1]
            if isinstance(mode, int) and mode & 0o111:
                cls = self.classify_path(args[0])
                site, chain, _ = self._call_site()
                self._add("file-exec-bit", f"chmod +x in {cls}", site, chain)

    def _handle_code(self, event: str, args: tuple[Any, ...]) -> None:
        if event == "object.__setattr__":
            if len(args) >= 2 and args[1] == "__code__":
                site, chain, _ = self._call_site()
                owner = chain[0] if chain else "stdlib"
                self._add("codegen", f"{owner}: assign function.__code__", site, chain)
            return
        site, chain, is_import = self._call_site()
        if is_import:
            return
        innermost = getattr(self._local, "innermost", "")
        # Name the stdlib generator (dataclasses, typing, collections, ...)
        # when the compile/exec comes from one, so a class definition that
        # triggers stdlib codegen is not mistaken for the SDK generating code.
        via_stdlib = (
            f" via stdlib {os.path.basename(innermost)}"
            if self._frame_label(innermost) is None
            else ""
        )
        if event == "compile":
            source, filename = args[0], args[1]
            kind = type(source).__name__
            if kind == "Module" or kind.endswith("AST"):
                kind = "AST"
            fname = os.fsdecode(filename) if filename is not None else None
            if kind == "AST" or fname is None:
                # CPython's "compile" audit event carries filename=None for an
                # AST input, so the origin is the call site, not the filename.
                origin = "an in-memory AST"
            elif fname.startswith("<"):
                origin = "in-memory source"
            else:
                origin = f"file-backed source ({self.classify_path(fname) or '?'})"
            owner = chain[0] if chain else "stdlib"
            self._add(
                "codegen",
                f"{owner}: compile({kind}) of {origin}{via_stdlib}",
                site,
                chain,
            )
            return
        if event == "exec":
            code = args[0]
            fname = str(getattr(code, "co_filename", "?"))
            origin = "in-memory code" if fname.startswith("<") else "file-backed code"
            owner = chain[0] if chain else "stdlib"
            self._add(
                "codegen", f"{owner}: exec/eval of {origin}{via_stdlib}", site, chain
            )

    # -- output -----------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            events = sorted(
                (dict(r) for r in self.records.values()),
                key=lambda r: (r["category"], r["detail"]),
            )
        return {
            "events": events,
            "blocked": sorted(set(self.blocked)),
            "elapsed_s": round(time.monotonic() - self.started, 3),
        }


def _argv0(event: str, args: tuple[Any, ...]) -> str:
    try:
        if event == "subprocess.Popen":
            executable, argv = args[0], args[1]
            if isinstance(argv, (list, tuple)) and argv:
                first = argv[0]
            elif isinstance(argv, (str, bytes)):
                first = os.fsdecode(argv).split()[0] if argv else ""
            else:
                first = executable
            return os.path.basename(os.fsdecode(first)) if first else "?"
        if event == "os.system":
            cmd = os.fsdecode(args[0])
            return os.path.basename(cmd.split()[0]) if cmd.split() else "?"
        if event in ("os.exec", "os.posix_spawn"):
            return os.path.basename(os.fsdecode(args[0]))
        if event == "os.spawn":
            return os.path.basename(os.fsdecode(args[1]))
        if event == "pty.spawn":
            argv = args[0]
            first = argv[0] if isinstance(argv, (list, tuple)) else argv
            return os.path.basename(os.fsdecode(first))
    except Exception:  # noqa: BLE001 - never let the recorder break the workload
        return "?"
    return event


def _url_origin(url: str) -> str:
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.hostname}:{parts.port or ''}".rstrip(":")


def _path_hint(path: Any) -> str:
    """A short, non-identifying hint of the written path (suffix only)."""
    try:
        name = os.path.basename(os.fsdecode(path))
    except TypeError:
        return "?"
    parts = name.split(".")[1:]
    while parts and (parts[-1].isdigit() or len(parts[-1]) > 12):
        parts.pop()  # atomic-write temp suffixes (pid, random token)
    return "*." + ".".join(parts) if parts else "(no suffix)"


def _child_main(workload: str, out_path: str, block_egress: bool) -> int:
    import importlib.util

    spec = importlib.util.find_spec("traigent")
    if spec is None or spec.origin is None:
        print("behaviour_audit: traigent is not importable", file=sys.stderr)
        return 2
    sdk_root = os.path.dirname(spec.origin)
    workload_file = os.path.join(_SCRIPTS_DIR, "audit_workloads.py")
    recorder = Recorder(
        block_egress=block_egress, sdk_root=sdk_root, workload_file=workload_file
    )
    sys.addaudithook(recorder.hook)

    status: dict[str, Any] = {"exit": "ok", "error": None}

    def dump() -> None:
        recorder.enabled = False  # the report file itself is not SDK behaviour
        payload = recorder.snapshot()
        payload.update(status)
        payload["traigent_file"] = spec.origin
        try:
            import litellm  # noqa: F401 - version only, already imported by now

            payload["litellm_version"] = _dist_version("litellm")
        except Exception:  # noqa: BLE001
            payload["litellm_version"] = None
        # Direct evidence of whether the SDK's litellm-download pin landed in
        # this child's environment -- lets a test assert on the *cause*
        # (pin unset) rather than only the *effect* (a host got recorded),
        # which would stay green even if the effect happened for an
        # unrelated reason.
        payload["env_litellm_local_model_cost_map"] = os.environ.get(
            "LITELLM_LOCAL_MODEL_COST_MAP"
        )
        tmp = out_path + ".partial"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        os.replace(tmp, out_path)

    atexit.register(dump)
    sys.path.insert(0, _SCRIPTS_DIR)
    import audit_workloads

    func: Callable[[], None] = audit_workloads.WORKLOADS[workload]
    try:
        func()
    except SystemExit as exc:
        if exc.code not in (0, None):
            status.update(exit="system-exit", error=str(exc.code))
    except BaseException as exc:  # noqa: BLE001 - report every failure shape
        status.update(exit="exception", error=f"{type(exc).__name__}: {exc}")
    return 0


def _dist_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Parent side: stub backend + driver
# ---------------------------------------------------------------------------


class _StubHandler(BaseHTTPRequestHandler):
    """Answers every request with 503 and logs method + path (never bodies)."""

    requests: list[str] = []

    def _answer(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        self.requests.append(f"{self.command} {self.path.split('?', 1)[0]}")
        body = b'{"error": "behaviour_audit stub backend"}'
        self.send_response(503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _answer

    def log_message(self, *_args: Any) -> None:  # silence stderr
        return


@contextmanager
def stub_backend() -> Iterator[tuple[str, list[str]]]:
    log: list[str] = []
    handler = type("Handler", (_StubHandler,), {"requests": log})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", log
    finally:
        server.shutdown()
        server.server_close()


def child_env(
    *,
    backend_url: str | None,
    pins: bool,
    extra: dict[str, str] | None,
) -> dict[str, str]:
    """Environment for the workload child.

    ``backend_url`` set: the SDK is pointed at the loopback stub, which the
    SDK's URL guard only admits in an explicit development environment
    (traigent/cloud/url_security.py validate_cloud_base_url), so the child
    runs with ENVIRONMENT=development. ``backend_url=None``: the SDK keeps its
    built-in production backend and production policy; every lookup of that
    host is recorded and (unless ``--allow-egress``) blocked.
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.upper().startswith(_SCRUBBED_ENV_PREFIXES)
        and k.upper() != "ENVIRONMENT"
    }
    env.update(
        {
            "TRAIGENT_API_KEY": DUMMY_API_KEY,
            "TRAIGENT_MOCK_LLM": "true",
            "OPENAI_API_KEY": "mock-key-for-behaviour-audit",  # pragma: allowlist secret
            "PYTHONUNBUFFERED": "1",
        }
    )
    if backend_url is not None:
        env.update(
            {
                "TRAIGENT_BACKEND_URL": backend_url,
                "TRAIGENT_API_URL": backend_url + "/api/v1",
                "ENVIRONMENT": "development",
                "TRAIGENT_ENV": "development",
                "TRAIGENT_ENVIRONMENT": "development",
            }
        )
    if pins:
        env.update(NETWORK_PINS)
    if extra:
        env.update(extra)
    return env


def run_workload(
    workload: str,
    *,
    pins: bool = False,
    block_egress: bool = True,
    backend: str = "stub",
    extra_env: dict[str, str] | None = None,
    timeout: float = 600.0,
    python: str | None = None,
) -> dict[str, Any]:
    """Run one workload in a fresh interpreter and return the report dict."""
    with tempfile.TemporaryDirectory(prefix="traigent-audit-") as workdir:
        out = os.path.join(workdir, "record.json")
        home = os.path.join(workdir, "home")
        os.makedirs(home)
        with stub_backend() as (url, stub_log):
            if backend not in ("stub", "default"):
                raise ValueError(
                    f"backend must be 'stub' or 'default', not {backend!r}"
                )
            env = child_env(
                backend_url=url if backend == "stub" else None,
                pins=pins,
                extra=extra_env,
            )
            env["HOME"] = home  # keep the real home out of reach and visible
            cmd = [python or sys.executable, _THIS_FILE, "_child", workload, out]
            if not block_egress:
                cmd.append("--allow-egress")
            started = time.monotonic()
            proc = subprocess.run(  # noqa: S603 - fixed argv, our own interpreter
                cmd,
                cwd=workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            wall = round(time.monotonic() - started, 2)
            stub_requests = sorted(set(stub_log))
        if not os.path.exists(out):
            raise RuntimeError(
                f"workload {workload!r} produced no record (rc={proc.returncode}); "
                f"stderr tail:\n{proc.stderr[-4000:]}"
            )
        with open(out, encoding="utf-8") as fh:
            report = json.load(fh)
    report.update(
        {
            "workload": workload,
            "pins": pins,
            "backend": backend,
            "block_egress": block_egress,
            "extra_env": extra_env or {},
            "returncode": proc.returncode,
            "wall_clock_s": wall,
            "stub_backend_requests": stub_requests,
            "stderr_tail": proc.stderr[-2000:],
            "python": sys.version.split()[0],
        }
    )
    report["summary"] = summarize(report)
    return report


def summarize(report: dict[str, Any]) -> dict[str, list[str]]:
    by_cat: dict[str, list[str]] = {}
    for ev in report["events"]:
        by_cat.setdefault(ev["category"], []).append(ev["detail"])
    hosts = sorted(
        {d.rsplit(":", 1)[0] for d in by_cat.get("dns", [])}
        | {d.rsplit(":", 1)[0] for d in by_cat.get("connect", [])}
    )
    return {
        "hosts": hosts,
        "external_hosts": [h for h in hosts if not Recorder._is_loopback(h)],
        "processes": by_cat.get("process", []),
        "file_writes": by_cat.get("file-write", []) + by_cat.get("file-exec-bit", []),
        "codegen": by_cat.get("codegen", []),
        "native": by_cat.get("native", []),
    }


def external_hosts(report: dict[str, Any]) -> set[str]:
    return set(report["summary"]["external_hosts"])


def to_markdown(reports: list[dict[str, Any]]) -> str:
    lines = []
    for rep in reports:
        lines.append(
            f"### Workload `{rep['workload']}`"
            f"{' + network pins' if rep['pins'] else ''}"
            f" (backend: {rep['backend']})\n"
        )
        lines.append(
            f"- exit: `{rep['exit']}` rc={rep['returncode']} "
            f"wall-clock {rep['wall_clock_s']} s; egress "
            f"{'blocked after recording' if rep['block_egress'] else 'allowed'}; "
            f"litellm {rep.get('litellm_version')}"
        )
        if rep.get("error"):
            lines.append(f"- workload error: `{rep['error']}`")
        lines.append(
            "- stub backend requests: "
            + (", ".join(f"`{r}`" for r in rep["stub_backend_requests"]) or "none")
        )
        lines.append("")
        lines.append("| category | detail | first-seen site | via | count |")
        lines.append("|---|---|---|---|---|")
        stdlib_codegen = [ev for ev in rep["events"] if "via stdlib" in ev["detail"]]
        for ev in rep["events"]:
            if ev in stdlib_codegen:
                continue
            lines.append(
                f"| {ev['category']} | `{ev['detail']}` | `{ev['first_site']}` | "
                f"{' ← '.join(ev['via'])} | {ev['count']} |"
            )
        if stdlib_codegen:
            gens = sorted(
                {ev["detail"].rsplit("via stdlib ", 1)[1] for ev in stdlib_codegen}
            )
            lines.append(
                f"| codegen | {len(stdlib_codegen)} rows / "
                f"{sum(ev['count'] for ev in stdlib_codegen)} events of stdlib "
                f"code generation ({', '.join(gens)}) triggered by class "
                "definitions at import; listed in the JSON report | | | |"
            )
        lines.append("")
    lines.append(
        "_Audit hooks do not observe child processes; every launch is listed "
        "under `process` so it can be audited separately._"
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list workloads")
    run = sub.add_parser("run", help="run workloads and write a report")
    run.add_argument("workloads", nargs="+")
    run.add_argument("--pins", action="store_true", help="set NETWORK_PINS")
    run.add_argument("--allow-egress", action="store_true")
    run.add_argument(
        "--backend",
        choices=("stub", "default"),
        default="stub",
        help="stub: loopback stub + development env; default: built-in backend",
    )
    run.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra env for the child (e.g. TRAIGENT_LITELLM_LIVE_PRICES=1)",
    )
    run.add_argument("--json", dest="json_out")
    run.add_argument("--markdown", dest="md_out")
    child = sub.add_parser("_child")
    child.add_argument("workload")
    child.add_argument("out")
    child.add_argument("--allow-egress", action="store_true")
    args = parser.parse_args(argv)

    if args.cmd == "_child":
        return _child_main(args.workload, args.out, not args.allow_egress)

    sys.path.insert(0, _SCRIPTS_DIR)
    import audit_workloads

    if args.cmd == "list":
        for name, fn in audit_workloads.WORKLOADS.items():
            print(f"{name}: {(fn.__doc__ or '').strip().splitlines()[0]}")
        return 0

    reports = [
        run_workload(
            w,
            pins=args.pins,
            block_egress=not args.allow_egress,
            backend=args.backend,
            extra_env=dict(kv.split("=", 1) for kv in args.env),
        )
        for w in args.workloads
    ]
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(reports, indent=2, default=str))
    md = to_markdown(reports)
    if args.md_out:
        Path(args.md_out).write_text(md)
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
