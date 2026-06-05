"""Integration tests for best-config runtime resolution on OptimizedFunction."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

import traigent
from traigent.api.decorators import optimize
from traigent.api.types import (
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)
from traigent.core.best_config_runtime import (
    BestConfigSnapshot,
    BestConfigSource,
    CloudPublishUnavailable,
    CloudPublishUnavailableReason,
    canonical_json,
    compute_spec_hash,
    function_ref_for,
    sha256_digest,
    write_repo_best_config,
)
from traigent.core.optimized_function import OptimizedFunction
from traigent.utils.exceptions import ConfigurationError

FAKE_TRAIGENT_API_KEY = "tg_" + "x" * 61  # pragma: allowlist secret


def plain_answer(text: str, **kwargs):
    return f"{text}:{kwargs.get('temperature', 'missing')}"


def context_answer(text: str):
    config = traigent.get_config()
    return f"{text}:{config.get('temperature')}"


def test_load_from_beats_env_path(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit.json"
    env = tmp_path / "env.json"
    explicit.write_text(json.dumps({"config": {"temperature": 0.2}}), encoding="utf-8")
    env.write_text(json.dumps({"config": {"temperature": 0.9}}), encoding="utf-8")
    monkeypatch.setenv("TRAIGENT_CONFIG_PATH", str(env))

    opt_func = OptimizedFunction(
        func=plain_answer,
        config_space={"temperature": [0.2, 0.9]},
        load_from=str(explicit),
        default_config={"temperature": 0.1},
    )

    assert opt_func.current_config["temperature"] == 0.2
    assert opt_func.best_config_snapshot.source == BestConfigSource.LOAD_FROM.value


def test_load_from_canonical_config_id_mismatch_does_not_legacy_fallback(tmp_path):
    spec_path = write_repo_best_config(
        tmp_path,
        config_id="other-answerer",
        config={"temperature": 0.9},
        function_ref=function_ref_for(plain_answer),
    )

    opt_func = OptimizedFunction(
        func=plain_answer,
        config_space={"temperature": [0.1, 0.9]},
        config_id="expected-answerer",
        load_from=str(spec_path),
        default_config={"temperature": 0.1},
    )

    assert opt_func.current_config["temperature"] == 0.1
    assert opt_func.best_config_snapshot.source == BestConfigSource.DEFAULT.value


def test_repo_best_config_source_loads_at_init(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_repo_best_config(
        tmp_path / ".traigent" / "best-configs",
        config_id="answerer",
        config={"temperature": 0.3},
        function_ref=function_ref_for(context_answer),
    )

    opt_func = OptimizedFunction(
        func=context_answer,
        config_space={"temperature": [0.1, 0.3]},
        config_id="answerer",
        best_config_source="repo",
        default_config={"temperature": 0.1},
    )

    assert opt_func.current_config["temperature"] == 0.3
    assert opt_func("hello") == "hello:0.3"
    assert opt_func.best_config_snapshot.source == BestConfigSource.REPO.value


def test_set_config_sticky_until_clear_override(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_repo_best_config(
        tmp_path / ".traigent" / "best-configs",
        config_id="answerer",
        config={"temperature": 0.3},
        function_ref=function_ref_for(context_answer),
    )
    opt_func = OptimizedFunction(
        func=context_answer,
        config_space={"temperature": [0.1, 0.3, 0.8]},
        config_id="answerer",
        best_config_source="repo",
        default_config={"temperature": 0.1},
    )

    opt_func.set_config({"temperature": 0.8})

    assert opt_func.current_config["temperature"] == 0.8
    assert opt_func.best_config_snapshot.source == BestConfigSource.OVERRIDE.value

    assert opt_func.clear_override() is True
    assert opt_func.current_config["temperature"] == 0.3
    assert opt_func.clear_override() is False


def test_auto_load_does_not_overwrite_override_set_during_resolution(monkeypatch):
    opt_func = OptimizedFunction(
        func=context_answer,
        config_space={"temperature": [0.1, 0.3, 0.8]},
        config_id="answerer",
        default_config={"temperature": 0.1},
    )
    resolved_snapshot = BestConfigSnapshot.from_config(
        {"temperature": 0.3},
        config_id="answerer",
        source=BestConfigSource.REPO.value,
    )

    def resolve_after_override():
        opt_func.set_config({"temperature": 0.8})
        return resolved_snapshot

    monkeypatch.setattr(opt_func._csm, "_resolve_mode_snapshot", resolve_after_override)

    opt_func._csm.maybe_auto_load_config()

    assert opt_func.current_config["temperature"] == 0.8
    assert opt_func.best_config_snapshot.source == BestConfigSource.OVERRIDE.value


def test_off_source_does_not_read_repo_or_cloud(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_repo_best_config(
        tmp_path / ".traigent" / "best-configs",
        config_id="answerer",
        config={"temperature": 0.3},
        function_ref=function_ref_for(plain_answer),
    )

    opt_func = OptimizedFunction(
        func=plain_answer,
        config_space={"temperature": [0.1, 0.3]},
        config_id="answerer",
        best_config_source="off",
        best_config_cache_dir=str(tmp_path / "cache"),
        default_config={"temperature": 0.1},
    )

    assert opt_func.current_config["temperature"] == 0.1
    assert opt_func.best_config_snapshot.source == BestConfigSource.DEFAULT.value


def test_cloud_fetch_failure_fails_closed_for_cloud_mode(tmp_path, monkeypatch):
    class FailingBestConfigClient:
        def fetch_best_config_sync(self, *args, **kwargs):  # noqa: ARG002
            raise RuntimeError("backend unavailable")

    import traigent.cloud.backend_client as backend_client

    monkeypatch.setattr(
        backend_client,
        "get_backend_client",
        lambda **kwargs: FailingBestConfigClient(),
    )

    with pytest.raises(CloudPublishUnavailable) as exc_info:
        OptimizedFunction(
            func=plain_answer,
            config_space={"temperature": [0.1, 0.3]},
            config_id="answerer",
            best_config_source="cloud",
            best_config_cache_dir=str(tmp_path / "missing-cache"),
            default_config={"temperature": 0.1},
        )

    assert exc_info.value.reason is CloudPublishUnavailableReason.REQUEST_FAILED


def test_invocation_does_not_re_resolve_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_repo_best_config(
        tmp_path / ".traigent" / "best-configs",
        config_id="answerer",
        config={"temperature": 0.3},
        function_ref=function_ref_for(context_answer),
    )
    opt_func = OptimizedFunction(
        func=context_answer,
        config_space={"temperature": [0.1, 0.3]},
        config_id="answerer",
        best_config_source="repo",
        default_config={"temperature": 0.1},
    )

    # If invocation re-read disk, deleting the repo spec would change or break it.
    (tmp_path / ".traigent" / "best-configs" / "answerer.json").unlink()

    assert opt_func("hello") == "hello:0.3"


def test_export_best_config_writes_repo_spec(tmp_path):
    opt_func = OptimizedFunction(
        func=plain_answer,
        config_space={"temperature": [0.1, 0.3]},
        config_id="answerer",
    )
    opt_func.set_config({"temperature": 0.3})

    spec_path = opt_func.export_best_config(tmp_path / ".traigent" / "best-configs")

    assert spec_path.name == "answerer.json"
    assert (spec_path.parent / "manifest.json").exists()


def test_fresh_process_cloud_publish_then_fetch_uses_backend_network_contract(tmp_path):
    seen_requests: list[dict[str, object]] = []
    state: dict[str, dict[str, object]] = {}

    class ContractBackend(BaseHTTPRequestHandler):
        def _send_json(self, status_code: int, payload: dict[str, object]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            seen_requests.append(
                {
                    "method": "GET",
                    "path": parsed.path,
                    "query": parse_qs(parsed.query),
                    "api_key": self.headers.get("X-API-Key"),
                }
            )
            if (
                parsed.path != "/api/v1/best-configs/fresh-promote"
                or "spec" not in state
            ):
                self.send_response(404)
                self.end_headers()
                return

            spec = state["spec"]
            self._send_json(
                200,
                {
                    "success": True,
                    "data": {
                        "config_id": "fresh-promote",
                        "environment": "staging",
                        "version": 1,
                        "etag": 'W/"1-freshpublish"',
                        "spec_hash": compute_spec_hash(spec),
                        "config_hash": sha256_digest(canonical_json(spec["config"])),
                        "spec": spec,
                    },
                },
            )

        def do_POST(self):  # noqa: N802
            parsed = urlparse(self.path)
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            body = json.loads(raw_body.decode("utf-8"))
            if parsed.path == "/api/v1/keys/validate":
                self._send_json(200, {"valid": True})
                return
            spec = body["spec"]
            state["spec"] = spec
            seen_requests.append(
                {
                    "method": "POST",
                    "path": parsed.path,
                    "api_key": self.headers.get("X-API-Key"),
                    "if_match": self.headers.get("If-Match"),
                    "environment": body.get("environment"),
                    "spec": spec,
                }
            )
            self._send_json(
                201,
                {
                    "success": True,
                    "data": {
                        "config_id": spec["config_id"],
                        "environment": spec["environment"],
                        "version": 1,
                        "etag": 'W/"1-freshpublish"',
                        "spec_hash": compute_spec_hash(spec),
                        "config_hash": sha256_digest(canonical_json(spec["config"])),
                        "spec": spec,
                    },
                },
            )

        def log_message(self, format, *args):  # noqa: A002
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), ContractBackend)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        backend_url = f"http://127.0.0.1:{server.server_port}"
        api_key = FAKE_TRAIGENT_API_KEY
        publish_script = textwrap.dedent("""
            import json

            from traigent.cloud.auth import AuthManager
            from traigent.core.optimized_function import OptimizedFunction

            async def _trust_contract_backend(self, api_key):
                return None

            AuthManager._validate_api_key_with_backend = _trust_contract_backend

            def fresh_answer(text: str, **kwargs):
                return f"{text}:{kwargs.get('temperature')}"

            wrapped = OptimizedFunction(
                func=fresh_answer,
                config_space={"temperature": [0.1, 0.77]},
                config_id="fresh-promote",
                best_config_environment="staging",
                default_config={"temperature": 0.1},
            )
            wrapped.set_config({"temperature": 0.77})
            result = wrapped.publish_best_config()
            print(json.dumps({
                "config_id": result["config_id"],
                "version": result["version"],
            }, sort_keys=True))
            """)
        fetch_script = textwrap.dedent("""
            import json
            import os

            from traigent.cloud.auth import AuthManager
            from traigent.core.optimized_function import OptimizedFunction

            async def _trust_contract_backend(self, api_key):
                return None

            AuthManager._validate_api_key_with_backend = _trust_contract_backend

            def fresh_answer(text: str, **kwargs):
                return f"{text}:{kwargs.get('temperature')}"

            wrapped = OptimizedFunction(
                func=fresh_answer,
                config_space={"temperature": [0.1, 0.77]},
                config_id="fresh-promote",
                best_config_source="cloud",
                best_config_cache_dir=os.environ["TRAIGENT_BEST_CONFIG_CACHE"],
                best_config_environment="staging",
                default_config={"temperature": 0.1},
            )
            print(json.dumps({
                "temperature": wrapped.current_config["temperature"],
                "source": wrapped.best_config_snapshot.source,
            }, sort_keys=True))
            """)
        env = os.environ.copy()
        env.update(
            {
                "TRAIGENT_API_KEY": api_key,
                "TRAIGENT_BACKEND_URL": backend_url,
                "TRAIGENT_API_URL": f"{backend_url}/api/v1",
                "TRAIGENT_BEST_CONFIG_CACHE": str(tmp_path / "fetch-cache"),
            }
        )
        repo_root = Path(__file__).resolve().parents[3]
        publish_result = subprocess.run(
            [sys.executable, "-c", publish_script],
            cwd=repo_root,
            env=env,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
        fetch_result = subprocess.run(
            [sys.executable, "-c", fetch_script],
            cwd=repo_root,
            env=env,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert publish_result.returncode == 0, publish_result.stderr
    assert json.loads(publish_result.stdout.strip().splitlines()[-1]) == {
        "config_id": "fresh-promote",
        "version": 1,
    }
    assert fetch_result.returncode == 0, fetch_result.stderr
    assert json.loads(fetch_result.stdout.strip().splitlines()[-1]) == {
        "source": BestConfigSource.CLOUD_FETCH.value,
        "temperature": 0.77,
    }
    assert [request["method"] for request in seen_requests] == ["GET", "POST", "GET"]
    assert seen_requests[1]["path"] == "/api/v1/best-configs"
    assert seen_requests[1]["if_match"] is None
    assert seen_requests[1]["environment"] == "staging"
    assert seen_requests[1]["spec"]["config"] == {"temperature": 0.77}


def test_publish_best_config_fails_without_active_best_config():
    opt_func = OptimizedFunction(
        func=plain_answer,
        config_space={"temperature": [0.1, 0.3]},
        best_config_source="repo",
    )

    with pytest.raises(ConfigurationError) as exc_info:
        opt_func.publish_best_config()

    assert "No best configuration available" in str(exc_info.value)


def test_decorator_passes_best_config_runtime_options(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def decorated_target(text: str):
        config = traigent.get_config()
        return f"{text}:{config.get('temperature')}"

    write_repo_best_config(
        tmp_path / ".traigent" / "best-configs",
        config_id="decorated",
        config={"temperature": 0.4},
        function_ref=function_ref_for(decorated_target),
    )

    wrapped = optimize(
        configuration_space={"temperature": [0.1, 0.4]},
        config_id="decorated",
        best_config_source="repo",
        default_config={"temperature": 0.1},
    )(decorated_target)

    assert wrapped.current_config["temperature"] == 0.4
    assert wrapped("ok") == "ok:0.4"


def test_decorator_passes_cloud_cache_ttl_options(tmp_path):
    cache_dir = tmp_path / "cache"
    write_repo_best_config(
        cache_dir,
        config_id="decorated-cache",
        config={"temperature": 0.4},
        function_ref=function_ref_for(context_answer),
    )
    metadata_path = cache_dir / "manifest.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["configs"]["decorated-cache"]["etag"] = "v1"
    metadata["configs"]["decorated-cache"]["loaded_at"] = "2026-05-22T00:00:00+00:00"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    wrapped = optimize(
        configuration_space={"temperature": [0.1, 0.4]},
        config_id="decorated-cache",
        best_config_source="cloud",
        best_config_cache_dir=str(cache_dir),
        best_config_cache_ttl_seconds=1,
        best_config_stale_ok_ttl_seconds=60 * 60 * 24 * 365,
        default_config={"temperature": 0.1},
    )(context_answer)

    assert wrapped.current_config["temperature"] == 0.4
    assert wrapped.best_config_snapshot.source == BestConfigSource.CLOUD_CACHE.value


def streaming_context_answer(text: str):
    config = traigent.get_config()
    yield f"{text}:{config.get("temperature")}"


async def async_context_answer(text: str):
    config = traigent.get_config()
    return f"{text}:{config.get("temperature")}"


async def async_streaming_context_answer(text: str):
    config = traigent.get_config()
    yield f"{text}:{config.get("temperature")}"


def test_repo_best_config_source_survives_stream_iteration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_repo_best_config(
        tmp_path / ".traigent" / "best-configs",
        config_id="streaming-answerer",
        config={"temperature": 0.6},
        function_ref=function_ref_for(streaming_context_answer),
    )
    opt_func = OptimizedFunction(
        func=streaming_context_answer,
        config_space={"temperature": [0.1, 0.6]},
        config_id="streaming-answerer",
        best_config_source="repo",
        default_config={"temperature": 0.1},
    )

    assert list(opt_func("hello")) == ["hello:0.6"]


@pytest.mark.asyncio
async def test_repo_best_config_source_supports_async_invocation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_repo_best_config(
        tmp_path / ".traigent" / "best-configs",
        config_id="async-answerer",
        config={"temperature": 0.7},
        function_ref=function_ref_for(async_context_answer),
    )
    opt_func = OptimizedFunction(
        func=async_context_answer,
        config_space={"temperature": [0.1, 0.7]},
        config_id="async-answerer",
        best_config_source="repo",
        default_config={"temperature": 0.1},
    )

    assert await opt_func("hello") == "hello:0.7"


@pytest.mark.asyncio
async def test_repo_best_config_source_supports_async_streaming(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_repo_best_config(
        tmp_path / ".traigent" / "best-configs",
        config_id="async-streaming-answerer",
        config={"temperature": 0.8},
        function_ref=function_ref_for(async_streaming_context_answer),
    )
    opt_func = OptimizedFunction(
        func=async_streaming_context_answer,
        config_space={"temperature": [0.1, 0.8]},
        config_id="async-streaming-answerer",
        best_config_source="repo",
        default_config={"temperature": 0.1},
    )

    values = []
    async for value in opt_func("hello"):
        values.append(value)

    assert values == ["hello:0.8"]


@pytest.mark.asyncio
async def test_optimization_completion_refreshes_best_config_snapshot():
    opt_func = OptimizedFunction(
        func=context_answer,
        config_space={"temperature": [0.1, 0.9]},
        default_config={"temperature": 0.1},
    )

    class RuntimeConfig:
        @staticmethod
        def is_edge_analytics_mode():
            return False

    opt_func.traigent_config = RuntimeConfig()
    result = OptimizationResult(
        trials=[
            TrialResult(
                trial_id="trial-1",
                config={"temperature": 0.9},
                metrics={"accuracy": 1.0},
                status=TrialStatus.COMPLETED,
                duration=0.1,
                timestamp=None,
                metadata={},
            )
        ],
        best_config={"temperature": 0.9},
        best_score=1.0,
        optimization_id="opt-1",
        duration=0.1,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="unit",
        timestamp=None,
        metadata={},
    )

    class FakeOrchestrator:
        async def optimize(self, *, func, dataset):
            return result

    await opt_func._run_and_finalize_optimization(
        FakeOrchestrator(),
        dataset=object(),
        effective_config_space={"temperature": [0.1, 0.9]},
        save_to=None,
    )

    assert opt_func("hello") == "hello:0.9"
    assert (
        opt_func.best_config_snapshot.source == BestConfigSource.APPLY_BEST_CONFIG.value
    )
    assert dict(opt_func.best_config_snapshot.config) == {"temperature": 0.9}
