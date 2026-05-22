"""Integration tests for best-config runtime resolution on OptimizedFunction."""

from __future__ import annotations

import json

import pytest

import traigent
from traigent.api.decorators import optimize
from traigent.core.best_config_runtime import (
    BestConfigSource,
    CloudPublishUnavailable,
    CloudPublishUnavailableReason,
    function_ref_for,
    write_repo_best_config,
)
from traigent.core.optimized_function import OptimizedFunction


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


def test_publish_best_config_fails_closed_by_mode():
    opt_func = OptimizedFunction(
        func=plain_answer,
        config_space={"temperature": [0.1, 0.3]},
        best_config_source="repo",
    )

    with pytest.raises(CloudPublishUnavailable) as exc_info:
        opt_func.publish_best_config()

    assert exc_info.value.reason is CloudPublishUnavailableReason.DISABLED_BY_CONFIG


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
