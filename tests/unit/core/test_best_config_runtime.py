"""Tests for runtime best-config specs and snapshots."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import MappingProxyType

import pytest

from traigent.core.best_config_runtime import (
    BEST_CONFIG_MANIFEST_SCHEMA_VERSION,
    BEST_CONFIG_SCHEMA_VERSION,
    BestConfigSnapshot,
    BestConfigSource,
    CloudPublishUnavailable,
    CloudPublishUnavailableReason,
    canonical_json,
    compute_spec_hash,
    load_best_config_spec,
    resolve_cloud_cache_best_config,
    resolve_repo_best_config,
    sha256_digest,
    snapshot_from_spec,
    source_order_for_mode,
    thaw_config,
    write_repo_best_config,
)
from traigent.utils.exceptions import ConfigurationError


def spec_for(config_id: str = "answerer", **overrides):
    spec = {
        "schema_version": BEST_CONFIG_SCHEMA_VERSION,
        "config_id": config_id,
        "config": {"model": "gpt-4o-mini", "temperature": 0.2},
    }
    spec.update(overrides)
    return spec


def test_canonical_json_and_hash_are_stable():
    first = canonical_json({"b": 1, "a": ["x", True]})
    second = canonical_json({"a": ["x", True], "b": 1})

    assert first == second == '{"a":["x",true],"b":1}'
    assert sha256_digest(first).startswith("sha256:")


def test_canonical_json_rejects_non_json_native_values():
    with pytest.raises(ConfigurationError, match="non-JSON-native"):
        canonical_json({"bad": object()})

    circular = {}
    circular["self"] = circular
    with pytest.raises(ConfigurationError, match="circular"):
        canonical_json(circular)


def test_canonical_json_coerces_numpy_scalars_losslessly():
    """Optuna integer dimensions return numpy scalars (e.g. np.int64) into
    best_config; np.int64 does not subclass int, so the strict normalizer used
    to reject it. They must coerce losslessly via .item() at the JSON boundary.

    Regression for TraigentBackend#1147: best-config serialization crashed for
    any bayesian run tuning integer knobs (fewshot_k / candidate_count).
    """
    np = pytest.importorskip("numpy")

    # numpy integer / bool scalars coerce to JSON-native Python values.
    assert canonical_json(np.int64(3)) == "3"
    assert canonical_json(np.bool_(True)) == "true"

    # A full best_config carrying numpy scalars serializes clean.
    config = {
        "fewshot_k": np.int64(3),
        "candidate_count": np.int64(5),
        "use_voting": np.bool_(True),
    }
    serialized = canonical_json(config)
    # Round-trips to native Python values, JSON-parseable.
    assert json.loads(serialized) == {
        "fewshot_k": 3,
        "candidate_count": 5,
        "use_voting": True,
    }

    # numpy float / str scalars already passed (they subclass float / str) and
    # must keep working untouched.
    assert canonical_json(np.float64(0.5)) == "0.5"
    assert json.loads(canonical_json({"name": np.str_("gpt-4o")})) == {"name": "gpt-4o"}


def test_canonical_json_still_rejects_multi_element_numpy_arrays():
    """The .item() coercion must NOT swallow multi-element numpy arrays:
    array.item() raises on >1 element, so we fall through to the reject.

    Regression guard for TraigentBackend#1147.
    """
    np = pytest.importorskip("numpy")

    with pytest.raises(ConfigurationError, match="non-JSON-native"):
        canonical_json(np.array([1, 2]))

    with pytest.raises(ConfigurationError, match="non-JSON-native"):
        canonical_json({"bad": np.array([1, 2, 3])})


def test_snapshot_deep_freezes_config():
    snapshot = BestConfigSnapshot.from_config(
        {"nested": {"values": [1, 2]}},
        config_id="frozen",
        source="test",
    )

    assert isinstance(snapshot.config, MappingProxyType)
    assert thaw_config(snapshot.config) == {"nested": {"values": [1, 2]}}

    with pytest.raises(TypeError):
        snapshot.config["x"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        snapshot.config["nested"]["values"] = []  # type: ignore[index]


def test_unknown_top_level_spec_keys_reject():
    with pytest.raises(ConfigurationError, match="Unknown top-level"):
        snapshot_from_spec(spec_for(extra=True))


def test_unknown_config_keys_reject_when_config_space_known():
    spec = spec_for(config={"temperature": 0.2, "unexpected": 1})

    with pytest.raises(ConfigurationError, match="Unknown best config keys"):
        snapshot_from_spec(spec, configuration_space={"temperature": [0.1, 0.2]})


def test_forward_compat_ignores_non_sensitive_unknown_keys():
    spec = spec_for(
        config={"temperature": 0.2, "display_label": "fast"},
        forward_compat=True,
    )

    snapshot = snapshot_from_spec(spec, configuration_space={"temperature": [0.1, 0.2]})

    assert thaw_config(snapshot.config) == {"temperature": 0.2}


def test_forward_compat_does_not_ignore_safety_sensitive_unknown_keys():
    spec = spec_for(
        config={"temperature": 0.2, "model": "gpt-4o-mini"},
        forward_compat=True,
    )

    with pytest.raises(ConfigurationError, match="safety-sensitive"):
        snapshot_from_spec(spec, configuration_space={"temperature": [0.1, 0.2]})


def test_source_order_modes_never_touch_cloud_for_off_or_repo():
    assert source_order_for_mode("off") == ()
    assert source_order_for_mode("repo") == (BestConfigSource.REPO,)
    assert source_order_for_mode("cloud") == (
        BestConfigSource.CLOUD_CACHE,
        BestConfigSource.CLOUD_FETCH,
    )
    assert source_order_for_mode("repo_then_cloud") == (
        BestConfigSource.REPO,
        BestConfigSource.CLOUD_CACHE,
        BestConfigSource.CLOUD_FETCH,
    )
    assert source_order_for_mode("cloud_then_repo") == (
        BestConfigSource.CLOUD_CACHE,
        BestConfigSource.CLOUD_FETCH,
        BestConfigSource.REPO,
    )


def test_repo_manifest_round_trip(tmp_path, monkeypatch):
    repo_root = tmp_path
    config_dir = repo_root / ".traigent" / "best-configs"

    spec_path = write_repo_best_config(
        config_dir,
        config_id="answerer",
        config={"temperature": 0.2},
        function_ref="pkg.mod:answer",
        provenance={"source": "unit"},
    )

    assert spec_path.exists()
    manifest = json.loads((config_dir / "manifest.json").read_text())
    assert "spec_hash" in manifest["configs"]["answerer"]
    assert "config_hash" in manifest["configs"]["answerer"]

    monkeypatch.chdir(repo_root)
    snapshot = resolve_repo_best_config(
        config_id="answerer",
        repo_root=repo_root,
        configuration_space={"temperature": [0.1, 0.2]},
        expected_function_ref="pkg.mod:answer",
        strict=True,
    )

    assert snapshot is not None
    assert snapshot.source == BestConfigSource.REPO.value
    assert thaw_config(snapshot.config) == {"temperature": 0.2}


def test_repo_invalid_hash_warns_or_raises(tmp_path, monkeypatch):
    repo_root = tmp_path
    config_dir = repo_root / ".traigent" / "best-configs"
    write_repo_best_config(
        config_dir, config_id="answerer", config={"temperature": 0.2}
    )

    manifest_path = config_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["configs"]["answerer"]["spec_hash"] = "sha256:bad"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.chdir(repo_root)
    assert (
        resolve_repo_best_config(
            config_id="answerer", repo_root=repo_root, strict=False
        )
        is None
    )
    with pytest.raises(ConfigurationError, match="spec_hash mismatch"):
        resolve_repo_best_config(config_id="answerer", repo_root=repo_root, strict=True)


def test_load_best_config_spec_non_strict_ignores_file_errors(tmp_path):
    missing_path = tmp_path / "missing.json"

    assert load_best_config_spec(missing_path, strict=False) is None

    with pytest.raises(ConfigurationError, match="Failed to load best config"):
        load_best_config_spec(missing_path, strict=True)


def test_config_id_rejects_path_traversal(tmp_path):
    with pytest.raises(ConfigurationError, match="path separators"):
        write_repo_best_config(
            tmp_path / ".traigent" / "best-configs",
            config_id="../outside",
            config={"temperature": 0.2},
        )

    with pytest.raises(ConfigurationError, match="path separators"):
        snapshot_from_spec(spec_for(config_id="nested/answerer"))


def test_manifest_path_traversal_is_rejected(tmp_path):
    repo_root = tmp_path
    config_dir = repo_root / ".traigent" / "best-configs"
    config_dir.mkdir(parents=True)
    outside_path = repo_root / ".traigent" / "outside.json"
    outside_spec = spec_for("answerer", config={"temperature": 0.2})
    outside_path.write_text(json.dumps(outside_spec), encoding="utf-8")
    (config_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": BEST_CONFIG_MANIFEST_SCHEMA_VERSION,
                "configs": {
                    "answerer": {
                        "path": "../outside.json",
                        "spec_hash": compute_spec_hash(outside_spec),
                        "config_hash": sha256_digest(
                            canonical_json(outside_spec["config"])
                        ),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    assert resolve_repo_best_config(config_id="answerer", repo_root=repo_root) is None
    with pytest.raises(ConfigurationError, match="inside its directory"):
        resolve_repo_best_config(
            config_id="answerer",
            repo_root=repo_root,
            strict=True,
        )


def test_cloud_cache_ttl_behavior(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    spec = spec_for("answerer", config={"temperature": 0.2})
    spec_hash = compute_spec_hash(spec)
    config_hash = sha256_digest(canonical_json(spec["config"]))
    (cache_dir / "answerer.json").write_text(json.dumps(spec), encoding="utf-8")
    (cache_dir / "answerer.metadata.json").write_text(
        json.dumps(
            {
                "spec_hash": spec_hash,
                "config_hash": config_hash,
                "etag": "v1",
                "loaded_at": "2026-05-22T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    snapshot = resolve_cloud_cache_best_config(
        config_id="answerer",
        cache_dir=cache_dir,
        ttl_seconds=60,
        now=datetime(2026, 5, 22, 0, 0, 30, tzinfo=UTC),
        strict=True,
    )
    assert snapshot is not None
    assert snapshot.source == BestConfigSource.CLOUD_CACHE.value

    stale = resolve_cloud_cache_best_config(
        config_id="answerer",
        cache_dir=cache_dir,
        ttl_seconds=60,
        now=datetime(2026, 5, 22, 0, 2, 0, tzinfo=UTC),
        strict=True,
    )
    assert stale is None


def test_cloud_cache_stale_ok_ttl_allows_offline_reuse(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    spec = spec_for("answerer", config={"temperature": 0.2})
    (cache_dir / "answerer.json").write_text(json.dumps(spec), encoding="utf-8")
    (cache_dir / "answerer.metadata.json").write_text(
        json.dumps(
            {
                "spec_hash": compute_spec_hash(spec),
                "config_hash": sha256_digest(canonical_json(spec["config"])),
                "version": "v1",
                "loaded_at": "2026-05-22T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    snapshot = resolve_cloud_cache_best_config(
        config_id="answerer",
        cache_dir=cache_dir,
        ttl_seconds=60,
        stale_ok_ttl_seconds=int(timedelta(hours=1).total_seconds()),
        now=datetime(2026, 5, 22, 0, 30, 0, tzinfo=UTC),
        strict=True,
    )

    assert snapshot is not None


def test_cloud_publish_unavailable_exposes_reason():
    exc = CloudPublishUnavailable(
        CloudPublishUnavailableReason.BACKEND_NOT_IMPLEMENTED,
        "Cloud publish not enabled",
    )

    assert exc.reason is CloudPublishUnavailableReason.BACKEND_NOT_IMPLEMENTED
    assert exc.details["reason"] == "backend_not_implemented"


def test_forward_compat_manifest_hashes_bind_raw_spec_before_filtering():
    spec = spec_for(
        config={"temperature": 0.2, "display_label": "fast"},
        forward_compat=True,
    )
    manifest_entry = {
        "spec_hash": compute_spec_hash(spec),
        "config_hash": sha256_digest(canonical_json(spec["config"])),
    }

    snapshot = snapshot_from_spec(
        spec,
        configuration_space={"temperature": [0.1, 0.2]},
        manifest_entry=manifest_entry,
    )

    assert thaw_config(snapshot.config) == {"temperature": 0.2}
    assert snapshot.spec_hash == manifest_entry["spec_hash"]
    assert snapshot.config_hash == sha256_digest(canonical_json({"temperature": 0.2}))
