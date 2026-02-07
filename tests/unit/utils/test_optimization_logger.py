"""Comprehensive tests for optimization_logger.py.

Covers sanitization functions, OptimizationLogger class lifecycle,
file I/O, trial buffering, checkpoints, metrics, results, and
experiment indexing.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from traigent.api.types import (
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)
from traigent.utils.optimization_logger import (
    OptimizationLogger,
    _is_sensitive_key,
    _looks_like_secret,
    _mask_string,
    _normalize_key_name,
    sanitize_for_logging,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trial(
    trial_id: str = "t1",
    status: str = "completed",
    metrics: dict[str, float] | None = None,
    metadata: dict[str, Any] | None = None,
) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        config={"model": "gpt-4o"},
        metrics=metrics or {"accuracy": 0.9},
        status=TrialStatus(status),
        duration=1.5,
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        metadata=metadata or {},
    )


def _make_opt_result(
    trials: list[TrialResult] | None = None,
    objectives: list[str] | None = None,
    error: str | None = None,
) -> OptimizationResult:
    return OptimizationResult(
        trials=trials or [_make_trial()],
        best_config={"model": "gpt-4o"},
        best_score=0.95,
        optimization_id="opt-123",
        duration=10.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=objectives or ["accuracy"],
        algorithm="bayesian",
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
    )


def _make_logger(tmp_path: Path, **kwargs: Any) -> OptimizationLogger:
    defaults: dict[str, Any] = {
        "experiment_name": "test_exp",
        "session_id": "sess12345678",
        "execution_mode": "edge_analytics",
        "base_path": tmp_path,
    }
    defaults.update(kwargs)
    return OptimizationLogger(**defaults)


# ---------------------------------------------------------------------------
# _mask_string
# ---------------------------------------------------------------------------


class TestMaskString:
    def test_empty(self) -> None:
        assert _mask_string("") == "***"

    def test_short_le4(self) -> None:
        assert _mask_string("abcd") == "***"
        assert _mask_string("ab") == "***"

    def test_medium_5_to_8(self) -> None:
        assert _mask_string("abcde") == "ab..."
        assert _mask_string("abcdefgh") == "ab..."

    def test_long_gt8(self) -> None:
        result = _mask_string("abcdefghi")
        assert result == "abcd...fghi"

    def test_boundary_9_chars(self) -> None:
        assert _mask_string("123456789") == "1234...6789"


# ---------------------------------------------------------------------------
# _normalize_key_name
# ---------------------------------------------------------------------------


class TestNormalizeKeyName:
    def test_lowercase_and_hyphens(self) -> None:
        assert _normalize_key_name("API-Key") == "api_key"

    def test_multiple_hyphens(self) -> None:
        assert _normalize_key_name("a-b-c") == "a_b_c"

    def test_already_normalized(self) -> None:
        assert _normalize_key_name("api_key") == "api_key"


# ---------------------------------------------------------------------------
# _is_sensitive_key
# ---------------------------------------------------------------------------


class TestIsSensitiveKey:
    @pytest.mark.parametrize(
        "key",
        [
            "api_key",
            "password",
            "token",
            "secret",
            "jwt",
            "authorization",
            "bearer_token",
        ],
    )
    def test_direct_keywords(self, key: str) -> None:
        assert _is_sensitive_key(key) is True

    @pytest.mark.parametrize(
        "key", ["my_token", "custom_secret", "db_password", "ssh_key"]
    )
    def test_suffix_patterns(self, key: str) -> None:
        assert _is_sensitive_key(key) is True

    @pytest.mark.parametrize("key", ["token_value", "secret_name", "jwt_payload"])
    def test_prefix_patterns(self, key: str) -> None:
        assert _is_sensitive_key(key) is True

    def test_hyphenated(self) -> None:
        assert _is_sensitive_key("Api-Key") is True
        assert _is_sensitive_key("Auth-Token") is True

    @pytest.mark.parametrize("key", ["model", "temperature", "max_tokens", "name"])
    def test_non_sensitive(self, key: str) -> None:
        assert _is_sensitive_key(key) is False


# ---------------------------------------------------------------------------
# _looks_like_secret
# ---------------------------------------------------------------------------


class TestLooksLikeSecret:
    @pytest.mark.parametrize(
        "value",
        [
            "sk-abc123def456ghi789",  # pragma: allowlist secret
            "bearer eyJtoken",
            "token-abc123def456",
        ],
    )
    def test_prefix_match(self, value: str) -> None:
        assert _looks_like_secret(value) is True

    def test_jwt_pattern(self) -> None:
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIx.SflKxwRJ"  # pragma: allowlist secret
        assert _looks_like_secret(jwt) is True

    def test_pem_marker(self) -> None:
        # Construct PEM header to avoid detect-private-key hook
        pem = "-----BEGIN " + "PRIVATE KEY" + "-----"  # pragma: allowlist secret
        assert _looks_like_secret(pem) is True

    def test_long_token_like(self) -> None:
        # >=32 chars, no spaces, has digits
        token = "a" * 30 + "12"  # 32 chars total
        assert _looks_like_secret(token) is True

    def test_normal_text(self) -> None:
        assert _looks_like_secret("Hello world") is False

    def test_short_no_prefix(self) -> None:
        assert _looks_like_secret("abc123") is False

    def test_long_with_spaces(self) -> None:
        # Long but has spaces — not token-like
        assert (
            _looks_like_secret("this is a long string with spaces and digit 1") is False
        )


# ---------------------------------------------------------------------------
# sanitize_for_logging
# ---------------------------------------------------------------------------


class TestSanitizeForLogging:
    def test_dict_sensitive_key(self) -> None:
        data = {"api_key": "sk-secret123456789"}  # pragma: allowlist secret
        result = sanitize_for_logging(data)
        assert result["api_key"] != "sk-secret123456789"  # pragma: allowlist secret
        assert "***" in result["api_key"] or "..." in result["api_key"]

    def test_dict_non_sensitive_key(self) -> None:
        data = {"model": "gpt-4o", "temperature": 0.7}
        result = sanitize_for_logging(data)
        assert result["model"] == "gpt-4o"
        assert result["temperature"] == 0.7

    def test_nested_dict(self) -> None:
        secret_val = "sk-secret123456789"  # pragma: allowlist secret
        data = {"config": {"api_key": secret_val}}
        result = sanitize_for_logging(data)
        assert (
            "***" in result["config"]["api_key"]  # pragma: allowlist secret
            or "..." in result["config"]["api_key"]
        )

    def test_list(self) -> None:
        data = [{"api_key": "secret_val"}]  # pragma: allowlist secret
        result = sanitize_for_logging(data)
        assert isinstance(result, list)
        assert result[0]["api_key"] != "secret_val"  # pragma: allowlist secret

    def test_tuple(self) -> None:
        data = ("hello", "world")
        result = sanitize_for_logging(data)
        assert isinstance(result, list)
        assert result == ["hello", "world"]

    def test_set(self) -> None:
        data = {"hello"}
        result = sanitize_for_logging(data)
        assert isinstance(result, list)

    def test_string_secret(self) -> None:
        result = sanitize_for_logging(
            "sk-abc123def456ghi789"
        )  # pragma: allowlist secret
        assert "***" in result or "..." in result

    def test_string_normal(self) -> None:
        assert sanitize_for_logging("hello") == "hello"

    def test_primitives(self) -> None:
        assert sanitize_for_logging(42) == 42
        assert sanitize_for_logging(3.14) == 3.14
        assert sanitize_for_logging(True) is True
        assert sanitize_for_logging(None) is None

    def test_unsupported_type(self) -> None:
        class Custom:
            def __str__(self) -> str:
                return "custom_obj"

        result = sanitize_for_logging(Custom())
        assert result == "custom_obj"

    def test_circular_reference_via_memo(self) -> None:
        d: dict[str, Any] = {"key": "value"}
        d["self"] = d
        result = sanitize_for_logging(d)
        # Memo cache returns the same sanitized dict (not "<recursion>")
        # because the dict is added to _memo before recursing children
        assert result["self"] is result

    def test_memo_cache(self) -> None:
        shared = {"inner": "value"}
        data = {"a": shared, "b": shared}
        result = sanitize_for_logging(data)
        # Both should reference same sanitized dict
        assert result["a"] is result["b"]

    def test_recursion_sentinel_on_list(self) -> None:
        """List self-reference triggers recursion sentinel (memo set after iteration)."""
        lst: list[Any] = [1, 2]
        lst.append(lst)
        result = sanitize_for_logging(lst)
        assert result[2] == "<recursion>"


# ---------------------------------------------------------------------------
# OptimizationLogger __init__
# ---------------------------------------------------------------------------


class TestOptimizationLoggerInit:
    def test_directory_structure(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path)
        for sub in ["meta", "trials", "metrics", "checkpoints", "artifacts", "logs"]:
            assert (log.run_path / sub).is_dir()

    def test_experiment_name_sanitized(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path, experiment_name="test/exp:name")
        assert "/" not in log.experiment_name
        assert ":" not in log.experiment_name

    def test_run_id_format(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path, session_id="abcdefghijklmnop")
        assert log.run_id.endswith("_abcdefgh")

    def test_short_session_id(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path, session_id="abc")
        assert log.run_id.endswith("_abc")

    def test_execution_mode_resolved(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path)
        assert log.execution_mode == "edge_analytics"

    def test_buffer_size(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path, buffer_size=5)
        assert log.buffer_size == 5


# ---------------------------------------------------------------------------
# _resolve_default_base_path
# ---------------------------------------------------------------------------


class TestResolveDefaultBasePath:
    def test_env_override(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("TRAIGENT_OPTIMIZATION_LOG_DIR", str(tmp_path / "custom"))
        result = OptimizationLogger._resolve_default_base_path()
        assert "custom" in str(result)

    def test_results_folder_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("TRAIGENT_OPTIMIZATION_LOG_DIR", raising=False)
        monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))
        result = OptimizationLogger._resolve_default_base_path()
        assert "results" in str(result)

    def test_no_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TRAIGENT_OPTIMIZATION_LOG_DIR", raising=False)
        monkeypatch.delenv("TRAIGENT_RESULTS_FOLDER", raising=False)
        result = OptimizationLogger._resolve_default_base_path()
        assert "optimization_logs" in str(result)


# ---------------------------------------------------------------------------
# _sanitize_name
# ---------------------------------------------------------------------------


class TestSanitizeName:
    def test_replaces_invalid_chars(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path)
        assert log._sanitize_name('a/b\\c:d*e?f"g<h>i|j') == "a_b_c_d_e_f_g_h_i_j"

    def test_truncates_long_names(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path)
        assert len(log._sanitize_name("x" * 200)) == 100

    def test_normal_name_unchanged(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path)
        assert log._sanitize_name("my_experiment") == "my_experiment"


# ---------------------------------------------------------------------------
# _atomic_write
# ---------------------------------------------------------------------------


class TestAtomicWrite:
    def test_writes_json_correctly(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path)
        target = log.run_path / "test.json"
        log._atomic_write(target, {"key": "value"})
        data = json.loads(target.read_text())
        assert data["key"] == "value"

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path)
        target = log.run_path / "sub" / "dir" / "test.json"
        log._atomic_write(target, {"nested": True})
        assert target.exists()

    def test_sanitizes_data(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path)
        target = log.run_path / "sanitized.json"
        log._atomic_write(target, {"api_key": "sk-secret123456789"})  # pragma: allowlist secret
        data = json.loads(target.read_text())
        assert data["api_key"] != "sk-secret123456789"  # pragma: allowlist secret

    def test_temp_cleaned_on_success(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path)
        target = log.run_path / "clean.json"
        log._atomic_write(target, {"ok": True})
        # No leftover .tmp files
        tmp_files = list(log.run_path.glob("*.tmp.*"))
        assert len(tmp_files) == 0


# ---------------------------------------------------------------------------
# _append_jsonl
# ---------------------------------------------------------------------------


class TestAppendJsonl:
    def test_appends_single_line(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path)
        target = log.run_path / "stream.jsonl"
        log._append_jsonl(target, {"event": "start"})
        lines = target.read_text().strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0])["event"] == "start"

    def test_appends_multiple_lines(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path)
        target = log.run_path / "stream.jsonl"
        log._append_jsonl(target, {"event": "a"})
        log._append_jsonl(target, {"event": "b"})
        lines = target.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_sanitizes_data(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path)
        target = log.run_path / "sanitized.jsonl"
        log._append_jsonl(target, {"password": "secret123"})  # pragma: allowlist secret
        line = json.loads(target.read_text().strip())
        assert line["password"] != "secret123"  # pragma: allowlist secret


# ---------------------------------------------------------------------------
# log_trial_result / _flush_trial_buffer
# ---------------------------------------------------------------------------


class TestLogTrialResult:
    def test_trial_added_to_buffer(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path, buffer_size=10)
        log.log_trial_result(_make_trial())
        assert len(log._trial_buffer) == 1

    def test_buffer_flushed_at_size(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path, buffer_size=2)
        log.log_trial_result(_make_trial(trial_id="t1"))
        log.log_trial_result(_make_trial(trial_id="t2"))
        # Buffer flushed → empty
        assert len(log._trial_buffer) == 0

    def test_cloud_mode_skips(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path, execution_mode="cloud")
        log.log_trial_result(_make_trial())
        assert len(log._trial_buffer) == 0

    def test_flush_creates_jsonl_file(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path, buffer_size=1)
        log.log_trial_result(_make_trial())
        trials_files = list((log.run_path / "trials").glob("*.jsonl"))
        assert len(trials_files) >= 1

    def test_flush_with_example_results(self, tmp_path: Path) -> None:
        meta = {"example_results": [{"id": "ex1", "score": 0.9}]}
        log = _make_logger(tmp_path, buffer_size=1)
        log.log_trial_result(_make_trial(metadata=meta))
        # Should also create a trial detail file
        detail_files = list((log.run_path / "trials").glob("trial_*_v*.json"))
        assert len(detail_files) >= 1

    def test_empty_flush_noop(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path)
        log._flush_trial_buffer()
        # No files created
        trials_files = list((log.run_path / "trials").glob("*.jsonl"))
        assert len(trials_files) == 0


# ---------------------------------------------------------------------------
# log_metrics_update
# ---------------------------------------------------------------------------


class TestLogMetricsUpdate:
    def test_creates_metric_files(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path)
        log.log_metrics_update({"accuracy": 0.9, "cost": 0.05})
        metric_files = list((log.run_path / "metrics").glob("*.jsonl"))
        assert len(metric_files) == 2

    def test_skips_none_values(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path)
        log.log_metrics_update({"accuracy": 0.9, "cost": None})
        metric_files = list((log.run_path / "metrics").glob("*.jsonl"))
        assert len(metric_files) == 1

    def test_entry_has_timestamp_and_value(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path)
        log.log_metrics_update({"accuracy": 0.95})
        metric_file = list((log.run_path / "metrics").glob("*.jsonl"))[0]
        entry = json.loads(metric_file.read_text().strip())
        assert "timestamp" in entry
        assert entry["value"] == 0.95


# ---------------------------------------------------------------------------
# save_checkpoint
# ---------------------------------------------------------------------------


class TestSaveCheckpoint:
    def test_creates_checkpoint_files(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path)
        trials = [_make_trial()]
        log.save_checkpoint(
            optimizer_state={"best_value": 0.95},
            trials_history=trials,
            trial_count=5,
        )
        checkpoint_dir = log.run_path / "checkpoints"
        # Main checkpoint file
        checkpoint_files = list(checkpoint_dir.glob("checkpoint_00005_v*.json"))
        assert len(checkpoint_files) == 1
        # Latest pointer
        latest_files = list(checkpoint_dir.glob("latest_checkpoint_v*.json"))
        assert len(latest_files) == 1
        # Trial history
        history_files = list(checkpoint_dir.glob("trial_history_v*.json"))
        assert len(history_files) == 1

    def test_checkpoint_content(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path)
        trial = _make_trial(status="completed")
        log.save_checkpoint(
            optimizer_state={"step": 10},
            trials_history=[trial],
            trial_count=3,
        )
        checkpoint_file = list(
            (log.run_path / "checkpoints").glob("checkpoint_00003_v*.json")
        )[0]
        data = json.loads(checkpoint_file.read_text())
        assert data["trial_count"] == 3
        assert data["optimizer_state"]["step"] == 10
        assert data["trials_summary"]["total"] == 1
        assert data["trials_summary"]["successful"] == 1
        assert "random_state" in data


# ---------------------------------------------------------------------------
# load_checkpoint
# ---------------------------------------------------------------------------


class TestLoadCheckpoint:
    def _setup_checkpoint_files(self, base_path: Path) -> Path:
        """Set up checkpoint files at the expected path and return run_path."""
        run_path = base_path / "experiments" / "test_exp" / "runs" / "run_001"
        checkpoints = run_path / "checkpoints"
        checkpoints.mkdir(parents=True)
        # Also need meta dir for version_info
        (run_path / "meta").mkdir(parents=True)

        latest = {"checkpoint_file": "checkpoint_00005_v2.json"}
        (checkpoints / "latest_checkpoint_v2.json").write_text(json.dumps(latest))

        checkpoint = {"trial_count": 5, "optimizer_state": {"val": 0.9}}
        (checkpoints / "checkpoint_00005_v2.json").write_text(json.dumps(checkpoint))

        history = [{"trial_id": "t1", "status": "completed"}]
        (checkpoints / "trial_history_v2.json").write_text(json.dumps(history))

        return run_path

    def test_loads_data(self, tmp_path: Path) -> None:
        self._setup_checkpoint_files(tmp_path)
        result = OptimizationLogger.load_checkpoint("test_exp", "run_001", tmp_path)
        assert result["trial_count"] == 5
        assert result["trial_history"][0]["trial_id"] == "t1"
        assert "run_path" in result

    def test_not_found(self, tmp_path: Path) -> None:
        run_path = tmp_path / "experiments" / "nope" / "runs" / "run_001"
        run_path.mkdir(parents=True)
        (run_path / "checkpoints").mkdir()
        (run_path / "meta").mkdir()
        with pytest.raises(FileNotFoundError):
            OptimizationLogger.load_checkpoint("nope", "run_001", tmp_path)


# ---------------------------------------------------------------------------
# log_results
# ---------------------------------------------------------------------------


class TestLogResults:
    def test_with_result_object(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path)
        result = _make_opt_result()
        log.log_results(result)
        artifacts = list((log.run_path / "artifacts").glob("best_config_v*.json"))
        assert len(artifacts) == 1
        data = json.loads(artifacts[0].read_text())
        assert data["score"] == 0.95

    def test_with_dict(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path)
        log.log_results({"best_config": {"model": "gpt-4o"}, "best_score": 0.8})
        artifacts = list((log.run_path / "artifacts").glob("best_config_v*.json"))
        assert len(artifacts) == 1

    def test_with_raw_results(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path)
        log.log_results("some raw string")
        artifacts = list((log.run_path / "artifacts").glob("best_config_v*.json"))
        data = json.loads(artifacts[0].read_text())
        assert data["full_results"]["raw_results"] == "some raw string"


# ---------------------------------------------------------------------------
# log_config / log_objectives
# ---------------------------------------------------------------------------


class TestLogConfigAndObjectives:
    def test_log_config(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path)
        log.log_config({"model": "gpt-4o", "temperature": 0.7})
        config_files = list((log.run_path / "meta").glob("config_v*.json"))
        assert len(config_files) == 1
        data = json.loads(config_files[0].read_text())
        assert data["model"] == "gpt-4o"

    def test_log_objectives_with_to_dict(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path)
        mock_obj = MagicMock()
        mock_obj.to_dict.return_value = {
            "objectives": [
                {"name": "acc", "orientation": "maximize", "weight": 1.0},
            ]
        }
        log.log_objectives(mock_obj)
        obj_files = list((log.run_path / "meta").glob("objectives_v*.json"))
        assert len(obj_files) == 1
        data = json.loads(obj_files[0].read_text())
        assert "summary" in data

    def test_log_objectives_plain_dict(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path)
        log.log_objectives({"custom": "objectives"})
        obj_files = list((log.run_path / "meta").glob("objectives_v*.json"))
        assert len(obj_files) == 1


# ---------------------------------------------------------------------------
# log_session_end
# ---------------------------------------------------------------------------


class TestLogSessionEnd:
    def test_basic_session_end(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path)
        result = _make_opt_result()
        log.log_session_end(result)

        # Metrics summary created
        summary_files = list((log.run_path / "metrics").glob("metrics_summary_v*.json"))
        assert len(summary_files) == 1

        # Best config artifact
        artifacts = list((log.run_path / "artifacts").glob("best_config_v*.json"))
        assert len(artifacts) == 1

        # Status file
        status_files = list(log.run_path.glob("status_v*.json"))
        assert len(status_files) == 1
        status_data = json.loads(status_files[0].read_text())
        assert status_data["status"] == "completed"

    def test_session_end_with_error(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path)
        result = _make_opt_result()
        log.log_session_end(result, error="something failed")

        status_files = list(log.run_path.glob("status_v*.json"))
        status_data = json.loads(status_files[0].read_text())
        assert status_data["status"] == "failed"

    def test_session_end_with_weighted_results(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path)
        result = _make_opt_result()
        log.log_session_end(result, weighted_results={"weighted": True})

        weighted_files = list(
            (log.run_path / "artifacts").glob("weighted_results_v*.json")
        )
        assert len(weighted_files) == 1

    def test_session_end_flushes_buffer(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path, buffer_size=100)
        log.log_trial_result(_make_trial())
        assert len(log._trial_buffer) == 1

        result = _make_opt_result()
        log.log_session_end(result)
        assert len(log._trial_buffer) == 0

    def test_session_end_multi_objective(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path)
        trial = _make_trial(metrics={"accuracy": 0.9, "cost": 0.05})
        result = _make_opt_result(trials=[trial], objectives=["accuracy", "cost"])

        mock_pareto = MagicMock()
        mock_pareto.calculate_pareto_front.return_value = [
            MagicMock(
                config={"model": "gpt-4o"}, objectives={"accuracy": 0.9, "cost": 0.05}
            )
        ]

        with patch(
            "traigent.utils.multi_objective.ParetoFrontCalculator",
            return_value=mock_pareto,
        ):
            log.log_session_end(result)

        pareto_files = list((log.run_path / "artifacts").glob("pareto_front_v*.json"))
        assert len(pareto_files) == 1

    def test_session_end_creates_manifest(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path)
        result = _make_opt_result()
        log.log_session_end(result)

        manifest_files = list((log.run_path / "meta").glob("manifest_v*.json"))
        assert len(manifest_files) == 1
        data = json.loads(manifest_files[0].read_text())
        assert data.get("finalized") is True


# ---------------------------------------------------------------------------
# _update_experiment_index
# ---------------------------------------------------------------------------


class TestUpdateExperimentIndex:
    def test_creates_index(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path)
        log._update_experiment_index()

        index_file = tmp_path / "index.json"
        assert index_file.exists()
        data = json.loads(index_file.read_text())
        assert log.experiment_name in data["experiments"]
        runs = data["experiments"][log.experiment_name]["runs"]
        assert len(runs) == 1
        assert runs[0]["run_id"] == log.run_id

    def test_no_duplicate_run_ids(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path)
        log._update_experiment_index()
        log._update_experiment_index()

        data = json.loads((tmp_path / "index.json").read_text())
        runs = data["experiments"][log.experiment_name]["runs"]
        assert len(runs) == 1

    def test_updates_existing_index(self, tmp_path: Path) -> None:
        log1 = _make_logger(tmp_path, session_id="sess_aaa_11111")
        log1._update_experiment_index()

        log2 = _make_logger(tmp_path, session_id="sess_bbb_22222")
        log2._update_experiment_index()

        data = json.loads((tmp_path / "index.json").read_text())
        runs = data["experiments"]["test_exp"]["runs"]
        assert len(runs) == 2
