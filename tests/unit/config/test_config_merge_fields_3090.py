"""TraigentConfig.merge / merge_with_context must not drop any field."""

import dataclasses
from typing import Any

import pytest

from traigent.config.context import merge_with_context, set_config
from traigent.config.types import TraigentConfig

# Non-default value for every dataclass field except the legacy selectors,
# which have dedicated characterization tests.
_NON_DEFAULTS: dict[str, Any] = {
    "model": "m",
    "temperature": 0.5,
    "max_tokens": 10,
    "top_p": 0.5,
    "frequency_penalty": 0.5,
    "presence_penalty": 0.5,
    "stop_sequences": ["x"],
    "seed": 3,
    "custom_params": {"k": "v"},
    "algorithm": "grid",
    "offline": True,
    "local_storage_path": "/tmp/traigent-merge-test",
    "minimal_logging": False,
    "auto_sync": True,
    "strict_metrics_nulls": True,
    "comparability_mode": "strict",
    "content_identity": False,
    "enable_usage_analytics": False,
    "analytics_endpoint": "https://example.invalid/a",
    "anonymous_user_id": "anon",
    "no_egress": True,
    "result_source": "src",
    "fallback_reason": "r",
    "fallback_reason_code": "c",
    "persistence_status": "s",
    "persistence_reason": "pr",
    "persistence_rejection_reason": "prr",
}
_SKIPPED = {"execution_mode", "privacy_enabled", "execution_policy"}


def _full_config() -> TraigentConfig:
    return TraigentConfig(**_NON_DEFAULTS)


def test_non_defaults_cover_every_field() -> None:
    names = {f.name for f in dataclasses.fields(TraigentConfig)}
    assert names - _SKIPPED == set(_NON_DEFAULTS)


@pytest.mark.parametrize("name", sorted(_NON_DEFAULTS))
def test_merge_empty_dict_keeps_every_field(name: str) -> None:
    merged = _full_config().merge({})
    assert getattr(merged, name) == _NON_DEFAULTS[name]


@pytest.mark.parametrize("name", sorted(_NON_DEFAULTS))
def test_merge_default_config_keeps_every_field(name: str) -> None:
    merged = _full_config().merge(TraigentConfig())
    assert getattr(merged, name) == _NON_DEFAULTS[name]


def test_other_config_overrides_non_default_fields() -> None:
    merged = TraigentConfig().merge(_full_config())
    for name, value in _NON_DEFAULTS.items():
        assert getattr(merged, name) == value, name


def test_no_egress_never_loosened() -> None:
    base = TraigentConfig(no_egress=True)
    assert base.merge(TraigentConfig(no_egress=False)).no_egress is True
    assert base.merge({"no_egress": False}).no_egress is True
    assert TraigentConfig().merge({"no_egress": True}).no_egress is True
    assert TraigentConfig().merge(TraigentConfig(no_egress=True)).no_egress is True


def test_analytics_and_content_identity_opt_outs_win() -> None:
    base = TraigentConfig(enable_usage_analytics=False, content_identity=False)
    merged = base.merge(
        TraigentConfig(enable_usage_analytics=True, content_identity=True)
    )
    assert merged.enable_usage_analytics is False
    assert merged.content_identity is False


def test_merge_with_context_keeps_context_privacy_fields() -> None:
    set_config(TraigentConfig(no_egress=True, enable_usage_analytics=False))
    try:
        merged = merge_with_context({"temperature": 0.2})
    finally:
        set_config(None)
    assert isinstance(merged, TraigentConfig)
    assert merged.no_egress is True
    assert merged.enable_usage_analytics is False
