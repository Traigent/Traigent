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


# --- review round 1: dict overrides, context routing, serialisation ---

_PRIVACY = ("no_egress", "enable_usage_analytics", "content_identity")


def test_context_config_with_dict_override_routes_privacy_to_fields() -> None:
    set_config(TraigentConfig())
    try:
        merged = merge_with_context(
            {"no_egress": True, "enable_usage_analytics": False}
        )
    finally:
        set_config(None)
    assert merged.no_egress is True
    assert merged.enable_usage_analytics is False
    assert "no_egress" not in merged.custom_params
    assert "enable_usage_analytics" not in merged.custom_params


def test_context_config_cannot_loosen_override_privacy() -> None:
    set_config(TraigentConfig(no_egress=False))
    try:
        merged = merge_with_context({"no_egress": True})
    finally:
        set_config(None)
    assert merged.no_egress is True


def test_dict_context_privacy_stricter_wins() -> None:
    set_config({"no_egress": False, "enable_usage_analytics": True})
    try:
        merged = merge_with_context(
            {"no_egress": True, "enable_usage_analytics": False}
        )
    finally:
        set_config(None)
    assert merged["no_egress"] is True
    assert merged["enable_usage_analytics"] is False


def test_merge_dict_override_does_not_leave_privacy_in_custom_params() -> None:
    merged = TraigentConfig(no_egress=True).merge({"no_egress": False})
    assert merged.no_egress is True
    assert "no_egress" not in merged.custom_params
    assert merged.to_dict().get("no_egress", True) is True


def test_existing_custom_params_privacy_key_is_stripped() -> None:
    base = TraigentConfig(custom_params={"no_egress": False, "keep": 1})
    merged = base.merge({"temperature": 0.1})
    assert merged.custom_params == {"keep": 1}


@pytest.mark.parametrize("override", [{"no_egress": False}, {}, {"no_egress": True}])
def test_to_dict_agrees_with_fields_for_privacy(override: dict) -> None:
    merged = TraigentConfig(no_egress=True, enable_usage_analytics=False).merge(
        {**override, "enable_usage_analytics": True}
    )
    serialised = merged.to_dict()
    for name in _PRIVACY:
        if name in serialised:
            assert serialised[name] == getattr(merged, name)


def test_explicit_default_dict_override_is_honoured_for_non_privacy() -> None:
    base = TraigentConfig(analytics_endpoint="https://example.invalid/a")
    assert base.merge({"analytics_endpoint": None}).analytics_endpoint is None
    base = TraigentConfig(result_source="x")
    assert base.merge({"result_source": None}).result_source is None


# --- review round 2: privacy values are normalised before merging ---


@pytest.mark.parametrize("raw,expected", [("false", False), (" FALSE ", False)])
def test_string_false_content_identity_is_normalised(raw: str, expected: bool) -> None:
    merged = TraigentConfig().merge({"content_identity": raw})
    assert merged.content_identity is expected


def test_string_true_no_egress_is_a_bool() -> None:
    merged = TraigentConfig().merge({"no_egress": " True "})
    assert merged.no_egress is True


def test_string_false_analytics_opt_out_wins() -> None:
    merged = TraigentConfig().merge({"enable_usage_analytics": "false"})
    assert merged.enable_usage_analytics is False


@pytest.mark.parametrize(
    "name", ["no_egress", "enable_usage_analytics", "content_identity"]
)
@pytest.mark.parametrize("bad", ["maybe", 1, 0, [], "yes"])
def test_invalid_privacy_value_raises_without_echo(name: str, bad: object) -> None:
    with pytest.raises(TypeError) as excinfo:
        TraigentConfig().merge({name: bad})
    assert name in str(excinfo.value)
    assert str(bad) not in str(excinfo.value) or str(bad) in name


def test_dict_context_single_sided_string_is_normalised() -> None:
    set_config({})
    try:
        merged = merge_with_context({"no_egress": "true"})
    finally:
        set_config(None)
    assert merged["no_egress"] is True


def test_config_override_explicit_defaults_reset_base() -> None:
    base = TraigentConfig(offline=True, algorithm="grid", temperature=0.5)
    merged = base.merge(TraigentConfig(offline=False, algorithm="auto"))
    assert merged.offline is False
    assert merged.algorithm == "auto"
    assert merged.temperature == 0.5


def test_config_override_untouched_defaults_keep_base() -> None:
    base = TraigentConfig(offline=True, algorithm="grid")
    merged = base.merge(TraigentConfig(model="m"))
    assert merged.offline is True
    assert merged.algorithm == "grid"
    assert merged.model == "m"


def test_config_override_post_init_assignment_is_explicit() -> None:
    base = TraigentConfig(offline=True)
    other = TraigentConfig()
    other.offline = False
    assert base.merge(other).offline is False
