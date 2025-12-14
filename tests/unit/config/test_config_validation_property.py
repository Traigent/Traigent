"""Property-based tests for TraigentConfig validation using hypothesis."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from traigent.config.types import ExecutionMode, TraigentConfig, resolve_execution_mode
from traigent.utils.exceptions import ValidationError

# ==============================================================================
# Property-based tests for temperature validation
# ==============================================================================


@given(st.floats(min_value=0.0, max_value=2.0))
def test_temperature_accepts_valid_range(temperature):
    """Property: Any temperature in [0.0, 2.0] should be accepted."""
    config = TraigentConfig(temperature=temperature)
    assert config.temperature == temperature


@given(st.floats().filter(lambda x: x < 0.0 or x > 2.0))
def test_temperature_rejects_invalid_range(temperature):
    """Property: Any temperature outside [0.0, 2.0] should be rejected."""
    with pytest.raises((ValueError, ValidationError)):
        TraigentConfig(temperature=temperature)


# ==============================================================================
# Property-based tests for max_tokens validation
# ==============================================================================


@given(st.integers(min_value=1, max_value=1000000))
def test_max_tokens_accepts_positive_integers(max_tokens):
    """Property: Any positive integer should be accepted for max_tokens."""
    config = TraigentConfig(max_tokens=max_tokens)
    assert config.max_tokens == max_tokens


@given(st.integers(max_value=0))
def test_max_tokens_rejects_non_positive(max_tokens):
    """Property: Zero or negative integers should be rejected for max_tokens."""
    with pytest.raises((ValueError, ValidationError)):
        TraigentConfig(max_tokens=max_tokens)


# ==============================================================================
# Property-based tests for top_p validation
# ==============================================================================


@given(st.floats(min_value=0.0, max_value=1.0))
def test_top_p_accepts_valid_probability(top_p):
    """Property: Any value in [0.0, 1.0] should be accepted for top_p."""
    config = TraigentConfig(top_p=top_p)
    assert config.top_p == top_p


@given(st.floats().filter(lambda x: x < 0.0 or x > 1.0))
def test_top_p_rejects_invalid_probability(top_p):
    """Property: Any value outside [0.0, 1.0] should be rejected for top_p."""
    with pytest.raises((ValueError, ValidationError)):
        TraigentConfig(top_p=top_p)


# ==============================================================================
# Property-based tests for penalties validation
# ==============================================================================


@given(st.floats(min_value=-2.0, max_value=2.0))
def test_frequency_penalty_accepts_valid_range(penalty):
    """Property: Any penalty in [-2.0, 2.0] should be accepted."""
    config = TraigentConfig(frequency_penalty=penalty)
    assert config.frequency_penalty == penalty


@given(st.floats(min_value=-2.0, max_value=2.0))
def test_presence_penalty_accepts_valid_range(penalty):
    """Property: Any penalty in [-2.0, 2.0] should be accepted."""
    config = TraigentConfig(presence_penalty=penalty)
    assert config.presence_penalty == penalty


@given(st.floats().filter(lambda x: x < -2.0 or x > 2.0))
def test_frequency_penalty_rejects_invalid_range(penalty):
    """Property: Any penalty outside [-2.0, 2.0] should be rejected."""
    with pytest.raises((ValueError, ValidationError)):
        TraigentConfig(frequency_penalty=penalty)


@given(st.floats().filter(lambda x: x < -2.0 or x > 2.0))
def test_presence_penalty_rejects_invalid_range(penalty):
    """Property: Any penalty outside [-2.0, 2.0] should be rejected."""
    with pytest.raises((ValueError, ValidationError)):
        TraigentConfig(presence_penalty=penalty)


# ==============================================================================
# Property-based tests for execution mode
# ==============================================================================


@given(
    st.sampled_from(
        [
            "edge_analytics",
            "privacy",
            "hybrid",
            "standard",
            "cloud",
            ExecutionMode.EDGE_ANALYTICS,
            ExecutionMode.PRIVACY,
            ExecutionMode.HYBRID,
            ExecutionMode.STANDARD,
            ExecutionMode.CLOUD,
        ]
    )
)
def test_execution_mode_accepts_valid_values(mode):
    """Property: All valid execution modes should be accepted."""
    config = TraigentConfig(execution_mode=mode)
    # Privacy mode should be converted to hybrid with privacy_enabled
    if isinstance(mode, str) and mode == "privacy":
        assert config.execution_mode == "hybrid"
        assert config.privacy_enabled is True
    elif isinstance(mode, ExecutionMode) and mode == ExecutionMode.PRIVACY:
        assert config.execution_mode == "hybrid"
        assert config.privacy_enabled is True
    else:
        expected = mode.value if isinstance(mode, ExecutionMode) else mode
        assert config.execution_mode == expected


@given(
    st.text().filter(
        lambda x: x
        not in ["edge_analytics", "privacy", "hybrid", "standard", "cloud", ""]
        and x.strip() != ""  # Exclude whitespace-only strings (treated as empty)
    )
)
def test_execution_mode_rejects_invalid_values(mode):
    """Property: Invalid execution mode strings should be rejected."""
    with pytest.raises((ValueError, ValidationError)):
        TraigentConfig(execution_mode=mode)


# ==============================================================================
# Property-based tests for to_dict/from_dict roundtrip
# ==============================================================================


@given(
    st.builds(
        TraigentConfig,
        model=st.one_of(st.none(), st.text(min_size=1, max_size=50)),
        temperature=st.one_of(st.none(), st.floats(min_value=0.0, max_value=2.0)),
        max_tokens=st.one_of(st.none(), st.integers(min_value=1, max_value=100000)),
        top_p=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0)),
        frequency_penalty=st.one_of(
            st.none(), st.floats(min_value=-2.0, max_value=2.0)
        ),
        presence_penalty=st.one_of(st.none(), st.floats(min_value=-2.0, max_value=2.0)),
    )
)
def test_config_roundtrip_preserves_values(config):
    """Property: to_dict/from_dict should preserve config values."""
    config_dict = config.to_dict()
    restored = TraigentConfig.from_dict(config_dict)

    # Compare non-None values
    if config.model is not None:
        assert restored.model == config.model
    if config.temperature is not None:
        assert restored.temperature == config.temperature
    if config.max_tokens is not None:
        assert restored.max_tokens == config.max_tokens
    if config.top_p is not None:
        assert restored.top_p == config.top_p
    if config.frequency_penalty is not None:
        assert restored.frequency_penalty == config.frequency_penalty
    if config.presence_penalty is not None:
        assert restored.presence_penalty == config.presence_penalty


# ==============================================================================
# Property-based tests for config merging
# ==============================================================================


@given(
    base_temp=st.one_of(st.none(), st.floats(min_value=0.0, max_value=2.0)),
    override_temp=st.one_of(st.none(), st.floats(min_value=0.0, max_value=2.0)),
)
def test_merge_takes_override_value_when_present(base_temp, override_temp):
    """Property: merge should prefer override values when present."""
    base = TraigentConfig(temperature=base_temp)
    override = TraigentConfig(temperature=override_temp)

    merged = base.merge(override)

    # Override value should be used if present, otherwise base value
    expected = override_temp if override_temp is not None else base_temp
    assert merged.temperature == expected


# ==============================================================================
# Property-based tests for custom_params
# ==============================================================================


@given(
    st.dictionaries(
        keys=st.text(min_size=1, max_size=20).filter(
            lambda k: k
            not in {
                "model",
                "temperature",
                "max_tokens",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
                "execution_mode",
                "local_storage_path",
            }
        ),
        values=st.one_of(
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.text(),
            st.booleans(),
        ),
        max_size=10,
    )
)
def test_custom_params_preserved(custom_params):
    """Property: Custom parameters should be preserved in config."""
    config = TraigentConfig(custom_params=custom_params)
    assert config.custom_params == custom_params

    # Should also be accessible via to_dict
    config_dict = config.to_dict()
    for key, value in custom_params.items():
        assert config_dict[key] == value


# ==============================================================================
# Property-based tests for resolve_execution_mode
# ==============================================================================


@given(
    st.sampled_from(
        ["edge_analytics", "privacy", "hybrid", "standard", "cloud", "", "  ", None]
    )
)
def test_resolve_execution_mode_handles_all_valid_inputs(mode_str):
    """Property: resolve_execution_mode should handle all valid inputs."""
    result = resolve_execution_mode(mode_str)
    assert isinstance(result, ExecutionMode)

    # Empty strings and None should use default
    if not mode_str or (isinstance(mode_str, str) and not mode_str.strip()):
        assert result == ExecutionMode.CLOUD  # default


@given(
    st.one_of(
        st.integers(),
        st.floats(),
        st.lists(st.text()),
        st.dictionaries(st.text(), st.text()),
    )
)
def test_resolve_execution_mode_rejects_invalid_types(invalid_value):
    """Property: resolve_execution_mode should reject non-string/non-enum types."""
    with pytest.raises(TypeError):
        resolve_execution_mode(invalid_value)
