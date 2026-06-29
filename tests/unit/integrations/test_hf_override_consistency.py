"""Regression guard for issue #1570: HuggingFace override consistency.

Ensures that every class declared by HuggingFacePlugin.get_target_classes() has
corresponding entries in the static PARAMETER_MAPPINGS and METHOD_MAPPINGS used by
FrameworkOverrideManager, and that the convenience override functions target the
correct huggingface_hub.* classes.
"""

import pytest

from traigent.integrations.llms.huggingface_plugin import HuggingFacePlugin
from traigent.integrations.mappings import METHOD_MAPPINGS, PARAMETER_MAPPINGS


@pytest.mark.unit
def test_hf_plugin_classes_in_parameter_mappings():
    """Every class declared by the HF plugin must have a PARAMETER_MAPPINGS entry."""
    plugin = HuggingFacePlugin()
    for cls in plugin.get_target_classes():
        assert cls in PARAMETER_MAPPINGS, (
            f"HuggingFacePlugin declares target class {cls!r} but PARAMETER_MAPPINGS "
            f"has no entry for it — override_huggingface() would inject no params."
        )


@pytest.mark.unit
def test_hf_plugin_classes_in_method_mappings():
    """Every class declared by the HF plugin must have a METHOD_MAPPINGS entry."""
    plugin = HuggingFacePlugin()
    for cls in plugin.get_target_classes():
        assert cls in METHOD_MAPPINGS, (
            f"HuggingFacePlugin declares target class {cls!r} but METHOD_MAPPINGS "
            f"has no entry for it — method-level param injection is fully disabled."
        )


@pytest.mark.unit
def test_hf_plugin_methods_in_method_mappings():
    """Every method declared by the HF plugin must appear in METHOD_MAPPINGS."""
    plugin = HuggingFacePlugin()
    for cls, methods in plugin.get_target_methods().items():
        assert cls in METHOD_MAPPINGS, (
            f"HuggingFacePlugin declares methods for {cls!r} but METHOD_MAPPINGS "
            f"has no entry for it."
        )
        for method in methods:
            assert method in METHOD_MAPPINGS[cls], (
                f"HuggingFacePlugin declares method {method!r} on {cls!r} but "
                f"METHOD_MAPPINGS[{cls!r}] has no entry for it."
            )


@pytest.mark.unit
def test_hf_parameter_mappings_have_core_params():
    """PARAMETER_MAPPINGS for huggingface_hub.InferenceClient covers key tunable params."""
    mapping = PARAMETER_MAPPINGS["huggingface_hub.InferenceClient"]
    assert mapping["model"] == "model"
    assert mapping["temperature"] == "temperature"
    assert mapping["max_tokens"] == "max_new_tokens"
    assert mapping["top_p"] == "top_p"
    assert mapping["stream"] == "stream"

    async_mapping = PARAMETER_MAPPINGS["huggingface_hub.AsyncInferenceClient"]
    assert async_mapping == mapping, (
        "Sync and async InferenceClient parameter mappings should be identical."
    )


@pytest.mark.unit
def test_hf_method_mappings_text_generation_includes_max_tokens():
    """text_generation METHOD_MAPPINGS must include max_tokens so max_new_tokens is injected."""
    assert "max_tokens" in METHOD_MAPPINGS["huggingface_hub.InferenceClient"]["text_generation"]
    assert "max_tokens" in METHOD_MAPPINGS["huggingface_hub.AsyncInferenceClient"]["text_generation"]


@pytest.mark.unit
def test_hf_method_mappings_chat_completion_excludes_max_tokens():
    """chat_completion must NOT list max_tokens: PARAMETER_MAPPINGS maps it to max_new_tokens
    which is not a valid chat_completion kwarg."""
    assert "max_tokens" not in METHOD_MAPPINGS["huggingface_hub.InferenceClient"]["chat_completion"]
    assert "max_tokens" not in METHOD_MAPPINGS["huggingface_hub.AsyncInferenceClient"]["chat_completion"]


@pytest.mark.unit
def test_override_huggingface_targets_huggingface_hub_classes(monkeypatch):
    """override_huggingface() must enable huggingface_hub.* classes, not transformers.*."""
    from traigent.integrations import framework_override as fo

    captured: list[list[str]] = []

    def fake_enable(targets):
        captured.append(list(targets))

    monkeypatch.setattr(fo, "enable_framework_overrides", fake_enable)
    fo.override_huggingface()

    assert captured, "override_huggingface() did not call enable_framework_overrides()"
    enabled = captured[0]

    assert "huggingface_hub.InferenceClient" in enabled, (
        f"override_huggingface() should enable huggingface_hub.InferenceClient, got {enabled}"
    )
    assert "huggingface_hub.AsyncInferenceClient" in enabled, (
        f"override_huggingface() should enable huggingface_hub.AsyncInferenceClient, got {enabled}"
    )
    assert "transformers.pipeline" not in enabled, (
        "override_huggingface() must NOT enable transformers.pipeline (wrong surface)"
    )
    assert "transformers.AutoModelForCausalLM" not in enabled, (
        "override_huggingface() must NOT enable transformers.AutoModelForCausalLM (wrong surface)"
    )


@pytest.mark.unit
def test_override_all_platforms_includes_huggingface_hub_classes(monkeypatch):
    """override_all_platforms() must include huggingface_hub.* classes, not transformers.*."""
    from traigent.integrations import framework_override as fo

    captured: list[list[str]] = []

    def fake_enable(targets):
        captured.append(list(targets))

    monkeypatch.setattr(fo, "enable_framework_overrides", fake_enable)
    fo.override_all_platforms()

    assert captured, "override_all_platforms() did not call enable_framework_overrides()"
    enabled = captured[0]

    assert "huggingface_hub.InferenceClient" in enabled
    assert "huggingface_hub.AsyncInferenceClient" in enabled
    assert "transformers.pipeline" not in enabled
    assert "transformers.AutoModelForCausalLM" not in enabled
