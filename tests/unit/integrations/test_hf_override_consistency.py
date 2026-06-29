"""Regression guard for issue #1570: HuggingFace override consistency.

Ensures that every class declared by HuggingFacePlugin.get_target_classes() has
corresponding entries in the static PARAMETER_MAPPINGS and METHOD_MAPPINGS used by
FrameworkOverrideManager, and that the convenience override functions target the
correct huggingface_hub.* classes.

Key invariant enforced here: the class-level PARAMETER_MAPPINGS for InferenceClient /
AsyncInferenceClient must contain ONLY constructor-valid kwargs. Generation params
(temperature, max_tokens, etc.) must NOT appear there — they would be blindly injected
into __init__ and raise TypeError (the bug fixed in #1570). Generation params live in
METHOD_MAPPINGS (for filtering) and METHOD_PARAMETER_TRANSLATIONS (for name translation).
"""

import pytest

from traigent.integrations.llms.huggingface_plugin import HuggingFacePlugin
from traigent.integrations.mappings import (
    METHOD_MAPPINGS,
    METHOD_PARAMETER_TRANSLATIONS,
    PARAMETER_MAPPINGS,
)


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
    """Class-level PARAMETER_MAPPINGS for InferenceClient must contain ONLY
    constructor-valid params. Generation params must NOT be present — they would be
    injected into InferenceClient.__init__ and raise TypeError (issue #1570 fix).

    InferenceClient.__init__ accepts: model, token, timeout, base_url, headers,
    cookies, api_key. It does NOT accept: temperature, max_tokens/max_new_tokens,
    top_p, top_k, stop, stream.
    """
    mapping = PARAMETER_MAPPINGS["huggingface_hub.InferenceClient"]

    # model IS a valid constructor kwarg
    assert mapping["model"] == "model"

    # Generation params must NOT be in the class-level mapping (TypeError risk)
    for gen_param in ("temperature", "max_tokens", "top_p", "top_k", "stop", "stream"):
        assert gen_param not in mapping, (
            f"{gen_param!r} is a generation param and must NOT appear in the class-level "
            f"PARAMETER_MAPPINGS for huggingface_hub.InferenceClient. "
            f"Injecting it into __init__ causes TypeError (issue #1570). "
            f"Place it in METHOD_MAPPINGS + METHOD_PARAMETER_TRANSLATIONS instead."
        )

    # max_tokens → max_new_tokens translation lives in METHOD_PARAMETER_TRANSLATIONS
    text_gen_translations = METHOD_PARAMETER_TRANSLATIONS.get(
        "huggingface_hub.InferenceClient", {}
    ).get("text_generation", {})
    assert "max_tokens" in text_gen_translations, (
        "max_tokens → max_new_tokens translation must be in "
        "METHOD_PARAMETER_TRANSLATIONS['huggingface_hub.InferenceClient']['text_generation']"
    )
    assert text_gen_translations["max_tokens"] == "max_new_tokens", (
        "text_generation must translate max_tokens → max_new_tokens (the HF framework name)"
    )

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


@pytest.mark.unit
def test_hf_constructor_no_generation_param_injection():
    """Codex repro #1570: enabling the HF override and constructing InferenceClient
    with an active config context containing temperature/max_tokens must NOT raise
    TypeError.

    Before fix: TypeError: InferenceClient.__init__() got an unexpected keyword
    argument 'temperature' (and/or 'max_new_tokens', 'top_p', etc.) because
    generation params were present in the class-level PARAMETER_MAPPINGS and the
    constructor override injected ALL mapped params regardless of constructor validity.

    After fix: class-level PARAMETER_MAPPINGS contains only constructor-valid kwargs
    (model), so the constructor override injects nothing that __init__ doesn't accept.
    """
    import huggingface_hub

    from traigent.config.context import config_context, set_config
    from traigent.config.types import TraigentConfig
    from traigent.integrations.framework_override import FrameworkOverrideManager

    manager = FrameworkOverrideManager()
    manager.activate_overrides(["huggingface_hub.InferenceClient"])

    # Config that used to cause TypeError when injected into __init__
    config = TraigentConfig(temperature=0.5, max_tokens=42, top_p=0.9)
    token = set_config(config)
    try:
        # This must not raise TypeError.
        client = huggingface_hub.InferenceClient(model="gpt2")
        assert client is not None, "InferenceClient instance must be created without error"
    finally:
        config_context.reset(token)
        manager.deactivate_overrides()


@pytest.mark.unit
def test_hf_manager_uses_all_target_classes():
    """Every class the HF plugin declares must have a mapping the FrameworkOverrideManager
    actually loads (i.e. present in both PARAMETER_MAPPINGS and METHOD_MAPPINGS).

    Regression guard: a class missing from either mapping silently disables all param
    injection for that class even when the override is active.
    """
    from traigent.integrations.framework_override import FrameworkOverrideManager

    manager = FrameworkOverrideManager()
    plugin = HuggingFacePlugin()

    for cls in plugin.get_target_classes():
        assert cls in manager._parameter_mappings, (
            f"FrameworkOverrideManager._parameter_mappings has no entry for {cls!r} "
            f"declared by HuggingFacePlugin.get_target_classes(). "
            f"Add it to PARAMETER_MAPPINGS in mappings.py."
        )
        assert cls in manager._method_mappings, (
            f"FrameworkOverrideManager._method_mappings has no entry for {cls!r} "
            f"declared by HuggingFacePlugin.get_target_classes(). "
            f"Add it to METHOD_MAPPINGS in mappings.py."
        )
