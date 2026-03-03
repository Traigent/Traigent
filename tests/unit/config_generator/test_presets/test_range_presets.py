"""Tests for config_generator.presets.range_presets."""

from __future__ import annotations

import pytest

from traigent.config_generator.presets.range_presets import (
    all_canonical_names,
    get_preset_range,
    has_preset,
)


class TestGetPresetRange:
    def test_temperature(self) -> None:
        preset = get_preset_range("temperature")
        assert preset is not None
        assert preset["range_type"] == "Range"
        assert preset["kwargs"]["low"] == 0.0
        assert preset["kwargs"]["high"] == 1.0

    def test_max_tokens(self) -> None:
        preset = get_preset_range("max_tokens")
        assert preset is not None
        assert preset["range_type"] == "IntRange"
        assert preset["kwargs"]["low"] == 256
        assert preset["kwargs"]["high"] == 4096

    def test_model(self) -> None:
        preset = get_preset_range("model")
        assert preset is not None
        assert preset["range_type"] == "Choices"
        assert isinstance(preset["kwargs"]["values"], list)
        assert len(preset["kwargs"]["values"]) >= 2

    def test_k(self) -> None:
        preset = get_preset_range("k")
        assert preset is not None
        assert preset["range_type"] == "IntRange"

    def test_prompting_strategy(self) -> None:
        preset = get_preset_range("prompting_strategy")
        assert preset is not None
        assert preset["range_type"] == "Choices"
        assert "direct" in preset["kwargs"]["values"]

    def test_unknown_returns_none(self) -> None:
        assert get_preset_range("unknown_variable") is None

    @pytest.mark.parametrize(
        "name",
        [
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "similarity_threshold",
            "mmr_lambda",
            "chunk_overlap_ratio",
            "max_tokens",
            "k",
            "chunk_size",
            "chunk_overlap",
            "few_shot_count",
            "batch_size",
            "n",
            "seed",
            "top_k",
            "model",
            "prompting_strategy",
            "context_format",
            "retriever_type",
            "embedding_model",
            "reranker_model",
        ],
    )
    def test_all_presets_have_valid_structure(self, name: str) -> None:
        preset = get_preset_range(name)
        assert preset is not None, f"Missing preset for {name}"
        assert "range_type" in preset
        assert "kwargs" in preset
        assert preset["range_type"] in ("Range", "IntRange", "LogRange", "Choices")


class TestHasPreset:
    def test_known(self) -> None:
        assert has_preset("temperature") is True
        assert has_preset("model") is True

    def test_unknown(self) -> None:
        assert has_preset("nonexistent") is False


class TestAllCanonicalNames:
    def test_returns_frozenset(self) -> None:
        names = all_canonical_names()
        assert isinstance(names, frozenset)

    def test_contains_key_names(self) -> None:
        names = all_canonical_names()
        assert "temperature" in names
        assert "max_tokens" in names
        assert "model" in names

    def test_at_least_20_presets(self) -> None:
        # We have 22 presets in the catalog
        assert len(all_canonical_names()) >= 20
