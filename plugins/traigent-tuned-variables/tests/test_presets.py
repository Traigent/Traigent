"""Tests for domain preset classes."""

from __future__ import annotations

import os
from unittest.mock import patch



class TestLLMPresets:
    """Tests for LLMPresets factory methods."""

    def test_temperature_default(self):
        """Test default temperature range."""
        from traigent_tuned_variables import LLMPresets

        temp = LLMPresets.temperature()

        assert temp.low == 0.0
        assert temp.high == 1.0
        assert temp.default == 0.7
        assert temp.name == "temperature"

    def test_temperature_conservative(self):
        """Test conservative temperature range for factual tasks."""
        from traigent_tuned_variables import LLMPresets

        temp = LLMPresets.temperature(conservative=True)

        assert temp.low == 0.0
        assert temp.high == 0.5
        assert temp.default == 0.2
        assert temp.name == "temperature"

    def test_temperature_creative(self):
        """Test creative temperature range."""
        from traigent_tuned_variables import LLMPresets

        temp = LLMPresets.temperature(creative=True)

        assert temp.low == 0.7
        assert temp.high == 1.5
        assert temp.default == 1.0
        assert temp.name == "temperature"

    def test_top_p(self):
        """Test top_p preset."""
        from traigent_tuned_variables import LLMPresets

        top_p = LLMPresets.top_p()

        assert top_p.low == 0.1
        assert top_p.high == 1.0
        assert top_p.default == 0.9
        assert top_p.name == "top_p"

    def test_max_tokens_short(self):
        """Test max_tokens for short tasks."""
        from traigent_tuned_variables import LLMPresets

        max_tokens = LLMPresets.max_tokens(task="short")

        assert max_tokens.low == 50
        assert max_tokens.high == 256
        assert max_tokens.name == "max_tokens"

    def test_max_tokens_long(self):
        """Test max_tokens for long tasks."""
        from traigent_tuned_variables import LLMPresets

        max_tokens = LLMPresets.max_tokens(task="long")

        assert max_tokens.low == 1024
        assert max_tokens.high == 4096
        assert max_tokens.name == "max_tokens"

    def test_model_default(self):
        """Test model preset returns Choices."""
        from traigent_tuned_variables import LLMPresets

        model = LLMPresets.model()

        assert model.name == "model"
        assert len(model.values) > 0

    def test_model_from_env(self):
        """Test model list from environment variable."""
        from traigent_tuned_variables import LLMPresets

        with patch.dict(
            os.environ, {"TRAIGENT_MODELS_OPENAI_FAST": "gpt-4o-mini,gpt-4o"}
        ):
            model = LLMPresets.model(provider="openai", tier="fast")

            assert "gpt-4o-mini" in model.values
            assert "gpt-4o" in model.values


class TestRAGPresets:
    """Tests for RAGPresets factory methods."""

    def test_k_retrieval_default(self):
        """Test default k retrieval range."""
        from traigent_tuned_variables import RAGPresets

        k = RAGPresets.k_retrieval()

        assert k.low == 1
        assert k.high == 10
        assert k.default == 3
        assert k.name == "k"

    def test_k_retrieval_custom_max(self):
        """Test k retrieval with custom max."""
        from traigent_tuned_variables import RAGPresets

        k = RAGPresets.k_retrieval(max_k=20)

        assert k.low == 1
        assert k.high == 20

    def test_chunk_size(self):
        """Test chunk size preset."""
        from traigent_tuned_variables import RAGPresets

        chunk_size = RAGPresets.chunk_size()

        assert chunk_size.low == 100
        assert chunk_size.high == 1000
        assert chunk_size.default == 500
        assert chunk_size.name == "chunk_size"

    def test_chunk_overlap(self):
        """Test chunk overlap preset."""
        from traigent_tuned_variables import RAGPresets

        overlap = RAGPresets.chunk_overlap()

        assert overlap.low == 0
        assert overlap.high == 200
        assert overlap.default == 50
        assert overlap.name == "chunk_overlap"

    def test_similarity_threshold(self):
        """Test similarity threshold preset."""
        from traigent_tuned_variables import RAGPresets

        threshold = RAGPresets.similarity_threshold()

        assert threshold.low == 0.0
        assert threshold.high == 1.0
        assert threshold.default == 0.5
        assert threshold.name == "similarity_threshold"

    def test_mmr_lambda(self):
        """Test MMR lambda preset."""
        from traigent_tuned_variables import RAGPresets

        mmr = RAGPresets.mmr_lambda()

        assert mmr.low == 0.0
        assert mmr.high == 1.0
        assert mmr.default == 0.5
        assert mmr.name == "mmr_lambda"


class TestPromptingPresets:
    """Tests for PromptingPresets factory methods."""

    def test_strategy(self):
        """Test prompting strategy preset."""
        from traigent_tuned_variables import PromptingPresets

        strategy = PromptingPresets.strategy()

        assert strategy.name == "prompting_strategy"
        assert "direct" in strategy.values
        assert "chain_of_thought" in strategy.values
        assert strategy.default == "direct"

    def test_context_format(self):
        """Test context format preset."""
        from traigent_tuned_variables import PromptingPresets

        fmt = PromptingPresets.context_format()

        assert fmt.name == "context_format"
        assert "bullet" in fmt.values
        assert "xml" in fmt.values

    def test_few_shot_count(self):
        """Test few shot count preset."""
        from traigent_tuned_variables import PromptingPresets

        count = PromptingPresets.few_shot_count()

        assert count.low == 0
        assert count.high == 10
        assert count.default == 3
        assert count.name == "few_shot_count"
