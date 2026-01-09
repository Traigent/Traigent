"""Tests for TVL registry resolvers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from traigent.tvl.registry import DictRegistryResolver, FileRegistryResolver


class TestFileRegistryResolver:
    """Tests for FileRegistryResolver."""

    def test_resolve_yaml_registry(self, tmp_path: Path) -> None:
        """Test resolving from a YAML registry file."""
        registry_dir = tmp_path / "registries"
        registry_dir.mkdir()

        registry_file = registry_dir / "models.yaml"
        registry_file.write_text(
            yaml.dump(
                {
                    "items": [
                        {"id": "gpt-4o", "provider": "openai", "version": "2024-08"},
                        {
                            "id": "gpt-4o-mini",
                            "provider": "openai",
                            "version": "2024-07",
                        },
                        {
                            "id": "claude-3-opus",
                            "provider": "anthropic",
                            "version": "2024-02",
                        },
                    ]
                }
            )
        )

        resolver = FileRegistryResolver(registry_dir)
        result = resolver.resolve("models")

        assert result == ["gpt-4o", "gpt-4o-mini", "claude-3-opus"]

    def test_resolve_json_registry(self, tmp_path: Path) -> None:
        """Test resolving from a JSON registry file."""
        registry_dir = tmp_path / "registries"
        registry_dir.mkdir()

        registry_file = registry_dir / "scorers.json"
        registry_file.write_text(
            json.dumps(
                {
                    "items": [
                        {"id": "accuracy_v1"},
                        {"id": "f1_score"},
                    ]
                }
            )
        )

        resolver = FileRegistryResolver(registry_dir)
        result = resolver.resolve("scorers")

        assert result == ["accuracy_v1", "f1_score"]

    def test_filter_by_equality(self, tmp_path: Path) -> None:
        """Test filtering by equality expression."""
        registry_dir = tmp_path / "registries"
        registry_dir.mkdir()

        registry_file = registry_dir / "models.yaml"
        registry_file.write_text(
            yaml.dump(
                {
                    "items": [
                        {"id": "gpt-4o", "provider": "openai"},
                        {"id": "claude-3", "provider": "anthropic"},
                    ]
                }
            )
        )

        resolver = FileRegistryResolver(registry_dir)
        result = resolver.resolve("models", filter_expr="provider == 'openai'")

        assert result == ["gpt-4o"]

    def test_filter_by_equality_single_equals(self, tmp_path: Path) -> None:
        """Test filtering with single = operator."""
        registry_dir = tmp_path / "registries"
        registry_dir.mkdir()

        registry_file = registry_dir / "models.yaml"
        registry_file.write_text(
            yaml.dump(
                {
                    "items": [
                        {"id": "gpt-4o", "provider": "openai"},
                        {"id": "claude-3", "provider": "anthropic"},
                    ]
                }
            )
        )

        resolver = FileRegistryResolver(registry_dir)
        result = resolver.resolve("models", filter_expr="provider = 'anthropic'")

        assert result == ["claude-3"]

    def test_filter_by_inequality(self, tmp_path: Path) -> None:
        """Test filtering by inequality expression."""
        registry_dir = tmp_path / "registries"
        registry_dir.mkdir()

        registry_file = registry_dir / "models.yaml"
        registry_file.write_text(
            yaml.dump(
                {
                    "items": [
                        {"id": "gpt-4o", "provider": "openai"},
                        {"id": "claude-3", "provider": "anthropic"},
                    ]
                }
            )
        )

        resolver = FileRegistryResolver(registry_dir)
        result = resolver.resolve("models", filter_expr="provider != 'openai'")

        assert result == ["claude-3"]

    def test_filter_by_version(self, tmp_path: Path) -> None:
        """Test filtering by version parameter."""
        registry_dir = tmp_path / "registries"
        registry_dir.mkdir()

        registry_file = registry_dir / "models.yaml"
        registry_file.write_text(
            yaml.dump(
                {
                    "items": [
                        {"id": "scorer_v1", "version": "1.0"},
                        {"id": "scorer_v2", "version": "2.0"},
                        {"id": "scorer_v3", "version": "2.0"},
                    ]
                }
            )
        )

        resolver = FileRegistryResolver(registry_dir)
        result = resolver.resolve("models", version="2.0")

        assert result == ["scorer_v2", "scorer_v3"]

    def test_filter_by_comparison(self, tmp_path: Path) -> None:
        """Test filtering by comparison expression."""
        registry_dir = tmp_path / "registries"
        registry_dir.mkdir()

        registry_file = registry_dir / "models.yaml"
        registry_file.write_text(
            yaml.dump(
                {
                    "items": [
                        {"id": "v1", "version": "1.0"},
                        {"id": "v2", "version": "2.0"},
                        {"id": "v3", "version": "3.0"},
                    ]
                }
            )
        )

        resolver = FileRegistryResolver(registry_dir)
        result = resolver.resolve("models", filter_expr="version >= '2.0'")

        assert result == ["v2", "v3"]

    def test_filter_by_in_expression(self, tmp_path: Path) -> None:
        """Test filtering by 'in' expression."""
        registry_dir = tmp_path / "registries"
        registry_dir.mkdir()

        registry_file = registry_dir / "models.yaml"
        registry_file.write_text(
            yaml.dump(
                {
                    "items": [
                        {"id": "gpt-4o", "provider": "openai"},
                        {"id": "claude-3", "provider": "anthropic"},
                        {"id": "llama-3", "provider": "meta"},
                    ]
                }
            )
        )

        resolver = FileRegistryResolver(registry_dir)
        result = resolver.resolve(
            "models", filter_expr="provider in ['openai', 'anthropic']"
        )

        assert set(result) == {"gpt-4o", "claude-3"}

    def test_registry_not_found_raises(self, tmp_path: Path) -> None:
        """Test that missing registry raises ValueError."""
        resolver = FileRegistryResolver(tmp_path)

        with pytest.raises(ValueError, match="not found"):
            resolver.resolve("nonexistent")

    def test_invalid_filter_raises(self, tmp_path: Path) -> None:
        """Test that invalid filter expression raises ValueError."""
        registry_dir = tmp_path / "registries"
        registry_dir.mkdir()

        registry_file = registry_dir / "models.yaml"
        registry_file.write_text(yaml.dump({"items": [{"id": "test"}]}))

        resolver = FileRegistryResolver(registry_dir)

        with pytest.raises(ValueError, match="Unsupported filter"):
            resolver.resolve("models", filter_expr="invalid syntax here")

    def test_extracts_name_fallback(self, tmp_path: Path) -> None:
        """Test that 'name' field is used as fallback when 'id' missing."""
        registry_dir = tmp_path / "registries"
        registry_dir.mkdir()

        registry_file = registry_dir / "prompts.yaml"
        registry_file.write_text(
            yaml.dump(
                {
                    "items": [
                        {"name": "default_prompt"},
                        {"name": "detailed_prompt"},
                    ]
                }
            )
        )

        resolver = FileRegistryResolver(registry_dir)
        result = resolver.resolve("prompts")

        assert result == ["default_prompt", "detailed_prompt"]

    def test_yml_extension_supported(self, tmp_path: Path) -> None:
        """Test that .yml extension is also supported."""
        registry_dir = tmp_path / "registries"
        registry_dir.mkdir()

        registry_file = registry_dir / "models.yml"
        registry_file.write_text(yaml.dump({"items": [{"id": "test"}]}))

        resolver = FileRegistryResolver(registry_dir)
        result = resolver.resolve("models")

        assert result == ["test"]

    def test_items_not_a_list_raises(self, tmp_path: Path) -> None:
        """Test that items being a dict instead of list raises ValueError."""
        registry_dir = tmp_path / "registries"
        registry_dir.mkdir()

        registry_file = registry_dir / "models.yaml"
        registry_file.write_text(yaml.dump({"items": {"id": "test"}}))

        resolver = FileRegistryResolver(registry_dir)

        with pytest.raises(ValueError, match="must have an 'items' list"):
            resolver.resolve("models")

    def test_unsupported_file_format_raises(self, tmp_path: Path) -> None:
        """Test that unsupported file extensions raise ValueError."""
        registry_dir = tmp_path / "registries"
        registry_dir.mkdir()

        # Create a .txt file and manually call _load_registry
        registry_file = registry_dir / "models.txt"
        registry_file.write_text("some content")

        resolver = FileRegistryResolver(registry_dir)

        with pytest.raises(ValueError, match="Unsupported registry file format"):
            resolver._load_registry(registry_file)

    def test_filter_less_than_operator(self, tmp_path: Path) -> None:
        """Test filtering with < operator."""
        registry_dir = tmp_path / "registries"
        registry_dir.mkdir()

        registry_file = registry_dir / "models.yaml"
        registry_file.write_text(
            yaml.dump(
                {
                    "items": [
                        {"id": "v1", "version": "1.0"},
                        {"id": "v2", "version": "2.0"},
                        {"id": "v3", "version": "3.0"},
                    ]
                }
            )
        )

        resolver = FileRegistryResolver(registry_dir)
        result = resolver.resolve("models", filter_expr="version < '2.0'")

        assert result == ["v1"]

    def test_filter_greater_than_operator(self, tmp_path: Path) -> None:
        """Test filtering with > operator."""
        registry_dir = tmp_path / "registries"
        registry_dir.mkdir()

        registry_file = registry_dir / "models.yaml"
        registry_file.write_text(
            yaml.dump(
                {
                    "items": [
                        {"id": "v1", "version": "1.0"},
                        {"id": "v2", "version": "2.0"},
                        {"id": "v3", "version": "3.0"},
                    ]
                }
            )
        )

        resolver = FileRegistryResolver(registry_dir)
        result = resolver.resolve("models", filter_expr="version > '2.0'")

        assert result == ["v3"]

    def test_filter_less_than_or_equal_operator(self, tmp_path: Path) -> None:
        """Test filtering with <= operator."""
        registry_dir = tmp_path / "registries"
        registry_dir.mkdir()

        registry_file = registry_dir / "models.yaml"
        registry_file.write_text(
            yaml.dump(
                {
                    "items": [
                        {"id": "v1", "version": "1.0"},
                        {"id": "v2", "version": "2.0"},
                        {"id": "v3", "version": "3.0"},
                    ]
                }
            )
        )

        resolver = FileRegistryResolver(registry_dir)
        result = resolver.resolve("models", filter_expr="version <= '2.0'")

        assert result == ["v1", "v2"]

    def test_comparison_skips_items_with_missing_field(self, tmp_path: Path) -> None:
        """Test that comparison filters skip items where field is None."""
        registry_dir = tmp_path / "registries"
        registry_dir.mkdir()

        registry_file = registry_dir / "models.yaml"
        registry_file.write_text(
            yaml.dump(
                {
                    "items": [
                        {"id": "v1", "version": "1.0"},
                        {"id": "v2"},  # Missing version field
                        {"id": "v3", "version": "3.0"},
                    ]
                }
            )
        )

        resolver = FileRegistryResolver(registry_dir)
        result = resolver.resolve("models", filter_expr="version >= '2.0'")

        # v2 should be skipped because it has no version
        assert result == ["v3"]

    def test_extract_values_fallback_to_whole_item(self, tmp_path: Path) -> None:
        """Test that items without 'id' or 'name' return the whole dict."""
        registry_dir = tmp_path / "registries"
        registry_dir.mkdir()

        registry_file = registry_dir / "configs.yaml"
        registry_file.write_text(
            yaml.dump(
                {
                    "items": [
                        {"key": "value1", "setting": "a"},
                        {"key": "value2", "setting": "b"},
                    ]
                }
            )
        )

        resolver = FileRegistryResolver(registry_dir)
        result = resolver.resolve("configs")

        # Without id or name, should return the whole dict
        assert result == [
            {"key": "value1", "setting": "a"},
            {"key": "value2", "setting": "b"},
        ]

    def test_extract_values_non_dict_items(self, tmp_path: Path) -> None:
        """Test that non-dict items are returned as-is."""
        registry_dir = tmp_path / "registries"
        registry_dir.mkdir()

        # Items can be simple strings/values
        registry_file = registry_dir / "simple.yaml"
        registry_file.write_text(
            yaml.dump(
                {
                    "items": [
                        "option_a",
                        "option_b",
                        "option_c",
                    ]
                }
            )
        )

        resolver = FileRegistryResolver(registry_dir)
        result = resolver.resolve("simple")

        assert result == ["option_a", "option_b", "option_c"]

    def test_path_traversal_prevention(self, tmp_path: Path) -> None:
        """Test that path traversal attempts are rejected."""
        resolver = FileRegistryResolver(tmp_path)

        # Test various path traversal attempts
        with pytest.raises(ValueError, match="must not contain path separators"):
            resolver.resolve("../etc/passwd")

        with pytest.raises(ValueError, match="must not contain path separators"):
            resolver.resolve("foo/bar")

        with pytest.raises(ValueError, match="must not contain path separators"):
            resolver.resolve("foo\\bar")

        with pytest.raises(ValueError, match="must not contain"):
            resolver.resolve("..models")


class TestDictRegistryResolver:
    """Tests for DictRegistryResolver."""

    def test_resolve_basic(self) -> None:
        """Test basic resolution from dict."""
        resolver = DictRegistryResolver(
            {
                "models": [
                    {"id": "gpt-4o"},
                    {"id": "claude-3"},
                ],
            }
        )

        result = resolver.resolve("models")
        assert result == ["gpt-4o", "claude-3"]

    def test_resolve_with_filter(self) -> None:
        """Test resolution with filter expression."""
        resolver = DictRegistryResolver(
            {
                "models": [
                    {"id": "gpt-4o", "provider": "openai"},
                    {"id": "claude-3", "provider": "anthropic"},
                ],
            }
        )

        result = resolver.resolve("models", filter_expr="provider == 'openai'")
        assert result == ["gpt-4o"]

    def test_resolve_with_version(self) -> None:
        """Test resolution with version filter."""
        resolver = DictRegistryResolver(
            {
                "scorers": [
                    {"id": "v1", "version": "1.0"},
                    {"id": "v2", "version": "2.0"},
                ],
            }
        )

        result = resolver.resolve("scorers", version="2.0")
        assert result == ["v2"]

    def test_registry_not_found_raises(self) -> None:
        """Test that missing registry raises ValueError."""
        resolver = DictRegistryResolver({"models": []})

        with pytest.raises(ValueError, match="not found"):
            resolver.resolve("nonexistent")

    def test_available_registries_in_error(self) -> None:
        """Test that error message shows available registries."""
        resolver = DictRegistryResolver(
            {
                "models": [],
                "scorers": [],
            }
        )

        with pytest.raises(ValueError, match="models.*scorers"):
            resolver.resolve("nonexistent")

    def test_resolve_with_name_fallback(self) -> None:
        """Test that 'name' is used when 'id' is missing."""
        resolver = DictRegistryResolver(
            {
                "prompts": [
                    {"name": "prompt_a"},
                    {"name": "prompt_b"},
                ],
            }
        )

        result = resolver.resolve("prompts")
        assert result == ["prompt_a", "prompt_b"]

    def test_resolve_whole_item_when_no_id_or_name(self) -> None:
        """Test that whole item is returned when no 'id' or 'name'."""
        resolver = DictRegistryResolver(
            {
                "configs": [
                    {"key": "value1"},
                    {"key": "value2"},
                ],
            }
        )

        result = resolver.resolve("configs")
        assert result == [{"key": "value1"}, {"key": "value2"}]

    def test_resolve_non_dict_items(self) -> None:
        """Test that non-dict items are returned as-is."""
        resolver = DictRegistryResolver(
            {
                "simple": ["a", "b", "c"],
            }
        )

        result = resolver.resolve("simple")
        assert result == ["a", "b", "c"]

    def test_resolve_empty_registry(self) -> None:
        """Test that empty registry returns empty list."""
        resolver = DictRegistryResolver({"models": []})

        result = resolver.resolve("models")
        assert result == []

    def test_filter_with_comparison_operators(self) -> None:
        """Test that comparison operators work in DictRegistryResolver."""
        resolver = DictRegistryResolver(
            {
                "versions": [
                    {"id": "v1", "rank": "1"},
                    {"id": "v2", "rank": "2"},
                    {"id": "v3", "rank": "3"},
                ],
            }
        )

        result = resolver.resolve("versions", filter_expr="rank > '1'")
        assert result == ["v2", "v3"]

        result = resolver.resolve("versions", filter_expr="rank < '3'")
        assert result == ["v1", "v2"]
