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

    def test_filter_by_comparison_uses_semantic_versions(self, tmp_path: Path) -> None:
        """Comparison filters should not use raw lexicographic ordering."""
        registry_dir = tmp_path / "registries"
        registry_dir.mkdir()

        registry_file = registry_dir / "models.yaml"
        registry_file.write_text(
            yaml.dump(
                {
                    "items": [
                        {"id": "v2", "version": "2.0"},
                        {"id": "v10", "version": "10.0"},
                        {"id": "v11", "version": "11.0"},
                    ]
                }
            )
        )

        resolver = FileRegistryResolver(registry_dir)

        assert resolver.resolve("models", filter_expr="version >= '2.0'") == [
            "v2",
            "v10",
            "v11",
        ]
        assert resolver.resolve("models", filter_expr="version < '2.0'") == []

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

    def test_filter_with_semantic_versions(self) -> None:
        """Semantic version comparison should work for multi-digit versions."""
        resolver = DictRegistryResolver(
            {
                "versions": [
                    {"id": "v2", "version": "2.0"},
                    {"id": "v10", "version": "10.0"},
                    {"id": "v11", "version": "11.0"},
                ],
            }
        )

        assert resolver.resolve("versions", filter_expr="version >= '2.0'") == [
            "v2",
            "v10",
            "v11",
        ]
        assert resolver.resolve("versions", filter_expr="version < '2.0'") == []


class TestBooleanFilterLogic:
    """Tests for T-2: Boolean logic (AND/OR) in registry filters."""

    def test_and_operator(self) -> None:
        """Test AND operator: both conditions must match."""
        resolver = DictRegistryResolver(
            {
                "models": [
                    {"id": "gpt-4o", "provider": "openai", "tier": "quality"},
                    {"id": "gpt-4o-mini", "provider": "openai", "tier": "fast"},
                    {"id": "claude-3-opus", "provider": "anthropic", "tier": "quality"},
                    {"id": "claude-3-haiku", "provider": "anthropic", "tier": "fast"},
                ],
            }
        )

        result = resolver.resolve(
            "models", filter_expr="provider == 'openai' AND tier == 'quality'"
        )
        assert result == ["gpt-4o"]

    def test_or_operator(self) -> None:
        """Test OR operator: either condition can match."""
        resolver = DictRegistryResolver(
            {
                "models": [
                    {"id": "gpt-4o", "provider": "openai"},
                    {"id": "claude-3", "provider": "anthropic"},
                    {"id": "llama-3", "provider": "meta"},
                ],
            }
        )

        result = resolver.resolve(
            "models", filter_expr="provider == 'openai' OR provider == 'anthropic'"
        )
        assert set(result) == {"gpt-4o", "claude-3"}

    def test_and_operator_case_insensitive(self) -> None:
        """Test that AND is case-insensitive."""
        resolver = DictRegistryResolver(
            {
                "models": [
                    {"id": "m1", "a": "x", "b": "y"},
                    {"id": "m2", "a": "x", "b": "z"},
                ],
            }
        )

        result = resolver.resolve("models", filter_expr="a == 'x' and b == 'y'")
        assert result == ["m1"]

        result = resolver.resolve("models", filter_expr="a == 'x' And b == 'y'")
        assert result == ["m1"]

    def test_or_operator_case_insensitive(self) -> None:
        """Test that OR is case-insensitive."""
        resolver = DictRegistryResolver(
            {
                "models": [
                    {"id": "m1", "type": "a"},
                    {"id": "m2", "type": "b"},
                    {"id": "m3", "type": "c"},
                ],
            }
        )

        result = resolver.resolve("models", filter_expr="type == 'a' or type == 'b'")
        assert set(result) == {"m1", "m2"}

    def test_multiple_and_conditions(self) -> None:
        """Test multiple AND conditions chained."""
        resolver = DictRegistryResolver(
            {
                "models": [
                    {"id": "m1", "a": "x", "b": "y", "c": "z"},
                    {"id": "m2", "a": "x", "b": "y", "c": "w"},
                    {"id": "m3", "a": "x", "b": "n", "c": "z"},
                ],
            }
        )

        result = resolver.resolve(
            "models", filter_expr="a == 'x' AND b == 'y' AND c == 'z'"
        )
        assert result == ["m1"]

    def test_multiple_or_conditions(self) -> None:
        """Test multiple OR conditions chained."""
        resolver = DictRegistryResolver(
            {
                "models": [
                    {"id": "m1", "type": "a"},
                    {"id": "m2", "type": "b"},
                    {"id": "m3", "type": "c"},
                    {"id": "m4", "type": "d"},
                ],
            }
        )

        result = resolver.resolve(
            "models", filter_expr="type == 'a' OR type == 'b' OR type == 'c'"
        )
        assert set(result) == {"m1", "m2", "m3"}

    def test_and_with_comparison_operators(self) -> None:
        """Test AND with comparison operators."""
        resolver = DictRegistryResolver(
            {
                "models": [
                    {"id": "v1", "version": "1.0", "tier": "fast"},
                    {"id": "v2", "version": "2.0", "tier": "fast"},
                    {"id": "v3", "version": "2.0", "tier": "quality"},
                ],
            }
        )

        result = resolver.resolve(
            "models", filter_expr="version >= '2.0' AND tier == 'fast'"
        )
        assert result == ["v2"]

    def test_or_with_in_operator(self) -> None:
        """Test OR with 'in' operator in one clause."""
        resolver = DictRegistryResolver(
            {
                "models": [
                    {"id": "m1", "provider": "openai", "type": "chat"},
                    {"id": "m2", "provider": "anthropic", "type": "chat"},
                    {"id": "m3", "provider": "openai", "type": "embedding"},
                ],
            }
        )

        result = resolver.resolve(
            "models", filter_expr="provider in ['anthropic'] OR type == 'embedding'"
        )
        assert set(result) == {"m2", "m3"}

    def test_parentheses_simple(self) -> None:
        """Test parentheses wrapping a simple expression."""
        resolver = DictRegistryResolver(
            {
                "models": [
                    {"id": "m1", "type": "a"},
                    {"id": "m2", "type": "b"},
                ],
            }
        )

        result = resolver.resolve("models", filter_expr="(type == 'a')")
        assert result == ["m1"]

    def test_and_has_precedence_over_or(self) -> None:
        """Test that AND has higher precedence than OR.

        Expression: A OR B AND C should be parsed as A OR (B AND C)
        """
        resolver = DictRegistryResolver(
            {
                "models": [
                    {"id": "m1", "a": "x", "b": "y", "c": "z"},  # a=x matches first OR
                    {"id": "m2", "a": "n", "b": "y", "c": "z"},  # b=y AND c=z matches
                    {"id": "m3", "a": "n", "b": "y", "c": "w"},  # only b=y, no match
                    {"id": "m4", "a": "n", "b": "n", "c": "n"},  # no match
                ],
            }
        )

        # a == 'x' OR (b == 'y' AND c == 'z')
        result = resolver.resolve(
            "models", filter_expr="a == 'x' OR b == 'y' AND c == 'z'"
        )
        assert set(result) == {"m1", "m2"}

    def test_empty_result_with_and(self) -> None:
        """Test AND that results in no matches."""
        resolver = DictRegistryResolver(
            {
                "models": [
                    {"id": "m1", "a": "x"},
                    {"id": "m2", "a": "y"},
                ],
            }
        )

        result = resolver.resolve(
            "models", filter_expr="a == 'x' AND a == 'y'"  # Impossible
        )
        assert result == []

    def test_preserves_order(self) -> None:
        """Test that filter results preserve original item order."""
        resolver = DictRegistryResolver(
            {
                "models": [
                    {"id": "m1", "type": "a"},
                    {"id": "m2", "type": "b"},
                    {"id": "m3", "type": "a"},
                    {"id": "m4", "type": "b"},
                ],
            }
        )

        result = resolver.resolve("models", filter_expr="type == 'a' OR type == 'b'")
        assert result == ["m1", "m2", "m3", "m4"]

    def test_parentheses_with_or_inside(self) -> None:
        """Test parentheses containing OR expression.

        This was a bug identified by Codex review - expressions like
        'A AND (B OR C)' would incorrectly split on the inner OR.
        """
        resolver = DictRegistryResolver(
            {
                "models": [
                    {"id": "m1", "a": "x", "b": "y"},  # a=x AND b=y -> match
                    {"id": "m2", "a": "x", "b": "z"},  # a=x AND b=z -> match
                    {"id": "m3", "a": "x", "b": "w"},  # a=x but b!=y,z -> no match
                    {"id": "m4", "a": "n", "b": "y"},  # a!=x -> no match
                ],
            }
        )

        # a == 'x' AND (b == 'y' OR b == 'z')
        result = resolver.resolve(
            "models", filter_expr="a == 'x' AND (b == 'y' OR b == 'z')"
        )
        assert set(result) == {"m1", "m2"}

    def test_parentheses_with_and_inside(self) -> None:
        """Test parentheses containing AND expression."""
        resolver = DictRegistryResolver(
            {
                "models": [
                    {
                        "id": "m1",
                        "a": "x",
                        "b": "y",
                        "c": "z",
                    },  # (a=x AND b=y) -> match
                    {"id": "m2", "a": "n", "c": "z"},  # c=z -> match
                    {"id": "m3", "a": "x", "b": "n"},  # a=x but b!=y -> no match
                    {"id": "m4", "a": "n", "c": "w"},  # no match
                ],
            }
        )

        # (a == 'x' AND b == 'y') OR c == 'z'
        result = resolver.resolve(
            "models", filter_expr="(a == 'x' AND b == 'y') OR c == 'z'"
        )
        assert set(result) == {"m1", "m2"}

    def test_nested_parentheses(self) -> None:
        """Test nested parentheses expressions."""
        resolver = DictRegistryResolver(
            {
                "models": [
                    {"id": "m1", "a": "x", "b": "y", "c": "z"},
                    {"id": "m2", "a": "x", "b": "n", "c": "z"},
                    {"id": "m3", "a": "n", "b": "y", "c": "z"},
                ],
            }
        )

        # a == 'x' AND (b == 'y' OR (c == 'z'))
        result = resolver.resolve(
            "models", filter_expr="a == 'x' AND (b == 'y' OR (c == 'z'))"
        )
        # m1: a=x, b=y -> match
        # m2: a=x, c=z -> match
        assert set(result) == {"m1", "m2"}

    def test_complex_mixed_expression(self) -> None:
        """Test complex expression with multiple operators and parentheses."""
        resolver = DictRegistryResolver(
            {
                "models": [
                    {"id": "m1", "provider": "openai", "tier": "fast", "version": "1"},
                    {
                        "id": "m2",
                        "provider": "openai",
                        "tier": "quality",
                        "version": "2",
                    },
                    {
                        "id": "m3",
                        "provider": "anthropic",
                        "tier": "fast",
                        "version": "2",
                    },
                    {
                        "id": "m4",
                        "provider": "anthropic",
                        "tier": "quality",
                        "version": "1",
                    },
                ],
            }
        )

        # provider == 'openai' AND (tier == 'fast' OR version == '2')
        result = resolver.resolve(
            "models",
            filter_expr="provider == 'openai' AND (tier == 'fast' OR version == '2')",
        )
        # m1: openai, fast -> match
        # m2: openai, version=2 -> match
        assert set(result) == {"m1", "m2"}

    def test_quoted_values_with_parentheses(self) -> None:
        """Test that parentheses inside quoted values don't break parsing."""
        resolver = DictRegistryResolver(
            {
                "models": [
                    {"id": "gpt-4 (preview)", "tier": "fast"},
                    {"id": "gpt-4 (stable)", "tier": "quality"},
                    {"id": "claude-3", "tier": "fast"},
                ],
            }
        )

        # Value contains parentheses - should not confuse depth tracking
        result = resolver.resolve(
            "models", filter_expr="id == 'gpt-4 (preview)' OR tier == 'quality'"
        )
        assert set(result) == {"gpt-4 (preview)", "gpt-4 (stable)"}

    def test_quoted_values_with_nested_parens(self) -> None:
        """Test values with multiple/nested parentheses in quotes."""
        resolver = DictRegistryResolver(
            {
                "functions": [
                    {"id": "func(a, b)", "type": "binary"},
                    {"id": "func((a))", "type": "nested"},
                    {"id": "simple", "type": "none"},
                ],
            }
        )

        # Complex parentheses in value
        result = resolver.resolve(
            "functions", filter_expr="id == 'func((a))' OR type == 'binary'"
        )
        assert set(result) == {"func(a, b)", "func((a))"}

    def test_mixed_quotes_and_parentheses(self) -> None:
        """Test expression with both quote styles and parentheses."""
        resolver = DictRegistryResolver(
            {
                "items": [
                    {"id": "a", "name": "Item (A)", "desc": 'Has "quotes"'},
                    {"id": "b", "name": "Item (B)", "desc": "Normal"},
                    {"id": "c", "name": "Item C", "desc": "Normal"},
                ],
            }
        )

        # Double quotes in filter, parentheses in value
        result = resolver.resolve(
            "items", filter_expr='name == "Item (A)" OR id == "c"'
        )
        assert set(result) == {"a", "c"}

    def test_and_or_keywords_in_quoted_values(self) -> None:
        """Test that AND/OR keywords inside quoted values are not treated as operators."""
        resolver = DictRegistryResolver(
            {
                "items": [
                    {"id": "1", "name": "Black AND White"},
                    {"id": "2", "name": "Red OR Blue"},
                    {"id": "3", "name": "Simple"},
                ],
            }
        )

        # AND in the value should not split
        result = resolver.resolve("items", filter_expr="name == 'Black AND White'")
        assert result == ["1"]

        # OR in the value should not split
        result = resolver.resolve("items", filter_expr="name == 'Red OR Blue'")
        assert result == ["2"]

    def test_outer_parens_with_quoted_parens_inside(self) -> None:
        """Test that outer parentheses containing quoted parens work correctly.

        This tests _strip_outer_parens() behavior when quoted values contain parens.
        The expression (id == 'val(ue)') should have outer parens stripped,
        not be confused by the parens inside the quoted value.
        """
        resolver = DictRegistryResolver(
            {
                "items": [
                    {"id": "val(ue)", "type": "a"},
                    {"id": "other", "type": "b"},
                ],
            }
        )

        # Expression with outer parens wrapping a filter with quoted parens
        result = resolver.resolve("items", filter_expr="(id == 'val(ue)')")
        assert result == ["val(ue)"]

        # Combined with OR
        result = resolver.resolve(
            "items", filter_expr="(id == 'val(ue)' OR type == 'b')"
        )
        assert set(result) == {"val(ue)", "other"}
