"""Unit tests for vector store integration plugins."""

import pytest

from traigent.config.types import TraigentConfig
from traigent.integrations.vector_stores.chromadb_plugin import ChromaDBPlugin
from traigent.integrations.vector_stores.pinecone_plugin import PineconePlugin
from traigent.integrations.vector_stores.weaviate_plugin import WeaviatePlugin


@pytest.fixture
def chroma_plugin():
    return ChromaDBPlugin()


@pytest.fixture
def pinecone_plugin():
    return PineconePlugin()


@pytest.fixture
def weaviate_plugin():
    return WeaviatePlugin()


def test_chroma_metadata(chroma_plugin):
    assert chroma_plugin.metadata.name == "chromadb"
    assert "chromadb" in chroma_plugin.metadata.supported_packages


def test_chroma_overrides(chroma_plugin):
    config = TraigentConfig(
        custom_params={"k": 5, "filter": {"metadata_field": "value"}}
    )
    kwargs = {}

    overridden = chroma_plugin.apply_overrides(kwargs, config)

    assert overridden["n_results"] == 5
    assert overridden["where"] == {"metadata_field": "value"}


def test_pinecone_metadata(pinecone_plugin):
    assert pinecone_plugin.metadata.name == "pinecone"
    assert "pinecone" in pinecone_plugin.metadata.supported_packages


def test_pinecone_overrides(pinecone_plugin):
    config = TraigentConfig(
        custom_params={"k": 10, "filter": {"genre": "action"}, "namespace": "ns1"}
    )
    kwargs = {}

    overridden = pinecone_plugin.apply_overrides(kwargs, config)

    assert overridden["top_k"] == 10
    assert overridden["filter"] == {"genre": "action"}
    assert overridden["namespace"] == "ns1"  # Namespace is mapped for Pinecone


def test_weaviate_metadata(weaviate_plugin):
    assert weaviate_plugin.metadata.name == "weaviate"
    assert "weaviate-client" in weaviate_plugin.metadata.supported_packages


def test_weaviate_overrides(weaviate_plugin):
    config = TraigentConfig(custom_params={"k": 20, "score_threshold": 0.9})
    kwargs = {}

    overridden = weaviate_plugin.apply_overrides(kwargs, config)

    assert overridden["limit"] == 20
    assert overridden["certainty"] == 0.9
