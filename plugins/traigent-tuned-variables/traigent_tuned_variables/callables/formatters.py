"""Built-in context formatting functions.

Provides common formatting strategies for RAG contexts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from traigent.api.parameter_ranges import Choices

from .callable import TunedCallable


class ContextFormatters:
    """Built-in context formatting functions for RAG.

    Example:
        ```python
        @traigent.optimize(
            context_format=ContextFormatters.as_choices(),
        )
        def rag_agent(query: str) -> str:
            config = traigent.get_config()
            docs = retrieve_documents(query)
            formatted = ContextFormatters.invoke(config["context_format"], docs)
            ...
        ```
    """

    @staticmethod
    def bullet(docs: list) -> str:
        """Format documents as bullet list.

        Args:
            docs: List of documents with page_content attribute

        Returns:
            Bullet-formatted string
        """
        lines = []
        for doc in docs:
            content = _get_content(doc)
            lines.append(f"• {content}")
        return "\n".join(lines)

    @staticmethod
    def numbered(docs: list) -> str:
        """Format documents as numbered list.

        Args:
            docs: List of documents with page_content attribute

        Returns:
            Numbered list string
        """
        lines = []
        for i, doc in enumerate(docs, 1):
            content = _get_content(doc)
            lines.append(f"{i}. {content}")
        return "\n".join(lines)

    @staticmethod
    def xml(docs: list) -> str:
        """Format documents with XML tags.

        Modern LLMs often perform better with XML-tagged content.

        Args:
            docs: List of documents with page_content attribute

        Returns:
            XML-formatted string
        """
        lines = []
        for i, doc in enumerate(docs, 1):
            content = _get_content(doc)
            lines.append(f'<document id="{i}">{content}</document>')
        return "\n".join(lines)

    @staticmethod
    def markdown(docs: list) -> str:
        """Format documents as markdown sections.

        Args:
            docs: List of documents with page_content attribute

        Returns:
            Markdown-formatted string
        """
        lines = []
        for i, doc in enumerate(docs, 1):
            content = _get_content(doc)
            # Try to get source metadata if available
            source = _get_metadata(doc, "source", f"Document {i}")
            lines.append(f"### {source}\n{content}")
        return "\n\n".join(lines)

    @staticmethod
    def json_array(docs: list) -> str:
        """Format documents as JSON array.

        Args:
            docs: List of documents with page_content attribute

        Returns:
            JSON array string
        """
        import json

        items = []
        for i, doc in enumerate(docs, 1):
            content = _get_content(doc)
            items.append({"id": i, "content": content})
        return json.dumps(items, indent=2)

    @staticmethod
    def plain(docs: list) -> str:
        """Format documents as plain text, separated by newlines.

        Args:
            docs: List of documents with page_content attribute

        Returns:
            Plain text string
        """
        lines = []
        for doc in docs:
            content = _get_content(doc)
            lines.append(content)
        return "\n\n---\n\n".join(lines)

    @classmethod
    def as_tuned_callable(cls) -> TunedCallable:
        """Get as TunedCallable.

        Returns:
            TunedCallable configured for formatter selection
        """
        return TunedCallable(
            name="context_format",
            callables={
                "bullet": cls.bullet,
                "numbered": cls.numbered,
                "xml": cls.xml,
                "markdown": cls.markdown,
                "json": cls.json_array,
                "plain": cls.plain,
            },
            description="Context formatting strategies for RAG",
        )

    @classmethod
    def as_choices(cls) -> Choices:
        """Get as Choices for configuration space.

        Returns:
            Choices instance with formatter options
        """
        from traigent.api.parameter_ranges import Choices

        return Choices(
            ["bullet", "numbered", "xml", "markdown", "json", "plain"],
            default="bullet",
            name="context_format",
        )

    @classmethod
    def invoke(cls, name: str, docs: list) -> str:
        """Invoke a formatter by name.

        Args:
            name: Formatter name
            docs: List of documents to format

        Returns:
            Formatted string
        """
        return cls.as_tuned_callable().invoke(name, docs)


def _get_content(doc: Any) -> str:
    """Extract content from a document object."""
    if hasattr(doc, "page_content"):
        return doc.page_content
    if isinstance(doc, dict):
        return doc.get("content", doc.get("text", str(doc)))
    return str(doc)


def _get_metadata(doc: Any, key: str, default: str = "") -> str:
    """Extract metadata from a document object."""
    if hasattr(doc, "metadata"):
        return doc.metadata.get(key, default)
    if isinstance(doc, dict):
        return doc.get("metadata", {}).get(key, default)
    return default
