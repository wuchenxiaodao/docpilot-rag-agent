"""Tests for DocPilot retrieval tools (format_contexts, path resolution)."""
import importlib.util
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Bypass src/agents/__init__.py (which triggers ollama import chain via agent loading)
# by importing the tools module directly from its file path
_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, _src_dir)

spec = importlib.util.spec_from_file_location(
    "tools_module",
    os.path.join(_src_dir, "agents", "tools.py"),
)
tools_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tools_module)

format_contexts = tools_module.format_contexts
_get_embedding_model_path = tools_module._get_embedding_model_path
_get_chroma_db_path = tools_module._get_chroma_db_path
database_search_func = tools_module.database_search_func
build_citations = tools_module.build_citations
_database_search_for_tool = tools_module._database_search_for_tool


def make_doc(page_content: str, source: str = "test.pdf") -> MagicMock:
    doc = MagicMock()
    doc.page_content = page_content
    doc.metadata = {"source": source}
    return doc


def make_doc_missing_source(page_content: str) -> MagicMock:
    doc = MagicMock()
    doc.page_content = page_content
    doc.metadata = {}
    return doc


class TestFormatContexts:
    def test_single_document(self):
        doc = make_doc("Hello world", "report.pdf")
        result = format_contexts([doc])
        assert "Document 1" in result
        assert "Source: report.pdf" in result
        assert "Content: Hello world" in result

    def test_multiple_documents(self):
        docs = [
            make_doc("First doc", "a.pdf"),
            make_doc("Second doc", "b.pdf"),
        ]
        result = format_contexts(docs)
        assert "Document 1" in result
        assert "Document 2" in result
        assert "Source: a.pdf" in result
        assert "Source: b.pdf" in result
        assert "Content: First doc" in result
        assert "Content: Second doc" in result

    def test_unknown_source_when_metadata_missing(self):
        doc = make_doc_missing_source("No source here")
        result = format_contexts([doc])
        assert "Source: Unknown" in result

    def test_empty_document_list(self):
        result = format_contexts([])
        assert result == ""

    def test_duplicate_sources(self):
        docs = [
            make_doc("Same source", "shared.pdf"),
            make_doc("Also same source", "shared.pdf"),
        ]
        result = format_contexts(docs)
        assert result.count("Source: shared.pdf") == 2
        assert "Document 1" in result
        assert "Document 2" in result

    def test_content_with_newlines(self):
        doc = make_doc("Line1\nLine2\nLine3", "test.pdf")
        result = format_contexts([doc])
        assert "Content: Line1\nLine2\nLine3" in result


def make_doc_none_metadata(page_content: str, source: str = "test.pdf") -> MagicMock:
    doc = MagicMock()
    doc.page_content = page_content
    doc.metadata = None
    return doc


class TestFormatContextsMalformedInput:
    @pytest.mark.parametrize(
        "docs,expected_exc",
        [
            pytest.param(None, TypeError, id="input_is_none"),
            pytest.param(
                [make_doc_none_metadata("bad doc")],
                AttributeError,
                id="doc_metadata_is_none",
            ),
        ],
    )
    def test_raises_on_malformed_input(self, docs, expected_exc):
        with pytest.raises(expected_exc):
            format_contexts(docs)


def make_doc_with_metadata(
    page_content: str,
    source: str = "test.pdf",
    page: int = 1,
    title: str = "AcmeTech Employee Handbook",
    section: str = "General",
    pages: str = "1",
    chunk_id: str = "test-p1-c1",
) -> MagicMock:
    doc = MagicMock()
    doc.page_content = page_content
    doc.metadata = {
        "source": source,
        "title": title,
        "section": section,
        "page": page,
        "pages": pages,
        "chunk_id": chunk_id,
    }
    return doc


class TestFormatContextsCitations:
    def test_citations_from_metadata(self):
        docs = [
            make_doc_with_metadata(
                "Mission content", source="handbook.pdf", page=1
            ),
            make_doc_with_metadata(
                "Remote work policy", source="handbook.pdf", page=3
            ),
        ]
        result = format_contexts(docs)
        assert "[handbook.pdf，第 1 页]" in result
        assert "[handbook.pdf，第 3 页]" in result
        assert "**References:**" in result

    def test_citations_dedup_preserves_order(self):
        docs = [
            make_doc_with_metadata(
                "Mission content", source="handbook.pdf", page=1
            ),
            make_doc_with_metadata(
                "More mission", source="handbook.pdf", page=1
            ),
            make_doc_with_metadata(
                "Remote work policy", source="handbook.pdf", page=3
            ),
            make_doc_with_metadata(
                "More remote work", source="handbook.pdf", page=3
            ),
        ]
        result = format_contexts(docs)
        refs = result.split("**References:**\n")[1] if "**References:**" in result else ""
        lines = [l for l in refs.split("\n") if l.strip()]
        # Deduped to 2 entries, first occurrence order: page 1 then page 3
        assert len(lines) == 2
        assert lines[0] == "[handbook.pdf，第 1 页]"
        assert lines[1] == "[handbook.pdf，第 3 页]"

    def test_citations_multiple_sources(self):
        docs = [
            make_doc_with_metadata(
                "Mission", source="handbook.pdf", page=1
            ),
            make_doc_with_metadata(
                "Code of conduct", source="handbook.pdf", page=5
            ),
            make_doc_with_metadata(
                "Benefits", source="policy.pdf", page=10
            ),
        ]
        result = format_contexts(docs)
        assert "[handbook.pdf，第 1 页]" in result
        assert "[handbook.pdf，第 5 页]" in result
        assert "[policy.pdf，第 10 页]" in result

    def test_citations_missing_page_skipped(self):
        doc = MagicMock()
        doc.page_content = "No page metadata"
        doc.metadata = {"source": "handbook.pdf"}
        result = format_contexts([doc])
        # No citations should be added
        assert "**References:**" not in result
        assert "Source: handbook.pdf" in result

    def test_citations_missing_source_skipped(self):
        doc = MagicMock()
        doc.page_content = "No source metadata"
        doc.metadata = {"page": 1}
        result = format_contexts([doc])
        # Source is Unknown in display, but no citation since src is empty
        assert "**References:**" not in result
        assert "Source: Unknown" in result

    def test_citations_empty_document_list(self):
        result = format_contexts([])
        assert result == ""

    def test_citations_no_duplicates_across_same_source_and_page(self):
        docs = [
            make_doc_with_metadata(
                "First chunk", source="report.pdf", page=2
            ),
            make_doc_with_metadata(
                "Second chunk same page", source="report.pdf", page=2
            ),
            make_doc_with_metadata(
                "Third chunk different page", source="report.pdf", page=5
            ),
            make_doc_with_metadata(
                "Fourth chunk same page again", source="report.pdf", page=2
            ),
        ]
        result = format_contexts(docs)
        refs = result.split("**References:**\n")[1] if "**References:**" in result else ""
        lines = [l for l in refs.split("\n") if l.strip()]
        # Only 2 unique (report.pdf, 2) and (report.pdf, 5)
        assert len(lines) == 2
        assert lines[0] == "[report.pdf，第 2 页]"
        assert lines[1] == "[report.pdf，第 5 页]"


class TestEmbeddingModelPath:
    def test_default_path(self):
        path = _get_embedding_model_path()
        expected = os.path.join(os.path.expanduser("~"), "Models", "Qwen3-Embedding-0.6B")
        assert path == expected

    def test_env_var_override(self):
        with (
            patch.dict(os.environ, {"EMBEDDING_MODEL_PATH": "/custom/path"}),
            patch("os.path.isdir", return_value=True),
        ):
            path = _get_embedding_model_path()
            assert path == "/custom/path"

    def test_missing_directory_raises_error(self):
        with patch.dict(os.environ, {"EMBEDDING_MODEL_PATH": "/nonexistent/path/xyz"}):
            with pytest.raises(RuntimeError, match="Embedding model directory not found"):
                _get_embedding_model_path()


class TestChromaDbPath:
    def test_default_path(self, monkeypatch):
        monkeypatch.delenv("CHROMA_DB_PATH", raising=False)
        assert _get_chroma_db_path() == "./chroma_db_qwen3_semantic_chunks"

    def test_env_var_override(self):
        with patch.dict(os.environ, {"CHROMA_DB_PATH": "/custom/db"}):
            path = _get_chroma_db_path()
            assert path == "/custom/db"

    def test_env_var_overrides_default(self, monkeypatch):
        monkeypatch.setenv("CHROMA_DB_PATH", "/tmp/custom_db")
        assert _get_chroma_db_path() == "/tmp/custom_db"


class _FakeRetriever:
    """Minimal retriever stand-in: invoke() returns prebuilt docs. No model/DB."""

    def __init__(self, docs):
        self._docs = docs

    def invoke(self, query):
        return self._docs


def make_contract_doc(page_content: str, metadata: dict) -> SimpleNamespace:
    """Fake document with only the two attributes the contract touches."""
    return SimpleNamespace(page_content=page_content, metadata=metadata)


def _patch_retriever(monkeypatch, docs):
    """Swap load_chroma_db in the tools module for a fake returning `docs`."""
    monkeypatch.setattr(tools_module, "load_chroma_db", lambda: _FakeRetriever(docs))


class TestDatabaseSearchContract:
    def test_no_answer_when_no_documents(self, monkeypatch):
        _patch_retriever(monkeypatch, [])
        result = database_search_func("anything")
        assert result["status"] == "no_answer"
        assert result["citations"] == []
        assert result["reason"] == "no_documents_retrieved"
        assert result["context"] == ""

    def test_answered_keeps_citation_metadata(self, monkeypatch):
        docs = [
            make_contract_doc(
                "Mission content",
                {"source": "data/handbook.pdf", "page": 1, "chunk_id": "handbook-p1-c2"},
            ),
            make_contract_doc(
                "Remote work policy",
                {"source": "data/handbook.pdf", "page": 3, "chunk_id": "handbook-p3-c9"},
            ),
        ]
        _patch_retriever(monkeypatch, docs)
        result = database_search_func("mission")
        assert result["status"] == "answered"
        assert result["reason"] is None
        assert len(result["citations"]) == 2
        # Order matches the retriever's order (relevance), chunk_id preserved.
        assert result["citations"][0] == {
            "source": "handbook.pdf",
            "page": 1,
            "chunk_id": "handbook-p1-c2",
        }
        assert result["citations"][1] == {
            "source": "handbook.pdf",
            "page": 3,
            "chunk_id": "handbook-p3-c9",
        }

    def test_citations_dedup_preserves_order(self, monkeypatch):
        dup_metadata = {
            "source": "data/handbook.pdf",
            "page": 1,
            "chunk_id": "handbook-p1-c2",
        }
        docs = [
            make_contract_doc("First", dup_metadata),
            make_contract_doc(
                "Second",
                {"source": "data/handbook.pdf", "page": 3, "chunk_id": "handbook-p3-c9"},
            ),
            make_contract_doc("Third is identical to first", dict(dup_metadata)),
        ]
        _patch_retriever(monkeypatch, docs)
        result = database_search_func("x")
        # First and third are identical -> deduped; order: first, second.
        assert len(result["citations"]) == 2
        assert result["citations"][0]["chunk_id"] == "handbook-p1-c2"
        assert result["citations"][1]["chunk_id"] == "handbook-p3-c9"

    def test_tool_wrapper_returns_sentinel_on_no_answer(self, monkeypatch):
        _patch_retriever(monkeypatch, [])
        assert _database_search_for_tool("x") == "NO_RELEVANT_DOCUMENTS_FOUND"

        docs = [
            make_contract_doc(
                "Content",
                {"source": "data/handbook.pdf", "page": 1, "chunk_id": "handbook-p1-c1"},
            )
        ]
        _patch_retriever(monkeypatch, docs)
        assert _database_search_for_tool("x") == database_search_func("x")["context"]