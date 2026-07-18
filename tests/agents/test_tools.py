"""Tests for DocPilot retrieval tools (format_contexts, path resolution)."""
import importlib.util
import os
import sys
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
    def test_default_path(self):
        path = _get_chroma_db_path()
        assert path == "./chroma_db_qwen3_test"

    def test_env_var_override(self):
        with patch.dict(os.environ, {"CHROMA_DB_PATH": "/custom/db"}):
            path = _get_chroma_db_path()
            assert path == "/custom/db"