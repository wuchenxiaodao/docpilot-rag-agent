"""Tests for online ingestion (chunking + idempotent write).

No GPU / real Chroma needed: the store is mocked; only load_and_chunk's
real-PDF test touches an actual parser (PyPDFLoader over the repo's handbook).
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from agents.ingestion import ingest_file, load_and_chunk

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HANDBOOK_PDF = os.path.join(REPO_ROOT, "data", "AcmeTech_Employee_Handbook.pdf")


def test_load_and_chunk_real_pdf():
    """Real parse + split of the repo handbook; anchors metadata shape."""
    with open(HANDBOOK_PDF, "rb") as f:
        content = f.read()
    chunks = load_and_chunk("AcmeTech_Employee_Handbook.pdf", content)

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.metadata["source"] == "AcmeTech_Employee_Handbook.pdf"
        assert chunk.metadata["chunk_id"].startswith("AcmeTech_Employee_Handbook-p")
        assert "page" in chunk.metadata
        assert isinstance(chunk.metadata["page"], int)
        assert chunk.page_content.strip()


def test_load_and_chunk_rejects_unsupported_extension():
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_and_chunk("notes.txt", b"hello")


def test_load_and_chunk_rejects_empty_content():
    with pytest.raises(ValueError, match="Empty file"):
        load_and_chunk("a.pdf", b"")


def test_ingest_replaces_same_source():
    """Same-name upload deletes old chunks first (idempotent re-ingest),
    including path-style sources from the offline builders (./data\\a.pdf)."""
    fake_docs = [
        Document(page_content="x", metadata={"source": "a.pdf", "chunk_id": "a-p1-c1", "page": 1})
    ]
    store = MagicMock()
    store.get.return_value = {
        "ids": ["old1", "old2"],
        "metadatas": [{"source": "./data\\a.pdf"}, {"source": "a.pdf"}],
    }

    with patch("agents.ingestion.load_and_chunk", return_value=fake_docs):
        result = ingest_file("a.pdf", b"whatever", store=store)

    store.delete.assert_called_once_with(ids=["old1", "old2"])
    store.add_documents.assert_called_once_with(fake_docs)
    assert result.chunks_added == 1
    assert result.chunks_deleted == 2


def test_ingest_first_upload_deletes_nothing():
    fake_docs = [Document(page_content="x", metadata={"source": "b.pdf", "chunk_id": "b-p0-c1"})]
    store = MagicMock()
    store.get.return_value = {"ids": ["other"], "metadatas": [{"source": "unrelated.pdf"}]}

    with patch("agents.ingestion.load_and_chunk", return_value=fake_docs):
        result = ingest_file("b.pdf", b"whatever", store=store)

    store.delete.assert_not_called()
    assert result.chunks_deleted == 0
    assert result.chunks_added == 1


def test_ingest_size_limit(monkeypatch):
    import agents.ingestion as ingestion_module

    monkeypatch.setattr(ingestion_module, "MAX_UPLOAD_BYTES", 4)
    with pytest.raises(ValueError, match="too large"):
        ingest_file("a.pdf", b"12345", store=MagicMock())
