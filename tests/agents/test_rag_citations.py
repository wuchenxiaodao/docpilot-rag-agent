"""Tests for structured citations: tool JSON line + collect_citations node."""

import json
from unittest.mock import patch

import pytest
from langchain_core.messages import ToolMessage

import agents.rag_assistant as rag_module
from agents.rag_assistant import collect_citations
from agents.tools import SearchResult, _database_search_for_tool


def _search_result(citations: list[dict]) -> SearchResult:
    return {
        "status": "answered",
        "context": "Document 1\nSource: a.pdf\nContent: hello",
        "citations": citations,
        "reason": None,
    }


def test_tool_output_appends_citations_json():
    """_database_search_for_tool 的文本末尾带一行可解析的 CITATIONS_JSON。"""
    citations = [{"source": "a.pdf", "page": 2, "chunk_id": "a-p2-c1", "excerpt": "hello"}]
    with patch(
        "agents.tools.database_search_func", return_value=_search_result(citations)
    ):
        out = _database_search_for_tool("q")
    assert out.startswith("Document 1")
    tail = out.rsplit("CITATIONS_JSON: ", 1)[1]
    assert json.loads(tail) == citations


def test_tool_output_no_answer_has_no_citations():
    with patch(
        "agents.tools.database_search_func",
        return_value={
            "status": "no_answer",
            "context": "",
            "citations": [],
            "reason": "no_documents_retrieved",
        },
    ):
        out = _database_search_for_tool("q")
    assert out == "NO_RELEVANT_DOCUMENTS_FOUND"


@pytest.mark.asyncio
async def test_collect_citations_parses_and_dispatches():
    citations = [{"source": "a.pdf", "page": 2, "chunk_id": "a-p2-c1", "excerpt": "x"}]
    state = {
        "messages": [
            ToolMessage(
                content=f"context...\n\nCITATIONS_JSON: {json.dumps(citations)}",
                tool_call_id="call_1",
            )
        ]
    }
    dispatched: list = []
    result = await collect_citations(state, writer=dispatched.append)
    assert result["citations"] == citations
    assert result["messages"] == []
    assert len(dispatched) == 1
    sent = dispatched[0]
    assert sent.role == "custom"
    assert sent.content[0] == {"docpilot_citations": citations}


@pytest.mark.asyncio
async def test_collect_citations_no_tool_message():
    state = {"messages": []}
    dispatched: list = []
    result = await collect_citations(state, writer=dispatched.append)
    assert result["citations"] == []
    assert dispatched == []


@pytest.mark.asyncio
async def test_collect_citations_bad_json_falls_back_empty():
    state = {
        "messages": [
            ToolMessage(content="CITATIONS_JSON: [broken", tool_call_id="call_1"),
        ]
    }
    result = await collect_citations(state, writer=lambda _m: None)
    assert result["citations"] == []


def test_citations_regex_matches_multiline_context():
    """正则须在多行 context 中锚定最后一个 CITATIONS_JSON 行。"""
    m = rag_module._CITATIONS_RE.search("line1\nline2\n\nCITATIONS_JSON: [{\"k\": 1}]")
    assert m is not None
    assert json.loads(m.group(1)) == [{"k": 1}]
