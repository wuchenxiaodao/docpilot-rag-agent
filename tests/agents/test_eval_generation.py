"""Unit tests for eval_generation helper functions (no Ollama needed)."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from scripts.eval_generation import (
    TRULY_OUT,
    check_citation_correctness,
    expected_refusal,
    extract_result,
    _extract_last_answer,
    _extract_tool_output,
    _is_refusal,
    parse_cited_sources,
    parse_judge_json,
)


# --- citation parsing ---


def test_parse_sources_standard_format():
    answer = "Based on the handbook, PTO is 15 days.\n\nSources:\n- AcmeTech_Employee_Handbook.pdf"
    result = parse_cited_sources(answer)
    assert result == ["AcmeTech_Employee_Handbook.pdf"]


def test_parse_sources_multiple():
    answer = "Sources:\n- AcmeTech_Employee_Handbook.pdf\n- Dependency_Upgrades.md\n- Security_Policy.docx"
    result = parse_cited_sources(answer)
    assert result == ["AcmeTech_Employee_Handbook.pdf", "Dependency_Upgrades.md", "Security_Policy.docx"]


def test_parse_sources_deduplicates():
    answer = "Sources:\n- file.pdf\n- file.pdf"
    result = parse_cited_sources(answer)
    assert result == ["file.pdf"]


def test_parse_sources_strips_annotations():
    answer = "Sources:\n- Handbook.pdf (page 2, chunk p1-c2)"
    result = parse_cited_sources(answer)
    # The annotation in parens is stripped by the regex.
    assert result == ["Handbook.pdf"]


def test_parse_sources_none_line():
    answer = "I couldn't find sufficient evidence.\n\nSources: None"
    result = parse_cited_sources(answer)
    # "None" is parsed as a source name (technically correct — no real file cited).
    # In practice the caller checks citation correctness against tool output.
    assert result == ["None"]


def test_parse_sources_no_sources_section():
    answer = "Here is the answer without sources."
    assert parse_cited_sources(answer) == []


# --- citation correctness ---


def test_citation_all_correct():
    tool_output = "Source: AcmeTech_Employee_Handbook.pdf\nContent: PTO is 15 days."
    correct, total = check_citation_correctness(["AcmeTech_Employee_Handbook.pdf"], tool_output)
    assert correct == 1 and total == 1


def test_citation_none_correct():
    tool_output = "Source: other.pdf"
    correct, total = check_citation_correctness(["missing.pdf"], tool_output)
    assert correct == 0 and total == 1


def test_citation_empty():
    correct, total = check_citation_correctness([], "any output")
    assert correct == 0 and total == 0


def test_citation_case_insensitive():
    tool_output = "source: HANDBOOK.PDF"
    correct, total = check_citation_correctness(["handbook.pdf"], tool_output)
    assert correct == 1


# --- refusal detection ---


def test_refusal_standard_phrases():
    assert _is_refusal("I couldn't find sufficient evidence in the knowledge base.", tool_called=True)
    assert _is_refusal("insufficient evidence to answer.", tool_called=True)
    assert _is_refusal("Sources: None", tool_called=True)


def test_refusal_no_tool_called():
    assert _is_refusal("Fallback answer.", tool_called=False)


def test_not_refusal():
    assert not _is_refusal("Based on the document, the answer is 42.", tool_called=True)


# --- message extraction ---


def test_extract_tool_output():
    msgs = [
        HumanMessage(content="Q"),
        AIMessage(content=""),
        ToolMessage(content="Source: a.pdf\nData", tool_call_id="c1"),
    ]
    assert _extract_tool_output(msgs) == "Source: a.pdf\nData"


def test_extract_tool_output_none():
    assert _extract_tool_output([HumanMessage(content="Q")]) == ""


def test_extract_last_answer():
    msgs = [
        HumanMessage(content="Q"),
        AIMessage(content="First"),
        ToolMessage(content="...", tool_call_id="c1"),
        AIMessage(content="Final answer."),
    ]
    assert _extract_last_answer(msgs) == "Final answer."


def test_extract_last_answer_empty_content():
    msgs = [HumanMessage(content="Q"), AIMessage(content="")]
    assert _extract_last_answer(msgs) == ""


# --- refusal expectation (truly_out vs stale labels) ---


def test_expected_refusal_truly_out():
    assert expected_refusal({"id": "Q6", "type": "out_corpus"}) is True
    assert all(expected_refusal({"id": qid, "type": "out_corpus"}) is True for qid in TRULY_OUT)


def test_expected_refusal_stale_out_not_scored():
    # Q41 等标签过期：不计分（None），不误报为应拒答。
    assert expected_refusal({"id": "Q41", "type": "out_corpus"}) is None
    assert expected_refusal({"id": "Q48", "type": "out_corpus"}) is None


def test_expected_refusal_in_corpus_should_answer():
    assert expected_refusal({"id": "Q1", "type": "in_corpus"}) is False
    assert expected_refusal({"id": "Q39", "type": "multi_hop"}) is False


# --- graph result extraction ---


def test_extract_result_full():
    result = {
        "messages": [
            HumanMessage(content="Q"),
            AIMessage(content=""),
            ToolMessage(
                content="Document 1\nSource: a.pdf\nCITATIONS_JSON: [{\"chunk_id\": \"a-p1-c1\"}]",
                tool_call_id="c1",
            ),
            AIMessage(content="The answer.\n\nSources:\n- a.pdf"),
        ],
        "citations": [{"source": "a.pdf", "chunk_id": "a-p1-c1"}],
    }
    answer, tool_called, cited_ids, context = extract_result(result)
    assert answer.startswith("The answer.")
    assert tool_called is True
    assert cited_ids == ["a-p1-c1"]
    # CITATIONS_JSON 尾行从判官证据里剥掉
    assert "CITATIONS_JSON" not in context
    assert "Document 1" in context


def test_extract_result_no_tool_no_citations():
    result = {"messages": [HumanMessage(content="Q"), AIMessage(content="No evidence.")], "citations": []}
    answer, tool_called, cited_ids, context = extract_result(result)
    assert answer == "No evidence."
    assert tool_called is False
    assert cited_ids == []
    assert context == ""


# --- judge JSON parsing ---


def test_parse_judge_json_plain():
    data = parse_judge_json('{"faithfulness": "supported", "reason": "ok"}')
    assert data == {"faithfulness": "supported", "reason": "ok"}


def test_parse_judge_json_fenced_with_prose():
    text = 'Here is my verdict:\n```json\n{"faithfulness": "partial", "reason": "mixed"}\n```\nDone.'
    data = parse_judge_json(text)
    assert data["faithfulness"] == "partial"


def test_parse_judge_json_invalid():
    assert parse_judge_json("no json here at all") is None
    assert parse_judge_json("{broken") is None
