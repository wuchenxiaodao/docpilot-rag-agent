"""语义章节切分器单元测试 — 不依赖 GPU / Embedding / Chroma / API Key。"""
import pytest

from experiments.semantic_chunker import (
    DOCUMENT_TITLE,
    SECTION_TITLES,
    _classify_line,
    split_by_sections,
)

# ── 辅助构造测试数据 ──────────────────────────────────────────────────────────

PAGE_1 = """AcmeTech Employee Handbook

Welcome Message
Welcome to AcmeTech! We are thrilled to have you on board.

Company Mission & Values
Our mission is to innovate and deliver excellence.

Work Hours & Attendance
Our core working hours are 10:00 AM to 3:00 PM.

Remote Work Policy
Employees may work remotely up to three days per week."""

PAGE_2 = """Remote employees must ensure they are available during core hours.

Code of Conduct
All employees must adhere to the highest standards of integrity.

Leave & Time-Off Policies
AcmeTech provides 12 weeks paid leave for parental leave."""

PAGE_3 = """Additional leave policies are detailed in the employee portal.

IT & Security Guidelines
All employees must follow security best practices.

Employee Benefits Overview
AcmeTech offers comprehensive health and dental coverage.

Contact & Support
For questions, please contact HR at hr@acmetech.com."""


def make_pages(*page_specs: tuple[str, int, str]) -> list[tuple[str, int, str]]:
    return list(page_specs)


# ── _classify_line ────────────────────────────────────────────────────────────

class TestClassifyLine:
    def test_known_title(self):
        assert _classify_line("Remote Work Policy", SECTION_TITLES) == "Remote Work Policy"

    def test_unknown_text(self):
        assert _classify_line("Some random paragraph", SECTION_TITLES) is None

    def test_empty_string(self):
        assert _classify_line("", SECTION_TITLES) is None

    def test_document_title_not_in_section_titles(self):
        assert _classify_line(DOCUMENT_TITLE, SECTION_TITLES) is None

    def test_partial_match_not_enough(self):
        assert _classify_line("Work Hours", SECTION_TITLES) is None


# ── split_by_sections ────────────────────────────────────────────────────────

class TestSplitBySections:
    def test_expected_section_count(self):
        pages = make_pages(
            (PAGE_1, 1, "handbook.pdf"),
            (PAGE_2, 2, "handbook.pdf"),
            (PAGE_3, 3, "handbook.pdf"),
        )
        chunks = split_by_sections(pages)
        assert len(chunks) == 9

    def test_title_retained_in_page_content(self):
        pages = make_pages(
            (PAGE_1, 1, "handbook.pdf"),
            (PAGE_2, 2, "handbook.pdf"),
            (PAGE_3, 3, "handbook.pdf"),
        )
        chunks = split_by_sections(pages)
        welcome = next(c for c in chunks if c["metadata"]["section"] == "Welcome Message")
        assert "Welcome Message" in welcome["page_content"]

    def test_source_preserved(self):
        pages = make_pages(
            (PAGE_1, 1, "handbook.pdf"),
            (PAGE_2, 2, "handbook.pdf"),
            (PAGE_3, 3, "handbook.pdf"),
        )
        chunks = split_by_sections(pages)
        for chunk in chunks:
            assert chunk["metadata"]["source"] == "handbook.pdf"

    def test_pages_is_string(self):
        pages = make_pages(
            (PAGE_1, 1, "handbook.pdf"),
            (PAGE_2, 2, "handbook.pdf"),
            (PAGE_3, 3, "handbook.pdf"),
        )
        chunks = split_by_sections(pages)
        for chunk in chunks:
            assert isinstance(chunk["metadata"]["pages"], str)
            assert chunk["metadata"]["pages"]  # not empty

    def test_remote_work_spans_pages_1_and_2(self):
        pages = make_pages(
            (PAGE_1, 1, "handbook.pdf"),
            (PAGE_2, 2, "handbook.pdf"),
            (PAGE_3, 3, "handbook.pdf"),
        )
        chunks = split_by_sections(pages)
        rw = next(c for c in chunks if c["metadata"]["section"] == "Remote Work Policy")
        assert rw["metadata"]["pages"] == "1,2"

    def test_page2_remote_work_continuation_in_remote_work_chunk(self):
        """第 2 页开头的 'Remote employees must ensure...' 应属于 Remote Work Policy。"""
        pages = make_pages(
            (PAGE_1, 1, "handbook.pdf"),
            (PAGE_2, 2, "handbook.pdf"),
            (PAGE_3, 3, "handbook.pdf"),
        )
        chunks = split_by_sections(pages)
        rw = next(c for c in chunks if c["metadata"]["section"] == "Remote Work Policy")
        assert "Remote employees must ensure" in rw["page_content"]
        assert "Remote Work Policy" in rw["page_content"]

    def test_document_title_not_a_chunk(self):
        pages = make_pages(
            (PAGE_1, 1, "handbook.pdf"),
            (PAGE_2, 2, "handbook.pdf"),
            (PAGE_3, 3, "handbook.pdf"),
        )
        chunks = split_by_sections(pages)
        sections = [c["metadata"]["section"] for c in chunks]
        assert DOCUMENT_TITLE not in sections

    def test_no_empty_chunks(self):
        pages = make_pages(
            (PAGE_1, 1, "handbook.pdf"),
            (PAGE_2, 2, "handbook.pdf"),
            (PAGE_3, 3, "handbook.pdf"),
        )
        chunks = split_by_sections(pages)
        for chunk in chunks:
            assert chunk["page_content"].strip(), f"Empty chunk for section {chunk['metadata']['section']}"

    def test_no_known_titles_raises_error(self):
        pages = make_pages(
            ("Some random text that has no section headers at all", 1, "unknown.pdf"),
            ("More random text continues here", 2, "unknown.pdf"),
        )
        with pytest.raises(ValueError, match="No known section titles found"):
            split_by_sections(pages)

    def test_single_page_with_all_sections(self):
        """如果所有章节都在一页内，也应正确切分。"""
        text = """Welcome Message
Hello and welcome!

Company Mission & Values
We innovate.

Work Hours & Attendance
Core hours 10-3.

Remote Work Policy
Work remotely.

Code of Conduct
Be ethical.

Leave & Time-Off Policies
12 weeks parental.

IT & Security Guidelines
Follow security rules.

Employee Benefits Overview
Great benefits.

Contact & Support
Contact HR."""
        pages = make_pages((text, 1, "single.pdf"))
        chunks = split_by_sections(pages)
        assert len(chunks) == 9

    def test_section_ordering(self):
        """Chunk 顺序应与 PDF 中章节出现顺序一致。"""
        pages = make_pages(
            (PAGE_1, 1, "handbook.pdf"),
            (PAGE_2, 2, "handbook.pdf"),
            (PAGE_3, 3, "handbook.pdf"),
        )
        chunks = split_by_sections(pages)
        actual_sections = [c["metadata"]["section"] for c in chunks]
        assert actual_sections == SECTION_TITLES