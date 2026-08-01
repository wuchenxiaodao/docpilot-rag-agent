"""Semantic section chunker -- pure functions, no GPU / Embedding / Chroma deps."""

import os

# AcmeTech Employee Handbook section titles, in PDF order.
SECTION_TITLES = [
    "Welcome Message",
    "Company Mission & Values",
    "Work Hours & Attendance",
    "Remote Work Policy",
    "Code of Conduct",
    "Leave & Time-Off Policies",
    "IT & Security Guidelines",
    "Employee Benefits Overview",
    "Contact & Support",
]

# Document title. Not emitted as its own chunk.
DOCUMENT_TITLE = "AcmeTech Employee Handbook"


def _classify_line(stripped: str, section_titles: list[str]) -> str | None:
    """Return the title if the line is a known section title, else None."""
    if stripped in section_titles:
        return stripped
    return None


def _doc_slug(source: str) -> str:
    """Filename without extension, used as the chunk_id prefix."""
    base = os.path.basename(source)
    stem, _ = os.path.splitext(base)
    return stem or "doc"


def _build_chunk(content: str, source: str, section: str, pages_set: set[int], index: int) -> dict:
    """Build one chunk dict with full citation metadata. index is 0-based."""
    sorted_pages = sorted(pages_set)
    first_page = sorted_pages[0] if sorted_pages else 0
    slug = _doc_slug(source)
    return {
        "page_content": content,
        "metadata": {
            "source": source,
            "title": DOCUMENT_TITLE,
            "section": section,
            "page": first_page,
            "pages": ",".join(str(p) for p in sorted_pages),
            "chunk_id": f"{slug}-p{first_page}-c{index + 1}",
        },
    }


def split_by_sections(pages: list[tuple[str, int, str]]) -> list[dict]:
    """Split page text into chunks by section title.

    Args:
        pages: list of (page_text, page_number_1based, source).
    Returns:
        list of dicts, each with page_content and metadata
        (source / title / section / page / pages / chunk_id).
    """
    section_titles = SECTION_TITLES
    skip_titles = {DOCUMENT_TITLE}

    chunks: list[dict] = []
    current_section: str | None = None
    current_lines: list[str] = []
    current_pages: set[int] = set()
    current_source: str = ""
    found_any_title = False

    for page_text, page_num, source in pages:
        lines = page_text.split("\n")
        for line in lines:
            stripped = line.strip()
            title = _classify_line(stripped, section_titles)
            if title is not None:
                found_any_title = True
                # Close the previous section.
                if current_section is not None and current_section not in skip_titles:
                    content = "\n".join(current_lines).strip()
                    if content:
                        chunks.append(
                            _build_chunk(
                                content,
                                current_source,
                                current_section,
                                current_pages,
                                len(chunks),
                            )
                        )
                # Start a new section.
                if title in skip_titles:
                    current_section = None
                    current_lines = []
                    current_pages = set()
                    current_source = source
                else:
                    current_section = title
                    current_lines = [stripped]
                    current_pages = {page_num}
                    current_source = source
            else:
                if current_section is not None:
                    current_lines.append(stripped)
                    current_pages.add(page_num)

    # Flush the final section.
    if current_section is not None and current_section not in skip_titles:
        content = "\n".join(current_lines).strip()
        if content:
            chunks.append(
                _build_chunk(
                    content,
                    current_source,
                    current_section,
                    current_pages,
                    len(chunks),
                )
            )

    if not found_any_title:
        raise ValueError(
            "No known section titles found in the document. Cannot create semantic chunks."
        )

    return chunks
