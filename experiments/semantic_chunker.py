"""语义章节切分器 — 纯函数，不依赖 GPU、Embedding 或 Chroma。"""

# AcmeTech Employee Handbook 的预定义章节标题（按 PDF 出现顺序）
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

# 文档标题，不生成独立 Chunk
DOCUMENT_TITLE = "AcmeTech Employee Handbook"


def _classify_line(stripped: str, section_titles: list[str]) -> str | None:
    """判断一行是否为已知章节标题，是则返回标题，否则返回 None。"""
    if stripped in section_titles:
        return stripped
    return None


def split_by_sections(pages: list[tuple[str, int, str]]) -> list[dict]:
    """按章节标题切分页面文本。

    Args:
        pages: 元素为 (page_text, page_number_1based, source) 的列表。

    Returns:
        list[dict]，每个 dict 包含：
            - page_content: str
            - metadata: dict 含 source / section / pages
    """
    section_titles = SECTION_TITLES
    skip_titles = {DOCUMENT_TITLE}

    chunks: list[dict] = []
    current_section: str | None = None
    current_lines: list[str] = []
    current_pages: set[int] = set()
    found_any_title = False

    for page_text, page_num, source in pages:
        lines = page_text.split("\n")

        for line in lines:
            stripped = line.strip()

            title = _classify_line(stripped, section_titles)
            if title is not None:
                found_any_title = True

                # 结束上一章节
                if current_section is not None and current_section not in skip_titles:
                    content = "\n".join(current_lines).strip()
                    if content:
                        chunks.append({
                            "page_content": content,
                            "metadata": {
                                "source": source,
                                "section": current_section,
                                "pages": ",".join(sorted(str(p) for p in current_pages)),
                            },
                        })

                # 开始新章节
                if title in skip_titles:
                    current_section = None
                    current_lines = []
                    current_pages = set()
                else:
                    current_section = title
                    current_lines = [stripped]
                    current_pages = {page_num}
            else:
                if current_section is not None:
                    current_lines.append(stripped)
                    current_pages.add(page_num)

    # 收尾最后章节
    if current_section is not None and current_section not in skip_titles:
        content = "\n".join(current_lines).strip()
        if content:
            chunks.append({
                "page_content": content,
                "metadata": {
                    "source": source,
                    "section": current_section,
                    "pages": ",".join(sorted(str(p) for p in current_pages)),
                },
            })

    if not found_any_title:
        raise ValueError(
            "No known section titles found in the document. "
            "Cannot create semantic chunks."
        )

    return chunks