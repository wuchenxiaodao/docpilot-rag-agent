"""Markdown section chunker for DocPilot v2 corpus expansion.

Pure functions, no GPU / Embedding / Chroma deps.

Design constraints (locked 2026-07-31):
- Explicit per-document heading config: filename -> heading level + exact
  section title list. NO heuristic heading detection (no "auto-find all ##").
- AcmeTech PDF does NOT go through this file; it uses the original
  semantic_chunker.split_by_sections code path, which is left byte-identical.
- chunk_id rule mirrors the AcmeTech one: "{slug}-p{page}-c{index+1}".
  A markdown file is treated as a single page, so page is always 1.
- Mixed chunking (AcmeTech PDF titles vs markdown headings) is a known
  boundary of the v2 corpus and must be disclosed in eval reports.

Difference from the AcmeTech chunker, deliberate:
- Content lines keep their leading whitespace (line.rstrip() instead of
  line.strip()) so markdown code blocks and nested lists stay intact.
  Heading detection still matches on the stripped line.
- Content before the first configured section heading is dropped
  (mirrors AcmeTech dropping the document title preamble). For VertexAI.md
  this drops the "## Using Gemini Developer API" section, which has no
  level-3 headings under it; that is a documented trade-off of choosing
  level 3 for this file.
"""

import os

from experiments.semantic_chunker import _doc_slug

# Explicit per-document section config.
#   level:     which heading level marks a section boundary (2 = "##", 3 = "###").
#   doc_title: the document's own top-level title (metadata only, not a chunk).
#   sections:  exact section heading texts in file order, copied from the files
#              on 2026-07-31. A section boundary is a line whose stripped form
#              equals "#"*level + " " + title. Validation at split time: every
#              configured title must be found verbatim, else ValueError.
MARKDOWN_SECTION_CONFIG: dict[str, dict] = {
    "docs/Dependency_Upgrades.md": {
        "level": 2,
        "doc_title": "Dependency & Version Upgrade Management",
        "sections": [
            "Purpose",
            "Where versions live",
            "Upgrade workflow (the recipe)",
            "Live end-to-end test (no API key needed)",
            "Triage principles",
            "Coupling constraints & gotchas (learned the hard way)",
            "Currently deferred upgrades (backlog)",
            "Python version policy",
        ],
    },
    "docs/maintenance/Weekly_Maintenance_Run.md": {
        "level": 2,
        "doc_title": "Weekly Maintenance Run (agent orchestrator prompt)",
        "sections": [
            "Step 0 — Parity gate (run or skip?)",
            "Step 1 — Calendar gate (which extra phases run today?)",
            "Ground rules (apply to every phase)",
            "Untrusted content & prompt-injection defense (read before Phase A)",
            "Phase A — Community triage (every run)",
            "Phase B — Stale sweep (every run; the one pre-authorized write)",
            "Phase C — Live app health check (every run)",
            "Phase D — Infra smoke tests (every run)",
            "Phase E — Model catalog refresh (first run of each month)",
            "Phase F — Dependency refresh (first run of each month)",
            "CI follow-through on PRs this run opened (monthly runs)",
            "Final step — The digest",
        ],
    },
    "docs/VertexAI.md": {
        "level": 3,
        "doc_title": "Working with Google Models",
        "sections": [
            "Prerequisites",
            "About Authentication",
            "Models",
            "Steps",
            "Verify Your Setup",
            "Production Note",
        ],
    },
    "docs/maintenance/Daily_Sentinel.md": {
        "level": 2,
        "doc_title": "Daily Sentinel (agent prompt)",
        "sections": [
            "Hard rules",
            "Checks (a few minutes total)",
            "The urgency bar — notify ONLY for",
            "If something IS urgent",
        ],
    },
    "docs/GitHub_MCP_Agent.md": {
        "level": 2,
        "doc_title": "GitHub MCP Agent",
        "sections": [
            "Features",
            "Configuration",
            "GitHub Personal Access Token",
            "Usage",
        ],
    },
    "docs/AGUI.md": {
        "level": 2,
        "doc_title": "AG-UI Protocol Support",
        "sections": [
            "Endpoints",
            "Connecting a frontend",
            "Trying it out",
            "Behavior notes",
        ],
    },
    "docs/architecture.md": {
        "level": 2,
        "doc_title": "DocPilot 请求架构",
        "sections": [
            "核心调用链",
            "各模块职责",
            "流式调用",
            "流式与非流式的区别",
        ],
    },
    "docs/File_Based_Credentials.md": {
        "level": 2,
        # The source file's own # line is "File Based Crendentials" (sic);
        # kept verbatim for metadata fidelity.
        "doc_title": "File Based Crendentials",
        "sections": [
            "How it works",
            "Suggested Use",
            "Production Options",
        ],
    },
    "docs/RAG_Assistant.md": {
        "level": 2,
        "doc_title": "Creating a RAG assistant",
        "sections": [
            "Setting up Chroma",
            "Configuring the RAG assistant",
        ],
    },
}


def _build_markdown_chunk(content: str, source: str, doc_title: str, section: str, index: int) -> dict:
    """Build one chunk dict with citation metadata. index is 0-based.

    page/pages are always 1/"1": a markdown file is a single logical page.
    """
    slug = _doc_slug(source)
    return {
        "page_content": content,
        "metadata": {
            "source": source,
            "title": doc_title,
            "section": section,
            "page": 1,
            "pages": "1",
            "chunk_id": f"{slug}-p1-c{index + 1}",
        },
    }


def split_markdown_by_headings(text: str, source: str, config: dict, dropped_out: list | None = None) -> list[dict]:
    """Split markdown text into chunks by explicitly configured section headings.

    Args:
        text:   full file content (newlines already normalized).
        source: path written into metadata.source (also drives the chunk_id slug).
        config: one entry from MARKDOWN_SECTION_CONFIG.
        dropped_out: optional list; if given, it is filled with the verbatim
            lines that fall BEFORE the first configured heading and are
            therefore not indexed (document title preamble, and for level-3
            configs any higher-level sections such as VertexAI's
            "## Using Gemini Developer API"). Caller uses this to report
            known-unindexed content. None means "don't track".

    Returns:
        list of dicts with page_content and metadata
        (source / title / section / page / pages / chunk_id).

    Raises:
        ValueError: if any configured section title is not found verbatim in
            the file. This is the stability guard for the explicit config:
            a renamed heading fails loudly instead of silently dropping chunks.
    """
    level = config["level"]
    prefix = "#" * level + " "
    known_headings = {prefix + t: t for t in config["sections"]}

    chunks: list[dict] = []
    found_titles: list[str] = []
    current_section: str | None = None
    current_lines: list[str] = []

    for line in text.split("\n"):
        stripped = line.strip()
        title = known_headings.get(stripped)
        if title is not None:
            found_titles.append(title)
            # Close the previous section.
            if current_section is not None:
                content = "\n".join(current_lines).strip()
                if content:
                    chunks.append(
                        _build_markdown_chunk(
                            content, source, config["doc_title"], current_section, len(chunks)
                        )
                    )
            # Start a new section.
            current_section = title
            current_lines = [stripped]
        else:
            if current_section is not None:
                # rstrip only: keep leading whitespace so code blocks survive.
                current_lines.append(line.rstrip())
            elif dropped_out is not None:
                # Pre-first-heading line: not indexed anywhere; record it so the
                # caller can report known-unindexed content.
                dropped_out.append(line.rstrip())

    # Flush the final section.
    if current_section is not None:
        content = "\n".join(current_lines).strip()
        if content:
            chunks.append(
                _build_markdown_chunk(
                    content, source, config["doc_title"], current_section, len(chunks)
                )
            )

    missing = [t for t in config["sections"] if t not in found_titles]
    if missing:
        raise ValueError(
            f"{source}: {len(missing)} configured section title(s) not found "
            f"verbatim in file: {missing}"
        )

    return chunks
