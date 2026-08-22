"""DocPilot 在线入库：把用户上传的 PDF/DOCX 切分、嵌入并写入 Chroma 库。

设计约定（与既有库的元数据形状保持一致，集成测试锚定这些字段）：
- chunk metadata 必带非空 `source`（上传文件的原始文件名）与 `chunk_id`
- PDF 保留 PyPDFLoader 给出的真实 `page`；DOCX 无页概念，不写 page 键
- 同名文件重复上传 = 全量替换：先按 source 删除旧 chunk 再写入，保证幂等
- 切分用 RecursiveCharacterTextSplitter（与 scripts/create_chroma_db.py 相同参数）
"""

import logging
import os
import re
import tempfile
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agents.tools import get_chroma_store

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50MB

_text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)

_SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9_\-]+")

# Chroma where 过滤器按 source 精确匹配；文件名入库前去掉路径与空白
def _normalize_source(filename: str) -> str:
    return os.path.basename(filename).strip()


def _safe_stem(source: str) -> str:
    """chunk_id 只保留字母数字-下划线-连字符，避免特殊字符进 id。"""
    stem = os.path.splitext(source)[0]
    cleaned = _SAFE_STEM_RE.sub("_", stem).strip("_") or "doc"
    return cleaned[:60]


@dataclass
class IngestResult:
    filename: str
    chunks_added: int
    chunks_deleted: int


def load_and_chunk(filename: str, content: bytes) -> list[Document]:
    """解析上传文件并切分。纯函数：不触向量库，可单测。"""
    source = _normalize_source(filename)
    ext = os.path.splitext(source)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {sorted(SUPPORTED_EXTENSIONS)}")
    if not content:
        raise ValueError("Empty file.")

    # loader 都是文件路径式的，落一个临时文件再解析
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        if ext == ".pdf":
            loader = PyPDFLoader(tmp_path)
        else:
            loader = Docx2txtLoader(tmp_path)
        raw_docs = loader.load()
    finally:
        os.unlink(tmp_path)

    chunks = _text_splitter.split_documents(raw_docs)
    stem = _safe_stem(source)

    result: list[Document] = []
    for i, chunk in enumerate(chunks, start=1):
        text = chunk.page_content.strip()
        if not text:
            continue
        # chunk_id 沿用既有约定 {stem}-p{page}-c{idx}；DOCX 无页码则 p0 占位
        page = chunk.metadata.get("page", 0) or 0
        metadata = {"source": source, "chunk_id": f"{stem}-p{page}-c{i}"}
        if ext == ".pdf":
            metadata["page"] = page
        result.append(Document(page_content=text, metadata=metadata))
    return result


def _existing_ids_for_basename(chroma: Chroma, source: str) -> list[str]:
    """按 basename 找同源旧 chunk。库里历史 source 是路径形式（./data\\x.pdf、
    docs/x.md），上传用的是纯文件名——只做精确匹配会漏删，重传就变双份。"""
    data = chroma.get(include=["metadatas"])
    ids = []
    for id_, meta in zip(data.get("ids", []), data.get("metadatas", [])):
        if meta and os.path.basename(str(meta.get("source", ""))) == source:
            ids.append(id_)
    return ids


def ingest_file(filename: str, content: bytes, store: Chroma | None = None) -> IngestResult:
    """切分 + 嵌入 + 写入当前配置的 Chroma 库。同名文件（按 basename）先删后加，幂等。"""
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError(f"File too large ({len(content)} bytes, max {MAX_UPLOAD_BYTES}).")

    source = _normalize_source(filename)
    chunks = load_and_chunk(filename, content)
    if not chunks:
        raise ValueError("No text content extracted from file.")

    chroma = store if store is not None else get_chroma_store()
    old_ids = _existing_ids_for_basename(chroma, source)
    if old_ids:
        chroma.delete(ids=old_ids)

    chroma.add_documents(chunks)
    logger.info("Ingested %s: %d chunks added, %d replaced", source, len(chunks), len(old_ids))
    return IngestResult(filename=source, chunks_added=len(chunks), chunks_deleted=len(old_ids))
