import math
import os
import re
import threading
from typing import Literal, TypedDict

import numexpr
from langchain_chroma import Chroma
from langchain_core.tools import BaseTool, tool
from langchain_huggingface import HuggingFaceEmbeddings


def calculator_func(expression: str) -> str:
    """Calculates a math expression using numexpr.

    Useful for when you need to answer questions about math using numexpr.
    This tool is only for math questions and nothing else. Only input
    math expressions.

    Args:
        expression (str): A valid numexpr formatted math expression.

    Returns:
        str: The result of the math expression.
    """

    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
        return re.sub(r"^\[|\]$", "", output)
    except Exception as e:
        raise ValueError(
            f'calculator("{expression}") raised error: {e}.'
            " Please try again with a valid numerical expression"
        )


calculator: BaseTool = tool(calculator_func)
calculator.name = "Calculator"


class Citation(TypedDict):
    """一条引用记录。source 必有，page 和 chunk_id 可能缺失。"""

    source: str
    page: int | None
    chunk_id: str | None


class SearchResult(TypedDict):
    """知识库检索的结构化返回值。

    status:    给程序判断用，只有 answered / no_answer 两个值
    context:   给模型看的文本
    citations: 支撑答案的证据，拒答时必须为空
    reason:    为什么没有答案，成功时必须为 None
    """

    status: Literal["answered", "no_answer"]
    context: str
    citations: list[Citation]
    reason: str | None


def build_citations(docs) -> list[Citation]:
    """从检索结果里提取结构化引用，保序去重。"""
    citations: list[Citation] = []
    seen: list[tuple] = []

    for doc in docs:
        src = doc.metadata.get("source", "")
        if not src:
            continue

        filename = os.path.basename(src)
        page = doc.metadata.get("page")
        chunk_id = doc.metadata.get("chunk_id")

        key = (filename, page, chunk_id)
        if key in seen:
            continue
        seen.append(key)

        citations.append(
            {
                "source": filename,
                "page": page,
                "chunk_id": chunk_id,
            }
        )

    return citations


# Format retrieved documents
def format_contexts(docs):
    if docs is None:
        raise TypeError("docs cannot be None")

    parts = []
    seen_citations = []  # list of (filename, page) for order-preserving dedup
    citation_lines = []  # formatted citation strings in order

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown")
        parts.append(f"Document {i}\nSource: {source}\nContent: {doc.page_content}")

        # Extract citation from real metadata only.
        src = doc.metadata.get("source", "")
        page = doc.metadata.get("page")
        if src and page is not None:
            filename = os.path.basename(src)
            pair = (filename, page)
            # Order-preserving dedup: no set().
            if pair not in seen_citations:
                seen_citations.append(pair)
                citation_lines.append(f"[{filename}，第 {page} 页]")

    result = "\n\n".join(parts)
    if citation_lines:
        result += "\n\n**References:**\n" + "\n".join(citation_lines)
    return result


def _get_embedding_model_path() -> str:
    path = os.environ.get(
        "EMBEDDING_MODEL_PATH",
        os.path.join(os.path.expanduser("~"), "Models", "Qwen3-Embedding-0.6B"),
    )
    if not os.path.isdir(path):
        raise RuntimeError(
            f"Embedding model directory not found: {path}. "
            "Set EMBEDDING_MODEL_PATH env var or place the model at the default location."
        )
    return path


def _get_chroma_db_path() -> str:
    return os.environ.get("CHROMA_DB_PATH", "./chroma_db_qwen3_semantic_chunks_v2")


# 嵌入模型加载进 CUDA 需数秒，不能每次检索都重建；按 (库路径, 模型路径)
# 缓存。换环境变量（测试场景）会得到新实例，路径解析语义不变。
_chroma_cache: dict[tuple[str, str], Chroma] = {}
_chroma_cache_lock = threading.Lock()


def _build_embeddings(model_path: str) -> HuggingFaceEmbeddings:
    try:
        return HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
            query_encode_kwargs={
                "normalize_embeddings": True,
                "prompt_name": "query",
            },
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize HuggingFaceEmbeddings with model at {model_path}."
        ) from e


def get_chroma_store() -> Chroma:
    """返回当前配置下的 Chroma 向量库句柄（缓存实例），检索与入库共用。"""
    model_path = _get_embedding_model_path()
    db_path = _get_chroma_db_path()
    key = (db_path, model_path)

    with _chroma_cache_lock:
        store = _chroma_cache.get(key)
        if store is None:
            store = Chroma(
                persist_directory=db_path,
                embedding_function=_build_embeddings(model_path),
            )
            _chroma_cache[key] = store
        return store


def load_chroma_db():
    return get_chroma_store().as_retriever(search_kwargs={"k": 3})


def database_search_func(query: str) -> SearchResult:
    """检索知识库，返回结构化结果。

    注意：本函数不做相似度阈值判断。当前配置下 Top-1 分数无法
    区分「有答案」和「无答案」——实测资料内最低分 0.4042 低于
    资料外最高分 0.4685，两组分布重叠。因此这里只处理一种确定
    情况：检索结果为空。
    """
    retriever = load_chroma_db()
    documents = retriever.invoke(query)

    if not documents:
        return {
            "status": "no_answer",
            "context": "",
            "citations": [],
            "reason": "no_documents_retrieved",
        }

    return {
        "status": "answered",
        "context": format_contexts(documents),
        "citations": build_citations(documents),
        "reason": None,
    }


def _database_search_for_tool(query: str) -> str:
    """Searches the configured DocPilot PDF/DOCX knowledge base via ChromaDB.

    Returns relevant text fragments and source metadata from indexed documents.
    给 LLM 用的薄包装：把结构化结果转成模型能读的文本。
    """
    result = database_search_func(query)

    if result["status"] == "no_answer":
        return "NO_RELEVANT_DOCUMENTS_FOUND"

    return result["context"]


database_search: BaseTool = tool(_database_search_for_tool)
database_search.name = "Database_Search"
