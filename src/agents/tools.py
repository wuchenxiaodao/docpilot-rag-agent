import json
import math
import os
import re
import threading
from collections import Counter
from typing import Literal, TypedDict

import numexpr
from langchain_chroma import Chroma
from langchain_core.documents import Document
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


class _CitationRequired(TypedDict):
    source: str
    page: int | None
    chunk_id: str | None


class Citation(_CitationRequired, total=False):
    """一条引用记录。source 必有，page 和 chunk_id 可能缺失；excerpt 为原文摘录。"""

    excerpt: str


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
                "excerpt": " ".join(doc.page_content.split())[:160],
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
    model_path = _get_embedding_model_path()
    db_path = _get_chroma_db_path()
    key = (db_path, model_path)

    # 注意不可持锁调用 get_chroma_store（非重入锁会死锁）：先查缓存，锁外建库
    with _chroma_cache_lock:
        retriever = _retriever_cache.get(key)
    if retriever is not None:
        return retriever
    retriever = HybridRetriever(get_chroma_store())
    with _chroma_cache_lock:
        return _retriever_cache.setdefault(key, retriever)


# ---------------------------------------------------------------------------
# 混合检索：BM25 词面匹配 + 向量语义检索，RRF 融合。
# 动机见 evals/failure_analysis.md 例1（Q12）：答案逐字在库内，却被
# 「表层语义相近」的无关 chunk 压出 Top-k。BM25 提供词面精确匹配信号，
# RRF（k=60 阻尼）对纯语义命中保持保守，不伤 paraphrase 类问题。
_BM25_K1 = 1.5
_BM25_B = 0.75
_RRF_K = 60
_CANDIDATE_K = 8
# BM25 融合权重（向量恒为 1.0）。50 题扫参：0.6–0.8 为稳定平台，
# Recall@3 达 100%（Q12 获救、Q39 升至第1），代价 Q17/Q32 从第1滑到第2；
# ≥0.9 词面信号开始压过语义，R@1 崩塌（1.0 时 82.9%）。
_BM25_WEIGHT = 0.7


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


class _BM25Index:
    """纯 Python BM25（k1=1.5, b=0.75）。当前语料 56 chunk，构建 <50ms。"""

    def __init__(self, docs: list[Document]):
        self._by_id: dict[str, Document] = {}
        self._doc_tokens: list[list[str]] = []
        self._doc_ids: list[str] = []
        for i, d in enumerate(docs):
            cid = d.metadata.get("chunk_id") or f"idx-{i}"
            self._doc_ids.append(cid)
            self._doc_tokens.append(_tokenize(d.page_content))
            self._by_id[cid] = d
        self._n = len(docs)
        self._avgdl = sum(len(t) for t in self._doc_tokens) / max(1, self._n)
        df: dict[str, int] = {}
        for toks in self._doc_tokens:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        self._df = df
        self._idf = {t: math.log((self._n - d + 0.5) / (d + 0.5) + 1) for t, d in df.items()}

    def doc_by_id(self, cid: str) -> Document | None:
        return self._by_id.get(cid)

    def search(self, query: str, k: int) -> list[str]:
        q_tokens = _tokenize(query)
        # 语料自适应停用词：df 超过 80% 文档的词（the/and/to…）不参与打分，
        # 否则它们给无关文档贡献微弱正分即可进榜，被融合放大。
        stop = {t for t in q_tokens if self._df.get(t, 0) > 0.8 * self._n}
        q_tokens = [t for t in q_tokens if t not in stop]
        scores: list[tuple[float, int]] = []
        for i, toks in enumerate(self._doc_tokens):
            dl = len(toks)
            tf = Counter(toks)
            s = 0.0
            for t in q_tokens:
                f = tf.get(t)
                if not f:
                    continue
                idf = self._idf.get(t)
                if idf is None:
                    continue
                norm = _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / max(1e-9, self._avgdl))
                s += idf * f * (_BM25_K1 + 1) / (f + norm)
            scores.append((s, i))
        scores.sort(key=lambda x: (-x[0], x[1]))
        return [self._doc_ids[i] for s, i in scores[:k] if s > 0]


class HybridRetriever:
    """向量 + BM25 的 RRF 融合检索器。

    duck-type 兼容原 Chroma retriever 的 invoke()/search_kwargs 接口
    （eval_retrieval.py 的 --k 依赖后者）。BM25 索引按库内条目数做
    廉价陈旧检查：在线入库/删除后自动重建。
    """

    def __init__(
        self,
        store: Chroma,
        rrf_k: int = _RRF_K,
        candidate_k: int = _CANDIDATE_K,
        bm25_weight: float = _BM25_WEIGHT,
    ):
        self._store = store
        self.search_kwargs: dict = {"k": 3}
        self._rrf_k = rrf_k
        self._candidate_k = candidate_k
        self._bm25_weight = bm25_weight
        self._index: _BM25Index | None = None
        self._index_count = -1

    def _ensure_index(self) -> _BM25Index:
        count = len(self._store.get()["ids"])
        if self._index is None or count != self._index_count:
            data = self._store.get(include=["documents", "metadatas"])
            docs = [
                Document(page_content=t, metadata=m or {})
                for t, m in zip(data["documents"], data["metadatas"])
            ]
            self._index = _BM25Index(docs)
            self._index_count = count
        return self._index

    @staticmethod
    def _cid(d: Document, fallback: str) -> str:
        return d.metadata.get("chunk_id") or fallback

    def invoke(self, query: str) -> list[Document]:
        k = int(self.search_kwargs.get("k", 3))
        vector_hits = self._store.similarity_search(query, k=self._candidate_k)
        index = self._ensure_index()
        bm25_ids = index.search(query, self._candidate_k)

        id_to_doc: dict[str, Document] = {}
        for rank, d in enumerate(vector_hits):
            id_to_doc[self._cid(d, f"vector-{rank}")] = d
        for cid in bm25_ids:
            if cid not in id_to_doc:
                d = index.doc_by_id(cid)
                if d is not None:
                    id_to_doc[cid] = d

        fused: dict[str, float] = {}
        for rank, d in enumerate(vector_hits):
            cid = self._cid(d, f"vector-{rank}")
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (self._rrf_k + rank + 1)
        for rank, cid in enumerate(bm25_ids):
            fused[cid] = fused.get(cid, 0.0) + self._bm25_weight / (self._rrf_k + rank + 1)

        top = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [id_to_doc[cid] for cid, _ in top if cid in id_to_doc]


_retriever_cache: dict[tuple[str, str], HybridRetriever] = {}


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
    给 LLM 用的薄包装：把结构化结果转成模型能读的文本；末尾附一行
    CITATIONS_JSON（结构化引用），由 collect_citations 节点解析给前端，
    模型可忽略该行。
    """
    result = database_search_func(query)

    if result["status"] == "no_answer":
        return "NO_RELEVANT_DOCUMENTS_FOUND"

    citations_json = json.dumps(result["citations"], ensure_ascii=False)
    return f"{result['context']}\n\nCITATIONS_JSON: {citations_json}"


database_search: BaseTool = tool(_database_search_for_tool)
database_search.name = "Database_Search"
