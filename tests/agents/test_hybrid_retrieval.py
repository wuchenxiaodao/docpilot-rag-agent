"""Tests for hybrid retrieval: BM25 + vector RRF fusion (roadmap #7).

FakeStore simulates the vector store; no GPU / real Chroma involved.
"""

from langchain_core.documents import Document

from agents.tools import _BM25Index, HybridRetriever


def _doc(cid: str, text: str) -> Document:
    return Document(page_content=text, metadata={"chunk_id": cid, "source": f"{cid}.pdf"})


class FakeStore:
    """Duck-type Chroma: preset vector ranking; get() reflects current docs."""

    def __init__(self, docs: list[Document]):
        self.docs = docs
        self.vector_ranking: list[str] = []  # chunk_ids in "semantic" order

    def get(self, include=None):
        ids = [d.metadata["chunk_id"] for d in self.docs]
        if include and "documents" in include:
            return {
                "ids": ids,
                "documents": [d.page_content for d in self.docs],
                "metadatas": [d.metadata for d in self.docs],
            }
        return {"ids": ids}

    def similarity_search(self, query: str, k: int) -> list[Document]:
        by_id = {d.metadata["chunk_id"]: d for d in self.docs}
        return [by_id[cid] for cid in self.vector_ranking[:k] if cid in by_id]


def test_bm25_exact_match_ranks_first():
    docs = [
        _doc("a", "If you have questions or need support, reach out to HR."),
        _doc("b", "Our regular office hours are 9:00 AM to 5:00 PM, Monday through Friday."),
        _doc("c", "The company mission is to develop cutting-edge software."),
    ]
    index = _BM25Index(docs)
    assert index.search("What are the regular office hours?", k=3)[0] == "b"


def test_hybrid_rescues_lexical_hit_buried_by_vector():
    """Q12 形态：目标 chunk 向量靠后但 BM25 第 1（逐字短语命中），
    干扰项与查询无词面重叠时，融合后目标应升至首位。"""
    target = _doc("target", "Work Hours & Attendance — Our regular office hours are 9:00 AM to 5:00 PM.")
    others = [
        _doc("distract1", "If you have questions or need support, reach out via payroll."),
        _doc("distract2", "Employee Benefits include health insurance and retirement plans."),
        _doc("distract3", "Cutting-edge software solutions define our mission."),
        _doc("filler", "Coffee is available in every kitchen."),
    ]
    store = FakeStore([target] + others)
    store.vector_ranking = ["distract1", "distract2", "distract3", "target"]

    retriever = HybridRetriever(store)
    results = retriever.invoke("What are the regular office hours?")
    ids = [r.metadata["chunk_id"] for r in results]
    assert ids[0] == "target"
    assert len(results) == 3


def test_hybrid_partial_overlap_distractor_still_in_top3():
    """真实 Q12 形态：干扰项既有更好的向量位次又含部分查询词（公司名）。
    加权融合救不回第 1，但目标必须进 Top-3（50 题实测 rank None->3）。"""
    target = _doc("target", "Work Hours & Attendance — Our regular office hours are 9:00 AM to 5:00 PM.")
    others = [
        _doc("distract1", "If you have questions or need support, reach out: HR, IT, payroll."),
        _doc("distract2", "Employee Benefits include health insurance and retirement plans."),
        _doc("distract3", "Our mission at AcmeTech is to develop cutting-edge software."),
        _doc("filler", "The kitchen has coffee."),
    ]
    store = FakeStore([target] + others)
    store.vector_ranking = ["distract1", "distract2", "distract3", "target"]

    retriever = HybridRetriever(store)
    results = retriever.invoke("What are AcmeTech's regular office hours?")
    ids = [r.metadata["chunk_id"] for r in results]
    assert "target" in ids  # 进 top-3 即救援成功；第 1 名被部分重叠干扰项占走


def test_hybrid_keeps_vector_only_semantic_hit():
    """纯语义命中（BM25 零分）仍应保留在结果里——RRF 不伤 paraphrase。"""
    semantic = _doc("semantic", "Staff may telecommute up to three days each week.")
    store = FakeStore([semantic])
    store.vector_ranking = ["semantic"]
    retriever = HybridRetriever(store)
    results = retriever.invoke("how often can people work from home")  # 无词面重叠
    assert [r.metadata["chunk_id"] for r in results] == ["semantic"]


def test_stale_index_rebuilt_after_ingest():
    store = FakeStore([_doc("a", "existing content about benefits")])
    store.vector_ranking = ["a"]
    retriever = HybridRetriever(store)
    assert retriever.invoke("benefits")[0].metadata["chunk_id"] == "a"

    # 在线入库新文档后（条目数变化），BM25 索引应自动重建并命中新内容
    store.docs.append(_doc("new", "brand new quarterly all-hands meeting schedule"))
    results = retriever.invoke("quarterly all-hands meeting")
    assert any(r.metadata["chunk_id"] == "new" for r in results)


def test_search_kwargs_k_respected():
    store = FakeStore([_doc("a", "alpha"), _doc("b", "beta"), _doc("c", "gamma")])
    store.vector_ranking = ["a", "b", "c"]
    retriever = HybridRetriever(store)
    retriever.search_kwargs["k"] = 2  # eval_retrieval.py --k 的调参入口
    assert len(retriever.invoke("anything")) == 2
