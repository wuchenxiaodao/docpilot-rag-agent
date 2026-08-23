"""Tests for hybrid retrieval: BM25 + vector RRF fusion (roadmap #7)
and multi-intent query split (roadmap #8).

FakeStore simulates the vector store; no GPU / real Chroma involved.
"""

from langchain_core.documents import Document

from agents.tools import _BM25Index, HybridRetriever, _split_query


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


# --- roadmap #8: multi-intent query split -------------------------------------


def test_split_query_single_intent_passthrough():
    """无 ', and <疑问词>' 结构的问题原样返回，单元素列表。"""
    assert _split_query("What are the regular office hours?") == ["What are the regular office hours?"]
    assert _split_query("How much PTO does a new parent get?") == [
        "How much PTO does a new parent get?"
    ]


def test_split_query_rejects_degenerate_second_clause():
    """Q23 形态：', and how?' 第二问太短，守卫应拒绝拆分、原样通过。"""
    q = "I don't have any LLM API keys — can I still live-test the upgraded service end to end, and how?"
    assert _split_query(q) == [q]


def test_split_query_rejects_anaphoric_pronoun():
    """Q17 形态：'...who should they contact?' 含代词 'they'，前指丢失、
    子查询语义不完整，拒绝拆分、走原单查询路径。"""
    q = "If an employee receives a suspicious email while working remotely, what should they do, and who should they contact?"
    assert _split_query(q) == [q]


def test_split_query_two_intents():
    """Q39 形态：'what command regenerates the lockfile, and which environment variable...' 拆成两段。"""
    q = "After bumping version pins, what command regenerates the lockfile, and which environment variable lets the test run without credentials?"
    parts = _split_query(q)
    assert len(parts) == 2
    assert parts[0].startswith("After bumping version pins")
    assert parts[0].endswith("what command regenerates the lockfile")
    assert parts[1].startswith("which environment variable")


def test_split_query_preserves_question_word_in_second_clause():
    """疑问词应留在第二子查询开头（'who/which/what ...'），不被吃掉。"""
    q = "After bumping version pins, what command regenerates the lockfile, and which environment variable lets the test run?"
    parts = _split_query(q)
    assert len(parts) == 2
    assert parts[1].startswith("which environment variable")


def test_hybrid_multi_intent_round_robin_rescues_both_expected_chunks():
    """Q39 真实形态：单查询时第二个目标 chunk（env var）排不进 top-3，
    拆分后两个自包含子查询各占一席，双 chunk 都进 top-3。代词守卫
    保证只对这种自包含子查询触发拆分。"""
    lockfile = _doc("lockfile", "Run uv lock --upgrade to regenerate the lockfile after editing version pins in pyproject.")
    envvar = _doc("envvar", "Set USE_FAKE_MODEL=true to run the e2e test suite without provider API credentials.")
    others = [
        _doc("distract1", "The CI pipeline runs on GitHub Actions with a Python version matrix."),
        _doc("distract2", "Pre-commit hooks enforce ruff linting and formatting on every commit."),
        _doc("filler", "Documentation is built with mkdocs material theme."),
    ]
    store = FakeStore([lockfile, envvar] + others)
    # 单查询时向量把两个干扰项排在 lockfile/envvar 之前
    store.vector_ranking = ["distract1", "distract2", "lockfile", "envvar", "filler"]

    retriever = HybridRetriever(store)
    q = "After bumping version pins, what command regenerates the lockfile, and which environment variable lets the test run without credentials?"
    results = retriever.invoke(q)
    ids = [r.metadata["chunk_id"] for r in results]
    assert "lockfile" in ids
    assert "envvar" in ids  # 拆分后第二席位被 envvar 占走


def test_hybrid_multi_intent_dedup_when_both_subqueries_hit_same_chunk():
    """两个子查询命中同一 chunk 时 round-robin 去重，只占一席。"""
    shared = _doc("shared", "The on-call runbook covers alert routing and escalation policies end to end.")
    others = [
        _doc("a", "Alert routing sends pages to the primary on-call engineer."),
        _doc("b", "Escalation policy triggers after 15 minutes without acknowledgement."),
        _doc("c", "The dashboard shows live system health metrics."),
    ]
    store = FakeStore([shared] + others)
    # 两个子查询向量都把 shared 排第一
    store.vector_ranking = ["shared", "a", "b", "c"]

    retriever = HybridRetriever(store)
    q = "What handles alert routing, and what covers escalation policies?"
    results = retriever.invoke(q)
    ids = [r.metadata["chunk_id"] for r in results]
    assert ids.count("shared") == 1  # 去重
    assert len(ids) == 3  # 席位补满到 k
