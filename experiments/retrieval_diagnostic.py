#!/usr/bin/env python
"""
DocPilot Retrieval Quality — Baseline (Configuration A)
Measures Recall@1, Recall@3, MRR, and score distributions.
"""

import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

MODEL_PATH = os.path.join(os.path.expanduser("~"), "Models", "Qwen3-Embedding-0.6B")
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "./chroma_db_qwen3_test")

# ── Questions (10) ──────────────────────────────────────────────────────────

in_domain = [
    {
        "id": "Q1",
        "question": "What are AcmeTech's mission and values?",
        "expected_evidence": "Company Mission & Values",
    },
    {
        "id": "Q2",
        "question": "How many days per week may employees work remotely?",
        "expected_evidence": "up to three days per week",
    },
    {
        "id": "Q3",
        "question": "What are AcmeTech's core working hours?",
        "expected_evidence": "10:00 AM to 3:00 PM",
    },
    {
        "id": "Q4",
        "question": "How many weeks of paid parental leave are provided?",
        "expected_evidence": "12 weeks paid leave",
    },
    {
        "id": "Q5",
        "question": "What should employees do about suspicious emails?",
        "expected_evidence": "suspicious emails or activity to IT immediately",
    },
]

out_of_domain = [
    {
        "id": "Q6",
        "question": "What is AcmeTech's stock ticker?",
        "expected_evidence": None,
    },
    {
        "id": "Q7",
        "question": "Who is AcmeTech's CEO?",
        "expected_evidence": None,
    },
    {
        "id": "Q8",
        "question": "In what year was AcmeTech founded?",
        "expected_evidence": None,
    },
    {
        "id": "Q9",
        "question": "Where is AcmeTech's headquarters?",
        "expected_evidence": None,
    },
    {
        "id": "Q10",
        "question": "What was AcmeTech's revenue last year?",
        "expected_evidence": None,
    },
]

all_questions = in_domain + out_of_domain


def build_embeddings(**extra_kwargs) -> HuggingFaceEmbeddings:
    kwargs = {
        "model_name": MODEL_PATH,
        "model_kwargs": {"device": "cuda"},
        "encode_kwargs": {"normalize_embeddings": True},
    }
    kwargs.update(extra_kwargs)
    return HuggingFaceEmbeddings(**kwargs)


def run_evaluation(label: str, embeddings: HuggingFaceEmbeddings):
    print(f"\n{'=' * 60}")
    print(f"  Configuration: {label}")
    print(f"  Query encode kwargs: {embeddings.query_encode_kwargs}")
    print(f"  Encode kwargs: {embeddings.encode_kwargs}")
    print(f"{'=' * 60}")

    chroma_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    retriever = chroma_db.as_retriever(search_kwargs={"k": 3})

    results = []
    recall_at_1 = 0
    recall_at_3 = 0
    reciprocal_ranks = []

    for q in all_questions:
        qid = q["id"]
        question = q["question"]
        expected = q["expected_evidence"]

        docs = retriever.invoke(question)
        retrieved = []
        for rank, doc in enumerate(docs, start=1):
            content = doc.page_content[:120].replace("\n", " ")
            source = doc.metadata.get("source", "Unknown")
            retrieved.append(
                {
                    "rank": rank,
                    "content_prefix": content,
                    "source": source,
                    "page_content": doc.page_content,
                }
            )

        # Determine if expected evidence is in results for in-domain questions
        correct_rank = None
        if expected is not None:
            for rank, r in enumerate(docs, start=1):
                if expected.lower() in r.page_content.lower():
                    correct_rank = rank
                    break

        is_in_domain = expected is not None

        if correct_rank is not None:
            recall_at_1 += 1
            recall_at_3 += 1
            reciprocal_ranks.append(1.0 / correct_rank)
        elif is_in_domain:
            # In-domain but not found
            reciprocal_ranks.append(0.0)
            recall_at_3 += 0  # already 0
        else:
            # Out-of-domain — not expected to find anything
            reciprocal_ranks.append(0.0)

        results.append(
            {
                "id": qid,
                "question": question,
                "is_in_domain": is_in_domain,
                "correct_rank": correct_rank,
                "retrieved": retrieved,
            }
        )

        status = "OK" if correct_rank == 1 else (f"@{correct_rank}" if correct_rank else "MISS")
        print(f"  {qid}: {question[:60]:60s} → Rank {correct_rank or '—'} {status}")

    # ── Compute metrics ─────────────────────────────────────────────────────

    # Separate in-domain for recall
    in_domain_results = [r for r in results if r["is_in_domain"]]

    r1 = sum(1 for r in in_domain_results if r["correct_rank"] == 1) / len(in_domain_results)
    r3 = sum(1 for r in in_domain_results if r["correct_rank"] is not None) / len(in_domain_results)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    # In-domain only MRR
    in_domain_rr = [
        1.0 / r["correct_rank"] if r["correct_rank"] is not None else 0.0 for r in in_domain_results
    ]
    mrr_in = sum(in_domain_rr) / len(in_domain_rr) if in_domain_rr else 0.0

    print(f"\n  ── Metrics ({label}) ──")
    print(
        f"  In-domain Recall@1: {r1:.1%}  ({int(r1 * len(in_domain_results))}/{len(in_domain_results)})"
    )
    print(
        f"  In-domain Recall@3: {r3:.1%}  ({int(r3 * len(in_domain_results))}/{len(in_domain_results)})"
    )
    print(f"  MRR (all 10):       {mrr:.3f}")
    print(f"  MRR (in-domain):    {mrr_in:.3f}")

    # Now do similarity_search_with_relevance_scores for score distribution
    print("\n  ── Score Distribution ──")
    for q in all_questions:
        qid = q["id"]
        question = q["question"]
        expected = q["expected_evidence"]

        docs_with_scores = chroma_db.similarity_search_with_relevance_scores(question, k=3)
        print(f"\n  {qid}: {question[:60]}")
        for rank, (doc, score) in enumerate(docs_with_scores, start=1):
            snippet = doc.page_content[:80].replace("\n", " ")
            src = doc.metadata.get("source", "Unknown")
            mark = " <--" if expected and expected.lower() in doc.page_content.lower() else ""
            print(f"    #{rank}  score={score:.4f}  source={src}")
            print(f"          {snippet}{mark}")

    return results, {"recall@1": r1, "recall@3": r3, "mrr": mrr, "mrr_in": mrr_in}


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("DocPilot Retrieval Quality — Diagnostic")
    print("=" * 60)

    # Configuration A: Baseline (no query instruction)
    print("\n>>> Loading Baseline (no prompt_name)...")
    emb_a = build_embeddings()
    results_a, metrics_a = run_evaluation("A — Baseline (no query instruction)", emb_a)

    # Configuration B: With Qwen3 query instruction
    print("\n\n>>> Loading with query prompt...")
    emb_b = build_embeddings(
        query_encode_kwargs={
            "normalize_embeddings": True,
            "prompt_name": "query",
        }
    )
    results_b, metrics_b = run_evaluation("B — With prompt_name='query'", emb_b)

    # ── Comparison ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  Comparison Summary")
    print(f"{'=' * 60}")
    for metric in ["recall@1", "recall@3", "mrr", "mrr_in"]:
        va = metrics_a[metric]
        vb = metrics_b[metric]
        delta = vb - va
        arrow = "+" if delta > 0 else ("-" if delta < 0 else "=")
        print(f"  {metric:12s}:  A={va:.3f}  B={vb:.3f}  Δ={delta:+.3f} {arrow}")
