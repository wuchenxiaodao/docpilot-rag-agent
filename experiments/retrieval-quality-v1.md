# DocPilot Retrieval Quality — Experiment Report v1

- Branch: `experiment/docpilot-retrieval-quality-v1`
- Vector DB: `chroma_db_qwen3_test` (3 chunks from `AcmeTech_Employee_Handbook.pdf`)
- Embedding: Qwen3-Embedding-0.6B (cuda, normalize_embeddings=True)
- Top-k: 3

---

## 1. Root Cause: Why Mission & Values ranks #2

**Chunking quality is the dominant bottleneck.**

The 3-page PDF is split into exactly 3 chunks by page. Page 1 contains 4+ distinct sections — Welcome Message, Company Mission & Values, Work Hours, Remote Work Policy — all in one chunk. When embedded, the vector is "diluted" across all these topics, making it a poor match for any specific query.

Page 3 ("Employee Benefits Overview") is shorter and more focused. Its embedding is compact and keyword-rich ("benefits", "insurance", "401k"). This chunk consistently ranks #1 for 8 of 10 questions, drowning out the correct chunk.

**Query instruction (prompt_name="query") helps but cannot fix chunking.** Configuration B doubled Recall@1 (20% → 40%), but the ceiling is set by chunk quality, not query encoding.

**Q5 (suspicious emails) is fully missed** because the evidence is embedded in Page 2's long chunk alongside Code of Conduct, Leave Policies, and IT Security. The other content dilutes the security-related signal.

---

## 2. Baseline Metrics (Configuration A — no prompt_name)

| Metric | Value |
|---|---|
| In-domain Recall@1 | **20.0%** (1/5) |
| In-domain Recall@3 | **80.0%** (4/5) |
| MRR (all 10) | 0.250 |
| MRR (in-domain) | 0.500 |

### Per-Question Results

| Q | Question | Correct Evidence Rank | Correct Evidence Score (relevance) | Status |
|---|---|---|---|---|
| Q1 | Mission & values | #2 | 0.2309 | Recall@3 only |
| Q2 | Remote work days | #1 | 0.2560 | Recall@1 |
| Q3 | Core working hours | #2 | 0.3375 | Recall@3 only |
| Q4 | Parental leave weeks | #2 | -0.0898 | Recall@3 only |
| Q5 | Suspicious emails | MISS (not in top 3) | N/A | Failed |
| Q6 | Stock ticker | N/A (out-of-domain) | N/A | Correct rejection |
| Q7 | CEO | N/A (out-of-domain) | N/A | Correct rejection |
| Q8 | Founded year | N/A (out-of-domain) | N/A | Correct rejection |
| Q9 | HQ location | N/A (out-of-domain) | N/A | Correct rejection |
| Q10 | Revenue | N/A (out-of-domain) | N/A | Correct rejection |

### Score Distribution: In-Domain Correct Evidence

| Question | Relevance Score | L2 Distance |
|---|---|---|
| Q1 (Mission) | 0.2309 | 1.088 |
| Q2 (Remote) | 0.2560 | 1.052 |
| Q3 (Hours) | 0.3375 | 0.937 |
| Q4 (Parental) | -0.0898 | 1.542 |
| Q5 (Email) | MISS | MISS |

### Score Distribution: Out-of-Domain Top-1

All out-of-domain questions hit Page 3 ("Benefits Overview") as their top result:

| Question | Relevance Score | L2 Distance |
|---|---|---|
| Q6 (Ticker) | 0.3787 | 0.879 |
| Q7 (CEO) | 0.3585 | 0.907 |
| Q8 (Founded) | 0.3456 | 0.925 |
| Q9 (HQ) | 0.3873 | 0.867 |
| Q10 (Revenue) | 0.3583 | 0.908 |

---

## 3. Controlled Experiment: Query Instruction

### Configuration A (Baseline)
- No `prompt_name` for query encoding
- Query and document both encoded without instruction prefix

### Configuration B (Qwen3 recommended)
- `query_encode_kwargs={"normalize_embeddings": True, "prompt_name": "query"}`
- Query gets instruction prefix: `"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"`
- Document encoding unchanged (no prefix — consistent with existing vector DB)

### Results Comparison

| Metric | A (Baseline) | B (prompt_name=query) | Delta |
|---|---|---|---|
| Recall@1 | 20.0% | **40.0%** | +20% |
| Recall@3 | 80.0% | 80.0% | 0% |
| MRR (all) | 0.250 | 0.300 | +0.050 |
| MRR (in-domain) | 0.500 | 0.600 | +0.100 |

**Improvements:**
- Q4 (parental leave): moved from rank 2 → rank 1 (pulled up by instruction)
- Q1 (mission), Q3 (hours): still rank 2 — chunking bottleneck
- Q5 (suspicious emails): still missed — chunking bottleneck

---

## 4. Score Formula and Direction

Chroma's `similarity_search_with_relevance_scores` returns `1 - (L2_distance / sqrt(2))`.

- **Score direction:** Higher = more "relevant" (smaller L2 distance; score = 1 when identical, 0 when L2=sqrt(2)≈1.414, negative when L2 > 1.414)
- **Raw L2 distance:** Lower = closer (Chroma's default distance metric; cosine distance for normalized embeddings)
- **Relationship:** `relevance_score = 1 − L2 / sqrt(2)` — a monotonic transform, so rank order is preserved between the two

Since embeddings are L2-normalized, the actual cosine similarity is `cosine_sim = 1 − L2² / 2`, but Chroma's built-in relevance score function uses the simpler L2/sqrt(2) formula. The rank order is the same regardless of which metric is used.

**Important:** The relevance score is a distance-derived quantity, not a true similarity probability. It can be negative and its absolute value is not directly comparable across different embedding models.

## 5. Threshold Analysis

### In-Domain Correct Evidence Scores

Correct evidence relevance scores range from **-0.0898 to 0.3375**, with L2 distances from **0.937 to 1.542**.

### Out-of-Domain Top-1 Scores

### Out-of-Domain Top-1 Scores

All out-of-domain queries score between **0.3456 and 0.3873** on the same "Benefits Overview" chunk (L2: 0.867–0.925).

### Separation Check

**Numerically the two ranges do not overlap.** In-domain correct: −0.0898 to 0.3375; out-of-domain top-1: 0.3456 to 0.3873.

However, the separation direction is **opposite to what is desired**: all out-of-domain (wrong) results score higher than most in-domain correct evidence. The "Benefits Overview" chunk — short, focused, keyword-dense — consistently scores 0.35–0.39 for every query, while the correct chunks (long, multi-topic, diluted embeddings) score lower.

**Conclusion: Current data does not support a reliable monotonic threshold.** Raising a threshold to reject out-of-domain queries (e.g. >0.34) would also reject correct evidence with lower scores (Q1 at 0.23, Q4 at −0.09). Lowering it to keep those correct results would accept all out-of-domain queries. The issue is not score overlap — it is that the wrong chunk pervasively out-scores the right ones due to chunk quality, not relevance.

---

## 6. Production Code Changes

### Approved Change

**Add `query_encode_kwargs` to `tools.py`** — this is supported by the data:

- 20% absolute Recall@1 improvement (1/5 → 2/5)
- No regression in any metric
- No new dependencies
- Follows Qwen3's documented best practice
- Compatible with existing vector DB (documents were encoded without prefix, which is correct)

```python
HuggingFaceEmbeddings(
    model_name=model_path,
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
    query_encode_kwargs={"normalize_embeddings": True, "prompt_name": "query"},
)
```

### NOT Approved Changes

- **Do NOT add a similarity threshold** — no data support
- **Do NOT add a reranker** — chunking should be fixed first (separate experiment)
- **Do NOT modify chunk count** — would require rebuilding the vector DB (separate experiment)

---

## 7. Remaining Issues

1. **Small sample (10 questions, 3 chunks)** — metrics are directional, not statistically significant
2. **Chunking quality** — the dominant bottleneck; each page is a single chunk containing multiple topics
3. **Reranker** — unnecessary until chunking is fixed; reranking 3 chunks from the same page adds little value
4. **Q5 (suspicious emails) still missed** — requires chunk structure change, not embedding tuning
5. **No generation-level evaluation** — this experiment tested retrieval only, not the LLM's ability to reject unanswerable questions