"""DocPilot 检索分数分布分析（roadmap #9 设计前置）。

复用 HybridRetriever 的融合逻辑，但额外保留每个 chunk 的 RRF 分数，
用于判断「绝对 Top-1 分数」或「Top-1 与 Top-2 的 margin」能否区分
资料内（in_corpus + multi_hop）与资料外（out_corpus）问题——即
检索层拒答阈值是否可行。

用法：
    .\\.venv\\Scripts\\python.exe scripts\\analyze_score_distribution.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections import defaultdict
from statistics import mean, median

from dotenv import load_dotenv

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(REPO_ROOT, ".env"))


def _load_tools_module():
    src_dir = os.path.join(REPO_ROOT, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    spec = importlib.util.spec_from_file_location(
        "tools_module_analysis",
        os.path.join(src_dir, "agents", "tools.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fuse_with_scores(retriever, query: str) -> list[tuple[str, float]]:
    """复刻 _retrieve_one 的融合，但返回 (chunk_id, score) 而非丢掉分数。"""
    vector_hits = retriever._store.similarity_search(query, k=retriever._candidate_k)
    index = retriever._ensure_index()
    bm25_ids = index.search(query, retriever._candidate_k)

    fused: dict[str, float] = {}
    for rank, d in enumerate(vector_hits):
        cid = retriever._cid(d, f"vector-{rank}")
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (retriever._rrf_k + rank + 1)
    for rank, cid in enumerate(bm25_ids):
        fused[cid] = fused.get(cid, 0.0) + retriever._bm25_weight / (retriever._rrf_k + rank + 1)

    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    tools = _load_tools_module()
    retriever = tools.load_chroma_db()
    # 处理多意图拆分：与生产 invoke 一致地拆分后对各子查询取分数。
    # 这里取【原查询】的融合分数做分布分析——拒答阈值判的是单次检索
    # 的置信度，多意图拆分后应分别判各子查询（留待实现时再定）。
    # 先看原查询分数分布能否区分。

    entries = []
    with open(os.path.join(REPO_ROOT, "evals", "eval_set.jsonl"), encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            entries.append(json.loads(line))

    by_type: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        if "待标注" in e["note"]:
            continue
        ranked = _fuse_with_scores(retriever, e["question"])
        if not ranked:
            continue
        top1_cid, top1_score = ranked[0]
        top2_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top1_score - top2_score
        by_type[e["type"]].append(
            {
                "id": e["id"],
                "top1": top1_score,
                "top2": top2_score,
                "margin": margin,
                "top1_cid": top1_cid,
                "expected": e.get("expected_chunk_ids", []),
                "hit": top1_cid in e.get("expected_chunk_ids", []),
            }
        )

    print("=" * 80)
    print("检索分数分布（post-#7/#8，RRF 融合分数）")
    print("=" * 80)

    def _stats(vals: list[float]) -> str:
        if not vals:
            return "(空)"
        return (
            f"min={min(vals):.4f} max={max(vals):.4f} "
            f"mean={mean(vals):.4f} median={median(vals):.4f}"
        )

    in_scores = [r["top1"] for r in by_type["in_corpus"]] + [
        r["top1"] for r in by_type["multi_hop"]
    ]
    out_scores = [r["top1"] for r in by_type["out_corpus"]]
    in_margins = [r["margin"] for r in by_type["in_corpus"]] + [
        r["margin"] for r in by_type["multi_hop"]
    ]
    out_margins = [r["margin"] for r in by_type["out_corpus"]]

    print(f"\n【Top-1 绝对分数】")
    print(f"  资料内 (in+multi, n={len(in_scores)}): {_stats(in_scores)}")
    print(f"  资料外 (out,        n={len(out_scores)}): {_stats(out_scores)}")
    print(f"  资料内 min={min(in_scores):.4f}, 资料外 max={max(out_scores):.4f}")
    overlap = min(in_scores) <= max(out_scores)
    print(f"  → 分布{'重叠' if overlap else '不重叠'}，绝对阈值{'不可行' if overlap else '可行'}")

    print(f"\n【Top-1 与 Top-2 的 margin】")
    print(f"  资料内 (in+multi, n={len(in_margins)}): {_stats(in_margins)}")
    print(f"  资料外 (out,        n={len(out_margins)}): {_stats(out_margins)}")
    print(f"  资料内 min margin={min(in_margins):.4f}, 资料外 max margin={max(out_margins):.4f}")
    margin_overlap = min(in_margins) <= max(out_margins)
    print(f"  → 分布{'重叠' if margin_overlap else '不重叠'}，margin 阈值{'不可行' if margin_overlap else '可行'}")

    # 逐题明细，按 top1 分数排序
    print(f"\n【逐题明细（按 Top-1 分数升序）】")
    print(f"{'id':<6}{'type':<12}{'top1':>8}{'margin':>9}{'hit':>5}  question")
    all_rows = []
    for t, rows in by_type.items():
        for r in rows:
            all_rows.append((t, r))
    all_rows.sort(key=lambda x: x[1]["top1"])
    for t, r in all_rows:
        q = next(e["question"] for e in entries if e["id"] == r["id"])[:40]
        hit_mark = "✓" if r["hit"] else "✗"
        print(f"{r['id']:<6}{t:<12}{r['top1']:>8.4f}{r['margin']:>9.4f}{hit_mark:>5}  {q}")

    # 找最佳分离阈值（绝对分数）
    print(f"\n【最佳绝对阈值扫描】")
    all_top1 = sorted(set(in_scores + out_scores))
    best_t = None
    best_f1 = -1.0
    for t in all_top1:
        # 阈值 t：score < t 判为拒答（资料外），score >= t 判为作答（资料内）
        tp = sum(1 for s in in_scores if s >= t)  # 资料内判作答（正确）
        fn = sum(1 for s in in_scores if s < t)  # 资料内判拒答（错误，漏答）
        fp = sum(1 for s in out_scores if s >= t)  # 资料外判作答（错误，误答）
        tn = sum(1 for s in out_scores if s < t)  # 资料外判拒答（正确）
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_t = (t, tp, fn, fp, tn, prec, rec, f1)
    if best_t:
        t, tp, fn, fp, tn, prec, rec, f1 = best_t
        print(
            f"  最佳阈值 top1<{t:.4f} 拒答: "
            f"TP(内作答)={tp} FN(内漏答)={fn} FP(外误答)={fp} TN(外拒答)={tn} "
            f"prec={prec:.2f} rec={rec:.2f} f1={f1:.2f}"
        )

    print(f"\n【最佳 margin 阈值扫描】")
    all_margins = sorted(set(in_margins + out_margins))
    best_mt = None
    best_mf1 = -1.0
    for t in all_margins:
        tp = sum(1 for s in in_margins if s >= t)
        fn = sum(1 for s in in_margins if s < t)
        fp = sum(1 for s in out_margins if s >= t)
        tn = sum(1 for s in out_margins if s < t)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if f1 > best_mf1:
            best_mf1 = f1
            best_mt = (t, tp, fn, fp, tn, prec, rec, f1)
    if best_mt:
        t, tp, fn, fp, tn, prec, rec, f1 = best_mt
        print(
            f"  最佳阈值 margin<{t:.4f} 拒答: "
            f"TP(内作答)={tp} FN(内漏答)={fn} FP(外误答)={fp} TN(外拒答)={tn} "
            f"prec={prec:.2f} rec={rec:.2f} f1={f1:.2f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
