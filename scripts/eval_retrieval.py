"""DocPilot 离线检索评测脚本。

对 evals/eval_set.jsonl 中的问题跑真实检索（复用 src/agents/tools.py 的
load_chroma_db，与生产同一套初始化，不另起炉灶），计算
Recall@1 / Recall@k / MRR，输出 Markdown 报告。

用法：
    .\\.venv\\Scripts\\python.exe scripts\\eval_retrieval.py
    .\\.venv\\Scripts\\python.exe scripts\\eval_retrieval.py --k 5 --out evals/report.md

说明：
- load_chroma_db 返回的 retriever 把 k=3 写死在 search_kwargs 里；
  --k 非 3 时仅修改该实例的 search_kwargs 属性，不动生产代码。
- out_corpus 题目不计入三个指标，只统计检索返回条数作观察。
- note 含「待标注」的题目从指标计算中排除，并在报告中显式列出。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METRIC_TYPES = ("in_corpus", "multi_hop")


def _load_tools_module():
    """从文件路径加载 src/agents/tools.py，绕开 src/agents/__init__.py 的
    ollama 导入链（与 tests/agents/test_tools_integration.py 同款方式）。"""
    src_dir = os.path.join(REPO_ROOT, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    spec = importlib.util.spec_from_file_location(
        "tools_module_eval",
        os.path.join(src_dir, "agents", "tools.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_eval_set(path: str) -> list[dict]:
    """读取 jsonl 评测集并校验固定字段，非法即退出（不静默跳过）。"""
    required = {"id", "question", "type", "expected_chunk_ids", "note"}
    valid_types = {"in_corpus", "out_corpus", "multi_hop"}
    entries = []
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"评测集第 {lineno} 行不是合法 JSON: {e}")
            missing = required - set(obj)
            if missing:
                raise SystemExit(f"评测集第 {lineno} 行缺字段 {sorted(missing)}")
            if obj["type"] not in valid_types:
                raise SystemExit(f"评测集第 {lineno} 行 type 非法: {obj['type']!r}")
            if (
                obj["type"] in METRIC_TYPES
                and not obj["expected_chunk_ids"]
                and "待标注" not in obj["note"]
            ):
                raise SystemExit(
                    f"评测集第 {lineno} 行（{obj['id']}）是资料内问题但 "
                    "expected_chunk_ids 为空且未标「待标注」——标注缺失，拒绝硬算"
                )
            entries.append(obj)
    if not entries:
        raise SystemExit(f"评测集为空: {path}")
    return entries


def _trunc(text: str, width: int = 50) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def _join_ids(ids: list[str]) -> str:
    return "<br>".join(ids) if ids else "(无)"


def main() -> int:
    parser = argparse.ArgumentParser(description="DocPilot 离线检索评测")
    parser.add_argument(
        "--eval-set",
        default="evals/eval_set.jsonl",
        help="评测集 jsonl 路径（默认 evals/eval_set.jsonl）",
    )
    parser.add_argument("--k", type=int, default=3, help="Top-k（默认 3）")
    parser.add_argument("--out", default=None, help="报告输出路径（默认 evals/report_<时间戳>.md）")
    parser.add_argument(
        "--db-path",
        default=None,
        help="Chroma 库路径（默认 None，即 tools.py 内建的 "
        "./chroma_db_qwen3_semantic_chunks；"
        "设置后通过 CHROMA_DB_PATH 环境变量生效）",
    )
    args = parser.parse_args()

    # --db-path 只改本进程环境变量，tools.py 的 _get_chroma_db_path 读取它。
    # 必须在 _load_tools_module()/load_chroma_db() 之前设置。
    if args.db_path:
        os.environ["CHROMA_DB_PATH"] = args.db_path

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    out_path = args.out or os.path.join(
        "evals", f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    entries = load_eval_set(args.eval_set)

    eligible = [e for e in entries if e["type"] in METRIC_TYPES and "待标注" not in e["note"]]
    pending = [e for e in entries if "待标注" in e["note"]]
    out_corpus = [e for e in entries if e["type"] == "out_corpus" and "待标注" not in e["note"]]

    tools_module = _load_tools_module()
    retriever = tools_module.load_chroma_db()
    if args.k != 3:
        retriever.search_kwargs["k"] = args.k

    # ---- 真实检索：指标题 ----
    results = []
    total = len(eligible) + len(out_corpus)
    for i, e in enumerate(eligible, start=1):
        docs = retriever.invoke(e["question"])
        top_ids = [d.metadata.get("chunk_id", "<缺 chunk_id>") for d in docs]
        expected = e["expected_chunk_ids"]
        hit_rank = next(
            (rank for rank, cid in enumerate(top_ids, start=1) if cid in expected),
            None,
        )
        covered = [c for c in expected if c in top_ids]
        results.append(
            {
                "entry": e,
                "top_ids": top_ids,
                "hit_rank": hit_rank,
                "covered": covered,
            }
        )
        print(
            f"[{i}/{total}] {e['id']} -> {'hit@' + str(hit_rank) if hit_rank else 'MISS'}",
            flush=True,
        )

    # ---- 真实检索：资料外观察题 ----
    observations = []
    for j, e in enumerate(out_corpus, start=len(eligible) + 1):
        docs = retriever.invoke(e["question"])
        top_ids = [d.metadata.get("chunk_id", "<缺 chunk_id>") for d in docs]
        observations.append({"entry": e, "n_retrieved": len(docs), "top_ids": top_ids})
        print(f"[{j}/{total}] {e['id']} (out_corpus) -> {len(docs)} 条返回", flush=True)

    # ---- 指标（按 type 分别计算，不混合平均；同时给总体值） ----
    def _metrics(sub: list[dict]) -> dict:
        """返回一组显式分子/分母指标。

        fc（full-coverage@k）：k 内【全部】期望 chunk 都命中的题数。
        对单期望 chunk 的题（in_corpus 全部、Q14-16 这类单 chunk multi_hop）
        与 recall@k 等价；对双期望 chunk 的 multi_hop（Q17、Q35-40）才是
        真正的「两个都中」口径。
        """
        m = len(sub)
        r1 = sum(1 for r in sub if r["hit_rank"] == 1)
        rk = sum(1 for r in sub if r["hit_rank"] is not None)
        ms = sum(1.0 / r["hit_rank"] for r in sub if r["hit_rank"])
        fc = sum(
            1
            for r in sub
            if r["entry"]["expected_chunk_ids"]
            and len(r["covered"]) == len(r["entry"]["expected_chunk_ids"])
        )
        return {"r1": r1, "rk": rk, "ms": ms, "m": m, "fc": fc}

    by_type: dict[str, list] = {}
    for r in results:
        by_type.setdefault(r["entry"]["type"], []).append(r)

    n = len(results)
    overall = _metrics(results)
    recall1_hits, recallk_hits, mrr_sum = overall["r1"], overall["rk"], overall["ms"]
    recall1 = recall1_hits / n if n else 0.0
    recallk = recallk_hits / n if n else 0.0
    mrr = mrr_sum / n if n else 0.0
    failures = [r for r in results if r["hit_rank"] is None]

    per_type_lines = []
    for t in METRIC_TYPES:
        sub = by_type.get(t, [])
        if not sub:
            continue
        mt = _metrics(sub)
        line = (
            f"{t} (n={mt['m']}): Recall@1 = {mt['r1']}/{mt['m']} = {mt['r1'] / mt['m'] * 100:.1f}%, "
            f"Recall@{args.k} = {mt['rk']}/{mt['m']} = {mt['rk'] / mt['m'] * 100:.1f}%, "
            f"MRR = {mt['ms']:.3f}/{mt['m']} = {mt['ms'] / mt['m']:.3f}"
        )
        if t == "multi_hop":
            line += (
                f", full-coverage@{args.k} = {mt['fc']}/{mt['m']} = {mt['fc'] / mt['m'] * 100:.1f}%"
            )
        per_type_lines.append(line)

    metric_lines = [
        f"Recall@1 = {recall1_hits}/{n} = {recall1 * 100:.1f}%",
        f"Recall@{args.k} = {recallk_hits}/{n} = {recallk * 100:.1f}%",
        f"MRR = {mrr_sum:.3f}/{n} = {mrr:.3f}",
    ]
    print("\n== 指标（分子/分母显式，先总体后分 type） ==")
    for line in metric_lines:
        print(line)
    for line in per_type_lines:
        print(line)
    print(
        f"指标分母 n={n}（in_corpus+multi_hop，已排除待标注 {len(pending)} 条、"
        f"out_corpus {len(out_corpus)} 条）"
    )

    # ---- 语料规模（文档数 / chunk 数，从库 metadata 实查） ----
    db_path = tools_module._get_chroma_db_path()
    import chromadb

    col = chromadb.PersistentClient(path=db_path).get_collection("langchain")
    all_meta = col.get(include=["metadatas"])["metadatas"]
    chunk_total = col.count()
    doc_total = len({m.get("source", "") for m in all_meta})
    L = []
    L.append("# DocPilot 检索评测报告")
    L.append("")
    L.append(
        f"> **边界声明**：本结果仅适用于当前评测集与当前 Chroma 库配置，"
        f"样本量 N={len(entries)}，不构成统计显著性结论，不可跨数据集比较。"
    )
    L.append(">")
    L.append(
        f"> 语料规模：{doc_total} 份文档 / {chunk_total} 个 chunk；k={args.k}。"
        f"混合切分：AcmeTech PDF 用硬编码章节标题切分，9 份 markdown 用显式配置的"
        f" heading 标题集切分（8 份 `##` 级、VertexAI 用 `###` 级）。"
        f"markdown 无页码，其 chunk_id 的 `p1` 为占位符（PDF 的 `p` 是真实页码），"
        f"会影响引用展示。"
    )
    L.append("")
    L.append("## 运行配置")
    L.append("")
    L.append(f"- 评测集: `{args.eval_set}`（共 {len(entries)} 条）")
    L.append(f"- Chroma 库: `{db_path}`")
    L.append(f"- 语料: {doc_total} 份文档 / {chunk_total} 个 chunk")
    L.append(f"- Top-k: {args.k}")
    L.append(f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(
        f"- 计入指标: {n} 条（in_corpus {len(by_type.get('in_corpus', []))} + "
        f"multi_hop {len(by_type.get('multi_hop', []))}）；"
        f"排除: 待标注 {len(pending)} 条、out_corpus {len(out_corpus)} 条（仅观察）"
    )
    L.append("")
    L.append("## 指标汇总")
    L.append("")
    L.append("**总体**（in_corpus + multi_hop）：")
    L.append("")
    L.append("| 指标 | 分子/分母 | 值 |")
    L.append("|---|---|---|")
    L.append(f"| Recall@1 | {recall1_hits}/{n} | {recall1 * 100:.1f}% |")
    L.append(f"| Recall@{args.k} | {recallk_hits}/{n} | {recallk * 100:.1f}% |")
    L.append(f"| MRR | {mrr_sum:.3f}/{n} | {mrr:.3f} |")
    L.append("")
    L.append("**分 type**（不混合平均）：")
    L.append("")
    L.append(
        "| type | n | Recall@1 | Recall@"
        + str(args.k)
        + " | MRR | full-coverage@"
        + str(args.k)
        + " |"
    )
    L.append("|---|---|---|---|---|---|")
    for t in METRIC_TYPES:
        sub = by_type.get(t, [])
        if not sub:
            continue
        mt = _metrics(sub)
        m = mt["m"]
        fc_cell = (
            f"{mt['fc']}/{m} = {mt['fc'] / m * 100:.1f}%"
            if t == "multi_hop"
            else "—（单期望 chunk，同 Recall@k）"
        )
        L.append(
            f"| {t} | {m} | {mt['r1']}/{m} = {mt['r1'] / m * 100:.1f}% | "
            f"{mt['rk']}/{m} = {mt['rk'] / m * 100:.1f}% | "
            f"{mt['ms']:.3f}/{m} = {mt['ms'] / m:.3f} | {fc_cell} |"
        )
    L.append("")
    L.append(
        f"> **口径说明**：Recall@{args.k} 对 multi_hop 的判定是「至少命中一个期望 chunk」，"
        f"与 multi_hop「需跨两 chunk 才能作答」的定义不符，仅供对照；"
        f"full-coverage@{args.k}（k 内全部期望 chunk 命中）才是 multi_hop 的严格口径。"
        f"另外 k={args.k} 对 multi_hop 天然不利——双期望 chunk 需同时挤进前 {args.k} 位，"
        f"两个 type 用同一 k 比较时必须附带此说明。"
    )
    L.append("")
    L.append("## 逐题明细")
    L.append("")
    L.append("| id | 问题 | 期望 chunk_id | 实际 Top-k chunk_id | 命中 | 名次 | 覆盖 |")
    L.append("|---|---|---|---|---|---|---|")
    for r in results:
        e = r["entry"]
        hit = "✅" if r["hit_rank"] else "❌"
        rank = str(r["hit_rank"]) if r["hit_rank"] else "-"
        cov = f"{len(r['covered'])}/{len(e['expected_chunk_ids'])}"
        L.append(
            f"| {e['id']} | {_trunc(e['question'])} | "
            f"{_join_ids(e['expected_chunk_ids'])} | {_join_ids(r['top_ids'])} | "
            f"{hit} | {rank} | {cov} |"
        )
    L.append("")
    L.append("## 失败案例（未命中）")
    L.append("")
    if failures:
        for r in failures:
            e = r["entry"]
            L.append(f"### {e['id']}: {e['question']}")
            L.append("")
            L.append(f"- 期望: `{_join_ids(e['expected_chunk_ids'])}`".replace("<br>", "`, `"))
            L.append(f"- 实际 Top-{args.k}: `{_join_ids(r['top_ids'])}`".replace("<br>", "`, `"))
            L.append(f"- 标注备注: {e['note']}")
            L.append("")
    else:
        L.append("无。所有计入指标的题目均命中。")
        L.append("")
    L.append("## 待标注排除")
    L.append("")
    if pending:
        L.append(f"共 {len(pending)} 条，未计入任何指标：")
        L.append("")
        for e in pending:
            L.append(f"- {e['id']}: {_trunc(e['question'])}")
        L.append("")
    else:
        L.append(f"本次无待标注条目，{len(entries)} 条全部计入各自分组。")
        L.append("")
    L.append("## out_corpus 观察（不计入指标）")
    L.append("")
    L.append("| id | 问题 | 检索返回条数 | Top-1 返回（观察） |")
    L.append("|---|---|---|---|")
    for o in observations:
        e = o["entry"]
        top1 = o["top_ids"][0] if o["top_ids"] else "(无)"
        L.append(f"| {e['id']} | {_trunc(e['question'])} | {o['n_retrieved']} | {top1} |")
    L.append("")

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L))
    print(f"\n报告已写入: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
