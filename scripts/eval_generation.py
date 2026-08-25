"""DocPilot 端到端生成质量评测脚本（忠实度 / 引用正确性 / 拒答准确率）。

对 evals/eval_set.jsonl 的题目调用 rag-assistant 图（与生产同一链路：
guard→model→tools→guard_retrieval→collect_citations→moderate_output），
对生成结果做三维度评估：

1. **忠实度（Faithfulness）** — LLM-as-judge（复用本地 Qwen，无新依赖；
   同模型自评有偏，报告已注明，仅作相对基线）。三级：
   supported / partial / unsupported，仅对作答题判定。--skip-judge 可跳过。

2. **引用正确性（两级机械判定）**
   - chunk 级（主指标）：作答题的 state.citations chunk_id 与标注
     expected_chunk_ids 有交集（引对了证据）；
   - 来源级（辅助）：解析回答末尾的 Sources: 段，逐条检查被引来源名
     是否出现在检索工具返回中（没有捏造文件名）。

3. **拒答准确率（Refusal accuracy）** — 与 eval_refusal.py 同判定标记，但
   区分标签状态：truly_out（真正库外 8 题，2026-08-23 拒答分析确认，
   evals/refusal_report_20260823_122841.md）应拒答；stale_out（Q41-Q45/
   Q47/Q48，v2 库实际可答、标签过期）只观察作答率不计入准确率；
   in_corpus/multi_hop 应作答。

用法：
    env -u SSL_CERT_FILE .venv/Scripts/python.exe scripts/eval_generation.py
    # --ids Q1,Q39   只跑指定题；--limit N 按 id 排序取前 N；--filter TYPE 同型过滤
    # --skip-judge   跳过 LLM 忠实度评判（只做机械检查，快）
    # --timeout S    单题整体超时（默认 900s，防单题卡死拖垮整轮）
    # --out PATH     报告路径（默认 evals/report_generation_quality_<时间戳>.md）
    # 明细逐题落盘 evals/generation_quality_details_<时间戳>.jsonl（跑挂可留档）

前置：本地推理服务（Ollama）已启动且 .env 指向它。逐题串行（单 GPU）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv  # noqa: E402

# tools.py 直接读 os.environ（不走 pydantic Settings），必须先把仓库 .env 装进环境变量。
load_dotenv(os.path.join(REPO_ROOT, ".env"))

sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

# 2026-08-23 拒答分析确认的真正库外题；其余 out_corpus（Q41-Q45/Q47/Q48）
# 在 v2 语料下实际可答（ingest 后入库），标签过期，只观察不计分。
TRULY_OUT = {"Q6", "Q7", "Q8", "Q9", "Q10", "Q46", "Q49", "Q50"}

# ---------- refusal detection (same markers as eval_refusal.py) ----------

REFUSAL_PATTERNS = [
    r"couldn'?t find sufficient evidence",
    r"could not find sufficient evidence",
    r"insufficient evidence",
    r"sources:\s*none",
]
_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)


def _is_refusal(answer: str, tool_called: bool) -> bool:
    if _REFUSAL_RE.search(answer):
        return True
    return not tool_called


def expected_refusal(q: dict) -> bool | None:
    """拒绝期望：True=应拒答，False=应作答，None=stale 标签（不计分）。"""
    if q["type"] == "out_corpus":
        return True if q["id"] in TRULY_OUT else None
    return False


# ---------- citation: prose 'Sources:' block ----------

# "Sources:" 起到字符串结尾的整段（DOTALL 让 . 跨行；\s* 兼容
# "Sources:\n- a.pdf" 与同行 "Sources: None" 两种写法）。
_SOURCES_BLOCK_RE = re.compile(r"Sources:\s*(.*)", re.DOTALL)


def parse_cited_sources(answer: str) -> list[str]:
    """Extract source identifiers from the answer's 'Sources:' section.

    Returns a deduplicated list of source name strings (e.g. filenames).
    Per line: strip list bullets, take the first whitespace-separated token,
    drop trailing punctuation/parenthesis noise (e.g. "(page 2)" annotations).
    """
    m = _SOURCES_BLOCK_RE.search(answer)
    if not m:
        return []
    block = m.group(1)
    cleaned: list[str] = []
    seen: set[str] = set()
    for line in block.splitlines():
        tokens = line.strip().lstrip("-*").strip().split()
        if not tokens:
            continue
        name = tokens[0].strip("():*，。")
        if name and name not in seen:
            seen.add(name)
            cleaned.append(name)
    return cleaned


def check_citation_correctness(
    cited_sources: list[str], tool_output: str
) -> tuple[int, int]:
    """Count how many cited sources appear in the retrieved tool output.

    Returns (correct, total). A source is "correct" if its name (case-insensitive)
    is found as a substring of the tool output text.
    """
    if not cited_sources:
        return 0, 0
    output_lower = tool_output.lower()
    correct = 0
    for src in cited_sources:
        if src.lower() in output_lower:
            correct += 1
    return correct, len(cited_sources)


# ---------- message / result extraction ----------

_CITATIONS_TAIL_RE = re.compile(r"CITATIONS_JSON: \[.*\]\s*$", re.DOTALL)


def _extract_tool_output(messages: list) -> str:
    """Get the text content of the last ToolMessage in the message list."""
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "tool":
            return str(msg.content)
    return ""


def _extract_last_answer(messages: list) -> str:
    """Get the text content of the last AIMessage with content."""
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "ai" and msg.content:
            return str(msg.content)
    return ""


def extract_result(result: dict) -> tuple[str, bool, list[str], str]:
    """从图执行结果取 (answer, tool_called, cited_chunk_ids, retrieved_context)。

    retrieved_context 为去掉尾部 CITATIONS_JSON 行的最后一个 ToolMessage。
    """
    msgs = result["messages"]
    tool_called = any(getattr(m, "type", None) == "tool" for m in msgs)
    answer = _extract_last_answer(msgs)

    citations = result.get("citations") or []
    cited_ids = [c.get("chunk_id") for c in citations if c.get("chunk_id")]

    context = ""
    for m in reversed(msgs):
        if getattr(m, "type", None) == "tool":
            context = _CITATIONS_TAIL_RE.sub("", str(m.content)).strip()
            break
    return answer, tool_called, cited_ids, context


# ---------- faithfulness judge (LLM-as-judge, local Qwen) ----------

_JUDGE_SYSTEM = """\
You are a strict evaluator for a retrieval-augmented QA system. \
Given the QUESTION, the RETRIEVED CONTEXT (the sole permissible evidence), \
and the ANSWER, judge the factual faithfulness of the ANSWER.

Verdict rules:
- "supported": every factual claim in the ANSWER is directly stated in or \
clearly entailed by the RETRIEVED CONTEXT.
- "partial": the ANSWER contains some supported claims plus at least one \
unsupported or extrapolated claim.
- "unsupported": the ANSWER's key claims are absent from or contradicted by \
the RETRIEVED CONTEXT.

Do not penalize style, brevity, or formatting. Do not use outside knowledge — \
only compare the ANSWER against the CONTEXT.
Respond with JSON only: \
{"faithfulness": "supported"|"partial"|"unsupported", "reason": "<short reason>"}\
"""


def parse_judge_json(text: str) -> dict | None:
    """Extract the judge's JSON object from raw model output.

    Tolerates markdown fences and surrounding prose; returns None if no
    parseable JSON object is found.
    """
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


async def judge_faithfulness(
    question: str, context: str, answer: str, model
) -> dict[str, str]:
    """Ask the LLM to judge faithfulness of *answer* given *context*.

    Returns {"faithfulness": ..., "reasoning": ...} or {"faithfulness": "unparsed", ...}.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    prompt = (
        f"QUESTION: {question}\n\n"
        f"RETRIEVED CONTEXT:\n{context[:8000]}\n\n"
        f"ANSWER:\n{answer[:4000]}\n\n"
        "Respond with JSON only."
    )
    try:
        resp = await model.ainvoke(
            [SystemMessage(content=_JUDGE_SYSTEM), HumanMessage(content=prompt)]
        )
    except Exception as e:  # noqa: BLE001
        return {"faithfulness": "unparsed", "reasoning": str(e)[:200]}
    text = str(resp.content)
    data = parse_judge_json(text)
    if data is None:
        return {"faithfulness": "unparsed", "reasoning": text[:200].replace("\n", " ")}
    return {
        "faithfulness": str(data.get("faithfulness", "unparsed")).lower(),
        "reasoning": str(data.get("reason", ""))[:200],
    }


# ---------- run one case through the rag graph ----------


async def _run_case(
    agent, q: dict, judge_model=None, timeout_s: int = 900
) -> dict:
    """Run a single eval case: invoke rag-assistant, score, optionally judge."""
    from langchain_core.messages import HumanMessage
    from langchain_core.runnables import RunnableConfig

    row: dict = {"id": q["id"], "type": q["type"], "truly_out": q["id"] in TRULY_OUT}
    t0 = time.perf_counter()
    result = await asyncio.wait_for(
        agent.ainvoke(
            {"messages": [HumanMessage(q["question"])]},
            config=RunnableConfig(configurable={"thread_id": f"eval-gen-{q['id']}"}),
        ),
        timeout=timeout_s,
    )
    row["latency_s"] = round(time.perf_counter() - t0, 1)

    answer, tool_called, cited_ids, context = extract_result(result)
    refused = _is_refusal(answer, tool_called)
    expect = expected_refusal(q)
    row.update(
        {
            "question": q["question"],
            "tool_called": tool_called,
            "refused": refused,
            "refusal_correct": refused == expect if expect is not None else None,
            "cited_chunk_ids": cited_ids,
            "answer_head": answer[:160].replace("\n", " "),
        }
    )

    # chunk 级引用命中：作答且题目标注了期望 chunk 才计分
    expected = q.get("expected_chunk_ids") or []
    row["citation_hit"] = (
        bool(set(cited_ids) & set(expected)) if expected and not refused else None
    )

    # 来源级引用检查：文内 Sources: 段的被引名字须出现在检索返回里
    cited_names = parse_cited_sources(answer)
    tool_output = _extract_tool_output(result["messages"])
    row["cited_sources"] = cited_names
    row["citation_name_correct"], row["citation_name_total"] = (
        check_citation_correctness(cited_names, tool_output)
    )

    # 忠实度（LLM 判官）：仅作答且有检索上下文
    if judge_model is None or refused or not context or not answer:
        row["faithfulness"] = None
        row["faithfulness_reasoning"] = ""
    else:
        verdict = await judge_faithfulness(q["question"], context, answer, judge_model)
        row["faithfulness"] = verdict["faithfulness"]
        row["faithfulness_reasoning"] = verdict["reasoning"]
    return row


# ---------- report ----------


def _write_report(results: list[dict], out_path: str, skip_judge: bool) -> None:
    from core.settings import settings

    model = settings.COMPATIBLE_MODEL or settings.DEFAULT_MODEL
    ok = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    truly_out = [r for r in ok if r.get("truly_out")]
    in_multi = [r for r in ok if r["type"] in ("in_corpus", "multi_hop")]
    stale = [r for r in ok if r["type"] == "out_corpus" and not r.get("truly_out")]

    to_refused = sum(1 for r in truly_out if r["refused"])
    im_answered = sum(1 for r in in_multi if not r["refused"])
    stale_answered = sum(1 for r in stale if not r["refused"])

    chunk_rows = [r for r in in_multi if r.get("citation_hit") is not None]
    chunk_ok = sum(1 for r in chunk_rows if r["citation_hit"])

    name_rows = [r for r in ok if r.get("citation_name_total")]
    name_ok = sum(1 for r in name_rows if r["citation_name_correct"] == r["citation_name_total"])

    judged = [r for r in ok if r.get("faithfulness") is not None]
    sup = sum(1 for r in judged if r["faithfulness"] == "supported")
    par = sum(1 for r in judged if r["faithfulness"] == "partial")
    unsup = sum(1 for r in judged if r["faithfulness"] == "unsupported")
    unparsed = len(judged) - sup - par - unsup

    def ratio(num: int, den: int) -> str:
        return f"{num / den:.1%}" if den else "-"

    lines = [
        "# DocPilot 端到端生成质量评测报告",
        "",
        f"- 生成模型: `{model}`（DEFAULT_MODEL={settings.DEFAULT_MODEL}）",
        f"- 评测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 链路: rag-assistant 图全流程，逐题串行",
        f"- 拒答判定标记: {' / '.join(REFUSAL_PATTERNS)}（完全没调用工具也视为拒答）",
        f"- 忠实度判官: 本地生成模型自评（无新依赖；同模型自评有偏，仅作相对基线）"
        + ("；本轮 --skip-judge 已跳过" if skip_judge else ""),
        "",
        "## 指标汇总",
        "",
        "| 维度 | 分组 | 分子/分母 | 值 |",
        "|---|---|---|---|",
        f"| 拒答准确率 | truly_out 应拒答（{len(truly_out)} 题） | {to_refused}/{len(truly_out)} | {ratio(to_refused, len(truly_out))} |",
        f"| 拒答准确率 | in_corpus+multi_hop 应作答 | {im_answered}/{len(in_multi)} | {ratio(im_answered, len(in_multi))} |",
        f"| 引用正确性 | 作答题命中期望 chunk | {chunk_ok}/{len(chunk_rows)} | {ratio(chunk_ok, len(chunk_rows))} |",
        f"| 引用正确性 | 文内 Sources 无捏造来源 | {name_ok}/{len(name_rows)} | {ratio(name_ok, len(name_rows))} |",
        f"| 忠实度 | supported | {sup}/{len(judged)} | {ratio(sup, len(judged))} |",
    ]
    if judged:
        lines += [
            f"| 忠实度 | partial | {par}/{len(judged)} | {ratio(par, len(judged))} |",
            f"| 忠实度 | unsupported | {unsup}/{len(judged)} | {ratio(unsup, len(judged))} |",
        ]
    if stale:
        lines.append(
            f"| 观察（不计分） | stale_out（v2 实际可答）作答 | {stale_answered}/{len(stale)} | {ratio(stale_answered, len(stale))} |"
        )
    lines.append("")
    if unparsed:
        lines.append(f"判官输出未解析 {unparsed} 例，计入分母但不计入 supported。")
        lines.append("")

    lines += [
        "## 逐题明细",
        "",
        "| id | type | 拒答 | 引用chunk | 引用来源 | 忠实度 | 延迟(s) | 回答摘录 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['id']} | {r['type']} | - | - | - | - | - | ERROR: {r['error']} |")
            continue
        ref = "✅" if r.get("refusal_correct") else ("❌" if r.get("refusal_correct") is not None else "·")
        cit = {True: "✅", False: "❌", None: "·"}[r.get("citation_hit")]
        names = (
            f"{r['citation_name_correct']}/{r['citation_name_total']}"
            if r.get("citation_name_total")
            else "-"
        )
        faith = r.get("faithfulness") or "·"
        lines.append(
            f"| {r['id']} | {r['type']}{' (truly_out)' if r.get('truly_out') else ''} "
            f"| {ref} | {cit} | {names} | {faith} | {r['latency_s']} | {r['answer_head']} |"
        )

    if errors:
        lines += ["", f"错误 {len(errors)} 例已从分母排除，见上表 ERROR 行。"]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------- main ----------


async def _main_async(args: argparse.Namespace) -> int:
    from agents import get_agent
    from core import get_model, settings

    with open(os.path.join(REPO_ROOT, "evals", "eval_set.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
        rows = [r for r in rows if r["id"] in wanted]
    if args.filter:
        rows = [r for r in rows if r["type"] == args.filter]
    rows.sort(key=lambda r: r["id"])
    if args.limit:
        rows = rows[: args.limit]
    print(f"共 {len(rows)} 题，judge={'off' if args.skip_judge else 'on'}，单题超时 {args.timeout}s", flush=True)

    judge_model = None
    if not args.skip_judge:
        try:
            judge_model = get_model(settings.DEFAULT_MODEL)
        except Exception as e:  # noqa: BLE001
            print(f"判官模型加载失败，跳过忠实度评判: {e}", flush=True)

    agent = get_agent("rag-assistant")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    details_path = os.path.join(REPO_ROOT, "evals", f"generation_quality_details_{stamp}.jsonl")
    out_path = args.out or os.path.join(
        REPO_ROOT, "evals", f"report_generation_quality_{stamp}.md"
    )

    results: list[dict] = []
    with open(details_path, "w", encoding="utf-8") as details:
        for q in rows:
            try:
                r = await _run_case(agent, q, judge_model=judge_model, timeout_s=args.timeout)
            except asyncio.TimeoutError:
                r = {"id": q["id"], "type": q["type"], "error": f"timeout>{args.timeout}s"}
            except Exception as e:  # noqa: BLE001
                r = {"id": q["id"], "type": q["type"], "error": str(e)[:200]}
            results.append(r)
            details.write(json.dumps(r, ensure_ascii=False) + "\n")
            details.flush()
            if "error" in r:
                print(f"{r['id']} [{r['type']}] ERROR: {r['error']}", flush=True)
            else:
                faith = r.get("faithfulness") or "-"
                print(
                    f"{r['id']} [{r['type']}] ref={'✓' if r['refusal_correct'] else ('✗' if r['refusal_correct'] is not None else '·')} "
                    f"cit={r.get('citation_hit')} names={r.get('citation_name_correct')}/{r.get('citation_name_total')} "
                    f"faith={faith} ({r['latency_s']}s)",
                    flush=True,
                )

    _write_report(results, out_path, args.skip_judge)
    print(f"\n报告已写入: {out_path}\n明细已写入: {details_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="DocPilot 端到端生成质量评测")
    parser.add_argument("--ids", default=None, help="逗号分隔的题目 id，如 Q1,Q39")
    parser.add_argument("--limit", type=int, default=0, help="按 id 排序取前 N 题（0=全部）")
    parser.add_argument("--filter", default=None, choices=["in_corpus", "out_corpus", "multi_hop"], help="只跑指定类型")
    parser.add_argument("--skip-judge", action="store_true", help="跳过 LLM 忠实度评判")
    parser.add_argument("--timeout", type=int, default=900, help="单题整体超时秒数")
    parser.add_argument("--out", default=None, help="报告输出路径")
    args = parser.parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
