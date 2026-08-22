"""DocPilot 拒答正确率评测脚本（生成层）。

对 evals/eval_set.jsonl 的题目直接调用 rag-assistant 图（与生产同一链路：
load_dotenv → get_agent → ainvoke），按系统提示词的约定判定：
- out_corpus 题**应拒答**：未调用检索工具，或回答含拒答标记
- in_corpus 题应作答：调用了检索工具且回答不含拒答标记

判定标记（rag_assistant.py 系统提示词要求的固定说法及其常见变体）：
"couldn't find sufficient evidence" / "insufficient evidence" / "Sources: None"。
不在标记内的回答视为「作答」；完全没调用工具也视为「拒答」。

用法：
    env -u SSL_CERT_FILE .\\.venv\\Scripts\\python.exe scripts\\eval_refusal.py
    # --in-corpus N 控制资料内抽题数（默认 10，按 id 排序取前 N）
    # --out 指定报告路径（默认 evals/refusal_report_<时间戳>.md，被 gitignore）

前置：本地推理服务（Ollama）已启动且 .env 指向它。
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

load_dotenv(os.path.join(REPO_ROOT, ".env"))

sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

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


async def _run_case(agent, q: dict) -> dict:
    from langchain_core.messages import HumanMessage
    from langchain_core.runnables import RunnableConfig

    t0 = time.perf_counter()
    result = await agent.ainvoke(
        {"messages": [HumanMessage(q["question"])]},
        config=RunnableConfig(configurable={"thread_id": f"eval-refusal-{q['id']}"}),
    )
    latency = time.perf_counter() - t0

    msgs = result["messages"]
    tool_called = any(m.type == "tool" for m in msgs)
    last_ai = next((m for m in reversed(msgs) if m.type == "ai" and m.content), None)
    answer = str(last_ai.content) if last_ai else ""

    refused = _is_refusal(answer, tool_called)
    expected_refusal = q["type"] == "out_corpus"
    return {
        "id": q["id"],
        "question": q["question"],
        "type": q["type"],
        "tool_called": tool_called,
        "refused": refused,
        "correct": refused == expected_refusal,
        "latency_s": round(latency, 1),
        "answer_head": answer[:160].replace("\n", " "),
    }


def _write_report(results: list[dict], out_path: str) -> None:
    from core.settings import settings

    model = settings.COMPATIBLE_MODEL or settings.DEFAULT_MODEL
    out_n = [r for r in results if r["type"] == "out_corpus" and "error" not in r]
    in_n = [r for r in results if r["type"] == "in_corpus" and "error" not in r]
    errors = [r for r in results if "error" in r]
    out_ok = sum(1 for r in out_n if r["correct"])
    in_ok = sum(1 for r in in_n if r["correct"])

    lines = [
        "# DocPilot 拒答正确率报告（生成层）",
        "",
        f"- 生成模型: `{model}`（DEFAULT_MODEL={settings.DEFAULT_MODEL}）",
        f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 判定标记: {' / '.join(REFUSAL_PATTERNS)}",
        "",
        "## 指标汇总",
        "",
        "| 分组 | 正确行为 | 分子/分母 | 值 |",
        "|---|---|---|---|",
        f"| out_corpus | 拒答 | {out_ok}/{len(out_n)} | {out_ok / len(out_n):.1%} |" if out_n else "| out_corpus | 拒答 | 0/0 | - |",
        f"| in_corpus | 作答 | {in_ok}/{len(in_n)} | {in_ok / len(in_n):.1%} |" if in_n else "| in_corpus | 作答 | 0/0 | - |",
        "",
        "## 逐题明细",
        "",
        "| id | type | 期望 | 工具调用 | 实际 | 判定 | 延迟(s) | 回答摘录 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['id']} | {r['type']} | - | - | ERROR | ✗ | - | {r['error']} |")
            continue
        expect = "拒答" if r["type"] == "out_corpus" else "作答"
        actual = "拒答" if r["refused"] else "作答"
        lines.append(
            f"| {r['id']} | {r['type']} | {expect} | {'✓' if r['tool_called'] else '✗'} "
            f"| {actual} | {'✅' if r['correct'] else '❌'} | {r['latency_s']} | {r['answer_head']} |"
        )
    if errors:
        lines += ["", f"错误 {len(errors)} 例已从分母排除，见上表 ERROR 行。"]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


async def _main_async(args: argparse.Namespace) -> int:
    from agents import get_agent

    with open(os.path.join(REPO_ROOT, "evals", "eval_set.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    out_corpus = sorted((r for r in rows if r["type"] == "out_corpus"), key=lambda r: r["id"])
    in_corpus = sorted((r for r in rows if r["type"] == "in_corpus"), key=lambda r: r["id"])[
        : args.in_corpus
    ]
    print(f"out_corpus {len(out_corpus)} 题 + in_corpus {len(in_corpus)} 题", flush=True)

    agent = get_agent("rag-assistant")
    results: list[dict] = []
    for q in out_corpus + in_corpus:
        try:
            r = await _run_case(agent, q)
        except Exception as e:  # noqa: BLE001
            r = {"id": q["id"], "type": q["type"], "error": str(e)[:200]}
        results.append(r)
        if "error" in r:
            print(f"{r['id']} [{r['type']}] ERROR: {r['error']}", flush=True)
        else:
            print(f"{r['id']} [{r['type']}] {'✓' if r['correct'] else '✗'} ({r['latency_s']}s)", flush=True)

    out_path = args.out or os.path.join(
        REPO_ROOT, "evals", f"refusal_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    _write_report(results, out_path)
    print(f"报告已写入: {out_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="DocPilot 拒答正确率评测（生成层）")
    parser.add_argument("--in-corpus", type=int, default=10, help="in_corpus 抽题数（默认 10）")
    parser.add_argument("--out", default=None, help="报告输出路径")
    args = parser.parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
