"""集成检查：锚定「真实 Chroma 库 metadata 带 chunk_id」这一隐含假设。

build_citations 依赖 docs 的 metadata["chunk_id"]，但全部单元测试都用手工
构造的假 Document，该假设天然为真、从未对真实库验证过。本文件是唯一
触碰真实向量库的检查，与 test_tools.py（纯单元、秒级）分开存放。

实现说明：真实检索在隔离子进程中执行，防护措施——
1. spawn 前在父进程 pop RUN_INTEGRATION：本机沙箱在 CreateProcess 时刻
   检查父进程环境块，含 RUN_INTEGRATION 即拦截子进程的 torch 原生加载
   （实测：不 pop 必崩，pop 后正常）；
2. 子进程 env 剔除 PYTEST_* 与 OPENAI_API_KEY：子进程启动环境块含这些
   变量时自身被标记，import torch 即段错误（0xC0000005，2026-07-28 后
   出现；均经独立对照实验确认。OPENAI_API_KEY 来自项目 pyproject
   [tool.pytest_env] 的 sk-fake 注入，本地检索链路不需要它）；
3. creationflags 脱离父进程组：深度防御，保留无害（其必要性未单独
   验证，前两条才是实测关键）。
普通 python 解释器无以上限制。父进程以 JSON 回收 metadata 做断言。

默认跳过；需设置 RUN_INTEGRATION=1 且真实库存在时才真实执行。
"""

import json
import os
import subprocess
import sys
import tempfile

import pytest

REAL_DB_PATH = "./chroma_db_qwen3_semantic_chunks"

# 子进程脚本：按真实路径加载 tools.py（与 test_tools.py 相同的 importlib 方式，
# 绕开 src/agents/__init__.py 的 ollama 链），真实检索并把 metadata 以 JSON 打印。
_CHILD_SCRIPT = r"""
import importlib.util
import json
import os
import sys

print("CHILD_S1: start", flush=True)
src_dir = os.path.abspath(os.path.join(os.getcwd(), "src"))
sys.path.insert(0, src_dir)

spec = importlib.util.spec_from_file_location(
    "tools_module_child",
    os.path.join(src_dir, "agents", "tools.py"),
)
tools_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tools_module)
print("CHILD_S2: tools.py loaded", flush=True)

retriever = tools_module.load_chroma_db()
print("CHILD_S3: retriever built", flush=True)

documents = retriever.invoke("DocPilot 的部署方式")
print("CHILD_S4: invoke done", flush=True)

print("METADATA_JSON:" + json.dumps([d.metadata for d in documents], ensure_ascii=False), flush=True)
"""

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1"
    or not os.path.isdir(REAL_DB_PATH),
    reason="集成检查：需真实 Chroma 库，且需设置 RUN_INTEGRATION=1",
)


def test_real_chroma_metadata_contains_chunk_id():
    """锚定假设：真实库每个片段的 metadata 都带非空 chunk_id 和 source。"""
    # 防护 1：父进程环境块在 CreateProcess 时刻不得含 RUN_INTEGRATION。
    os.environ.pop("RUN_INTEGRATION", None)
    # 防护 2：子进程启动环境块剔除实测的毒源变量（存在即段错误）：
    # - PYTEST_* 前缀（PYTEST_VERSION / PYTEST_CURRENT_TEST）
    # - OPENAI_API_KEY（项目 pyproject [tool.pytest_env] 注入的 sk-fake key；
    #   真实检索走本地嵌入模型与本地 Chroma，不需要任何 API key）
    child_env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("PYTEST_") and k != "OPENAI_API_KEY"
    }
    # 防护 3：脱离父进程组（深度防御，保留无害，必要性未单独验证）。
    creationflags = 0
    for name in ("DETACHED_PROCESS", "CREATE_NEW_PROCESS_GROUP", "CREATE_BREAKAWAY_FROM_JOB"):
        creationflags |= getattr(subprocess, name, 0)
    # 子脚本落盘到自建的 Temp 子目录执行（文件版经三连跑验证稳定；`-c` 内联
    # 形式未在解毒后重新验证，从简不复用）。
    child_dir = None
    try:
        child_dir = tempfile.mkdtemp(prefix="docpilot_it_")
        child_path = os.path.join(child_dir, "child_retrieve.py")
        with open(child_path, "w", encoding="utf-8") as f:
            f.write(_CHILD_SCRIPT)
        proc = subprocess.run(
            [sys.executable, child_path],
            capture_output=True,
            text=True,
            timeout=300,
            env=child_env,
            creationflags=creationflags,
        )
    finally:
        if child_dir and os.path.isdir(child_dir):
            for name in os.listdir(child_dir):
                os.unlink(os.path.join(child_dir, name))
            os.rmdir(child_dir)
    assert proc.returncode == 0, f"子进程真实检索失败:\n{proc.stderr}"

    marker = "METADATA_JSON:"
    json_line = next(
        (line[len(marker):] for line in proc.stdout.splitlines() if line.startswith(marker)),
        None,
    )
    assert json_line is not None, f"子进程未输出 metadata JSON:\n{proc.stdout}\n{proc.stderr}"

    metadatas = json.loads(json_line)
    assert metadatas, "真实库检索返回空，无法验证 metadata（先确认库路径与内容）"

    for metadata in metadatas:
        assert "chunk_id" in metadata, f"缺 chunk_id: {metadata}"
        assert metadata["chunk_id"], f"chunk_id 为空值: {metadata}"
        assert "source" in metadata, f"缺 source: {metadata}"
