"""集成检查：锚定「真实 Chroma 库 metadata 带 chunk_id」这一隐含假设。

build_citations 依赖 docs 的 metadata["chunk_id"]，但全部单元测试都用手工
构造的假 Document，该假设天然为真、从未对真实库验证过。本文件是唯一
触碰真实向量库的检查，与 test_tools.py（纯单元、秒级）分开存放。

实现说明：真实检索在独立子进程中执行，父进程以 JSON 回收 metadata 做
断言。子进程架构仅为隔离与超时控制，无其他环境对策——2026-07-30 经
五轮单变量减法实测（PowerShell）：父进程 pop RUN_INTEGRATION、子进程
剔除 PYTEST_*、剔除 OPENAI_API_KEY、脱离进程组、脚本落盘 Temp 子目录，
逐一去掉后本测试均仍通过，故全部删除。早前曾观察到 pytest 进程内
import torch 触发 access violation (0xC0000005) 的现象；该现象在本机
PowerShell 环境不可复现，机制未定位到单一变量，勿据此推断原因。

默认跳过；需设置 RUN_INTEGRATION=1 且真实库存在时才真实执行。
"""

import json
import os
import subprocess
import sys

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
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD_SCRIPT],
        capture_output=True,
        text=True,
        timeout=300,
    )
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
