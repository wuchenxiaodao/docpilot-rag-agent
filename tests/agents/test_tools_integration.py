"""集成检查：锚定「真实 Chroma 库 metadata 带 chunk_id」这一隐含假设。

build_citations 依赖 docs 的 metadata["chunk_id"]，但全部单元测试都用手工
构造的假 Document，该假设天然为真、从未对真实库验证过。本文件是唯一
触碰真实向量库的检查，与 test_tools.py（纯单元、秒级）分开存放。

默认跳过；需设置 RUN_INTEGRATION=1 且真实库存在时才真实执行。
"""

import importlib.util
import os
import sys

import pytest

REAL_DB_PATH = "./chroma_db_qwen3_semantic_chunks"


def _load_tools_module():
    """从文件路径加载 src/agents/tools.py（与 test_tools.py 同款，绕开
    src/agents/__init__.py 的 ollama 导入链）。

    重量级：exec_module 会连带拉起 langchain / torch / chromadb。
    定义在顶层不等于执行——调用必须只发生在测试函数体内，
    否则 pytest collection 期就会付这笔导入成本，默认 skip 路径
    会从 0.03s 退化到秒级以上。
    """
    _src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    sys.path.insert(0, _src_dir)

    spec = importlib.util.spec_from_file_location(
        "tools_module_integration",
        os.path.join(_src_dir, "agents", "tools.py"),
    )
    tools_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tools_module)
    return tools_module

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1"
    or not os.path.isdir(REAL_DB_PATH),
    reason="集成检查：需真实 Chroma 库，且需设置 RUN_INTEGRATION=1",
)


def test_real_chroma_metadata_contains_chunk_id():
    """锚定假设：真实库每个片段的 metadata 都带非空 chunk_id 和 source。"""
    # 真实定义位置：src/agents/tools.py（load_chroma_db，已 grep 确认）
    load_chroma_db = _load_tools_module().load_chroma_db
    retriever = load_chroma_db()
    documents = retriever.invoke("DocPilot 的部署方式")

    assert documents, "真实库检索返回空，无法验证 metadata（先确认库路径与内容）"

    for doc in documents:
        assert "chunk_id" in doc.metadata, f"缺 chunk_id: {doc.metadata}"
        assert doc.metadata["chunk_id"], f"chunk_id 为空值: {doc.metadata}"
        assert "source" in doc.metadata, f"缺 source: {doc.metadata}"
