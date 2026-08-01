# DocPilot 集成测试交接记录（Day 12 收尾）

> 面向后续维护者。读这一份就够：怎么用、两个已确认的环境坑、以及排查方法论。

## 1. 这个测试是什么

`tests/agents/test_tools_integration.py` 锚定一个隐含假设：**真实 Chroma 库（`./chroma_db_qwen3_semantic_chunks`）中每个片段的 metadata 都带非空 `chunk_id` 和 `source`**。

为什么需要它：`build_citations()` 的去重 key 依赖 `metadata["chunk_id"]`。`tests/agents/test_tools.py` 的全部 25 个单元测试都用手工构造的假 Document，该假设天然为真、无法证伪。若真实库字段缺失或改名（如 `chunkId`），引用会照常生成、程序照常返回、单元测试照常全绿——但所有引用无法定位到具体段落，全程无报错。这个集成测试就是防这种沉默故障的下游锚点。

边界：它锚定的是**当前这个库**的形状，不保证重建库时字段不变。根治做法是在建库侧加校验（如 Pydantic 约束 metadata 结构），本次只做廉价下游锚点。

## 2. 怎么跑

```powershell
# 默认：跳过（集成测试不拖慢日常套件）
.\.venv\Scripts\python.exe -m pytest tests/agents -q

# 真实执行：设置 RUN_INTEGRATION=1（会真实加载 GPU 嵌入模型，约 30s）
$env:RUN_INTEGRATION="1"
.\.venv\Scripts\python.exe -m pytest tests/agents/test_tools_integration.py -q
Remove-Item Env:\RUN_INTEGRATION   # 跑完务必清理，避免污染环境
```

当前目录级真实状态（PowerShell）：`1 failed, 43 passed, 1 skipped` —— 唯一 failed 见下节，与代码无关。

## 3. 环境坑一：`ACC_PRODUCT_CONFIG_V3` 超长变量（已知，勿排查）

`tests/agents/test_tools.py::TestEmbeddingModelPath::test_env_var_override` 在部分宿主环境中失败，报错：

```text
ValueError: the environment variable is longer than 32767 characters
```

**成因与代码无关**：某些宿主 shell 会注入超长环境变量（如 `ACC_PRODUCT_CONFIG_V3`，是超过 Windows 单变量 32767 字符上限的 JSON）。`unittest.mock.patch.dict` 在**恢复阶段**把原值写回 `os.environ` 时触发该上限抛错——注意是恢复阶段炸，不是设置阶段，这个失败模式本身容易误判。

处理方式：知道它存在即可，不要为这个失败改测试逻辑。若未来 CI 环境无此变量，该用例自然转绿。

## 4. 环境坑二（方法论核心）：异常现象，先怀疑执行通道

Day 12 排查中曾观察到的现象：`pytest` 进程内 `import torch` 段错误（0xC0000005）。三轮共约 25 组实验一度归因于「环境变量毒源」「沙箱按 Job Object 拦截」「环境回归」，并给测试加了五道防护对策。

**最终证伪（单变量探针 + 五轮减法 + 第六轮去掉子进程）**：该现象只出现在 **WorkBuddy Bash 工具**这一执行通道内；换 PowerShell 后同一 `.venv`、同一 pytest、同一环境变量组合下一切正常。与项目代码、pytest 配置、依赖版本、环境变量**全部无关**。五道对策逐一删除后测试仍稳定通过，最终形态就是现在的直白进程内版本。

两条方法论，按价值排序：

1. **当现象只在某一个执行通道里出现时，第一个该换的变量是执行通道本身，而不是被测代码。** 换 shell/终端/工具复现一次，成本几分钟；在错误通道里加防护，成本是两块的工作量 plus 一堆错误注释。
2. **多变量改动不得用于单变量归因。** 块 3 最后一次实验同时引入四道对策，随后把功劳分配给单个变量——结论全是错的。每个实验只动一个变量，动两处结果作废。

另：在 Bash 工具沙箱内 `git commit`/`update-ref` 会打印成功但分支指针不动（PowerShell 下正常）。涉及 git 写操作时留意执行通道。

## 5. 当前测试形态说明

最终形态是有意的直白：模块级 importlib 加载 `tools.py`（绕开 `src/agents/__init__.py` 的 ollama 链，与 `test_tools.py` 同款），测试函数内直接 `load_chroma_db().invoke()` 并逐条断言 metadata。

已知代价：因 collection 阶段加载 langchain→torch 链，即使默认 skip 也需 ~10s。换取的是断言失败时的 pytest 原生 traceback 与零维护负担，值得。

不要重新引入的子进程/环境剔除/脱离进程组等对策——它们服务于一个已被证伪的前提（`650f93c`、`9d7084d` 两次提交的 diff 即减法证据）。
