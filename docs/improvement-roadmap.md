# DocPilot 改进路线图（简化版）

> 来源：2026-08-22 全项目通读 + 对照常见在线问答助手的差距分析。
> 用法：从上往下逐项解决，每完成一项勾选并注明日期。范围只列"做什么"，不展开"怎么做"。

## 阶段 0：先能真跑（小改动，先做）

- [ ] 1. 关闭 `.env` 中 `USE_FAKE_MODEL=true`，接入真实 LLM（待模型切换方案确认：Qwen3.6 本地 Ollama，本机尚未安装）
- [x] 2. 嵌入模型 / Chroma 检索器改为单例，消除每次检索的模型加载开销（2026-08-22 完成：按 (库路径, 模型路径) 缓存，环境变量变更仍生效）
- [x] 3. `DEFAULT_AGENT` 从 `research-assistant` 改为 `rag-assistant`（2026-08-22 完成：同步更新 test_info 断言期望值）
- [x] 13. 钉住 `CHROMA_DB_PATH`（v2 库）与 `EMBEDDING_MODEL_PATH` 到 `.env`，避免代码默认值静默指向 v1 旧库（2026-08-22 完成；`.env.example` 已补文档，废弃库目录 v1/test 待确认后清理）

## 阶段 1：像个产品（用户可感知）

- [x] 4. 界面支持上传 PDF/DOCX 并在线入库（2026-08-23 完成：POST /ingest 端点 + AgentClient.ingest + 侧边栏上传控件；RecursiveCharacterTextSplitter 切分，同名按 basename 先删后加幂等替换；真实库冒烟通过，含 9 项新测试）
- [x] 5. 侧边栏会话历史列表（2026-08-23 完成：GET /threads 枚举端点 + AgentClient.list_threads + 侧边栏 Chat history 面板（点击恢复会话）。注意：sqlite 的 AsyncSqliteSaver 不支持跨线程 alist，线程 ID 列表改用只读 SQL 按 rowid 近期排序，详情走公开 API aget_tuple；postgres 部署暂返回 503，user_id 过滤留待 #10 用户体系）
- [x] 6. 可点击引用：引用卡片挂在 AI 回答下方（2026-08-23 完成：工具输出附 CITATIONS_JSON → collect_citations 节点经 custom 流向前端发射 → 展开式来源卡片带 chunk_id 与原文摘录；markdown 假页码按扩展名区分只显示"片段"；同批附侧边栏改造路线图面板，实时读本文件渲染）

## 阶段 2：答得更准（failure_analysis.md 已给方向）

- [ ] 7. 混合检索 BM25 + 向量（修 Q12 类：词面精确匹配被语义排序压过）
- [ ] 8. 查询改写 / 多意图拆分（修 Q20 口语化、Q36/Q39 双主题问题）
- [ ] 9. rerank + 拒答阈值（当前分数分布重叠，库外问题无法在检索层拒答）

## 阶段 3：能对外服务

- [ ] 10. 用户体系 + 接口限流（当前仅一个可为空的 bearer token）
- [ ] 11. 输出侧内容审查 + 检索内容的提示注入防护（当前只查输入）
- [ ] 12. 端到端生成质量评测（忠实度 / 引用正确性 / 拒答准确率，现有评测只覆盖检索层）

---

原则：一次只做一项；涉及检索质量的改动（7-9）每项先在 50 题评测集上跑基线对比再合入。

已知通道坑（Git Bash，2026-08-22 实测）：conda 注入的 `SSL_CERT_FILE` 指向不存在的路径会让 ollama 包 import 失败，跑测试须加 `env -u SSL_CERT_FILE`；`tests/app/test_streamlit_app.py::test_app_simple_non_streaming` 在本通道既有失败（Streamlit 读 home 目录超时，干净树同样失败），与代码改动无关。
