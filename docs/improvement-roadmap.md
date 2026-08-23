# DocPilot 改进路线图（简化版）

> 来源：2026-08-22 全项目通读 + 对照常见在线问答助手的差距分析。
> 用法：从上往下逐项解决，每完成一项勾选并注明日期。范围只列"做什么"，不展开"怎么做"。

## 阶段 0：先能真跑（小改动，先做）

- [x] 1. 关闭 `.env` 中 `USE_FAKE_MODEL=true`，接入真实 LLM（2026-08-23 完成：Qwen3.6 35b-a3b 本地 Ollama，OpenAI 兼容端点；8GB 显存机用 gpu12 变体固定 12 层 GPU offload；检索回归逐字一致，拒答评测 8/8 真·库外全对，见 README 对比表。阶段 0 至此全部完成）
- [x] 2. 嵌入模型 / Chroma 检索器改为单例，消除每次检索的模型加载开销（2026-08-22 完成：按 (库路径, 模型路径) 缓存，环境变量变更仍生效）
- [x] 3. `DEFAULT_AGENT` 从 `research-assistant` 改为 `rag-assistant`（2026-08-22 完成：同步更新 test_info 断言期望值）
- [x] 13. 钉住 `CHROMA_DB_PATH`（v2 库）与 `EMBEDDING_MODEL_PATH` 到 `.env`，避免代码默认值静默指向 v1 旧库（2026-08-22 完成；`.env.example` 已补文档，废弃库目录 v1/test 待确认后清理）

## 阶段 1：像个产品（用户可感知）

- [x] 4. 界面支持上传 PDF/DOCX 并在线入库（2026-08-23 完成：POST /ingest 端点 + AgentClient.ingest + 侧边栏上传控件；RecursiveCharacterTextSplitter 切分，同名按 basename 先删后加幂等替换；真实库冒烟通过，含 9 项新测试）
- [x] 5. 侧边栏会话历史列表（2026-08-23 完成：GET /threads 枚举端点 + AgentClient.list_threads + 侧边栏 Chat history 面板（点击恢复会话）。注意：sqlite 的 AsyncSqliteSaver 不支持跨线程 alist，线程 ID 列表改用只读 SQL 按 rowid 近期排序，详情走公开 API aget_tuple；postgres 部署暂返回 503，user_id 过滤留待 #10 用户体系）
- [x] 6. 可点击引用：引用卡片挂在 AI 回答下方（2026-08-23 完成：工具输出附 CITATIONS_JSON → collect_citations 节点经 custom 流向前端发射 → 展开式来源卡片带 chunk_id 与原文摘录；markdown 假页码按扩展名区分只显示"片段"；同批附侧边栏改造路线图面板，实时读本文件渲染）

## 阶段 2：答得更准（failure_analysis.md 已给方向）

- [x] 7. 混合检索 BM25 + 向量（2026-08-23 完成：纯 Python BM25 + 加权 RRF 融合，w_b=0.7 由 50 题扫参选定（0.6-0.8 平台）；df>0.8N 词作语料自适应停用词；索引按库条目数自动重建（在线入库即生效）。**Recall@3 34→35（100% 满覆盖）**，Q12 未命中→第3、Q39 第2→第1；代价 Q17/Q32 从第1滑到第2（R@1 91.4%→88.6%，仍在上下文内不影响生成），MRR 0.938→0.933）
- [x] 8. 查询改写 / 多意图拆分（2026-08-23 完成：`, and <疑问词>` 复合问题保守正则拆分 + round-robin 交错合并；代词守卫拒绝含人称代词的子查询（前指丢失、检索太弱，Q17/Q35/Q36 形态不拆不退化）。**multi_hop full-coverage@3 8→9（Q39 双 chunk 齐了）**，R@1 85.7%→88.6%、MRR 0.919→0.933，in_corpus 与其余 48 题零退化。Q17 仍 1/2——其子查询 "who should they contact?" 的代词 "they" 经守卫拒绝拆分，需 LLM 改写才能修，留待后续；同形态已确保不退化）
- [~] 9. rerank + 拒答阈值（2026-08-23 调查后搁置：`scripts/analyze_score_distribution.py` 实测 RRF 融合分数资料内 min=资料外 max=0.0279 完全重叠，margin 也重叠，检索层拒答阈值不可行——FN=1（Q33 被误拒）不可接受。根因是 RRF 衡量「两信号一致性」非「语义相关性」，需 cross-encoder 新模型才能解，违反不引入依赖约束。生成层拒答已 8/8 真正库外正确（巧克力曲奇/股票代码/CEO/总部等），满足需求。详见 `evals/analysis_refusal_threshold.md`）

## 阶段 3：能对外服务

- [x] 10. 用户体系 + 接口限流（2026-08-23 完成：`core/auth.py` 多用户 API key（JSON 文件/内联，按 mtime 缓存热重载）+ `core/ratelimit.py` 内存固定窗口限流；`verify_bearer` 解析为 `Principal` 挂 `request.state`，api-key 来源服务端钉 `user_id`（客户端不可冒充），共享 `AUTH_SECRET`/匿名仍走客户端 `user_id`（向后兼容）；`RATE_LIMIT_PER_MIN` 默认 0=关闭，命中 `/invoke`、`/stream`、`/ingest`，超限 429+Retry-After，被拒请求不占名额。无新依赖。224 测试通过（+21 新增，零退化）。详见 README「API hardening」节）
- [ ] 11. 输出侧内容审查 + 检索内容的提示注入防护（当前只查输入）
- [ ] 12. 端到端生成质量评测（忠实度 / 引用正确性 / 拒答准确率，现有评测只覆盖检索层）

---

原则：一次只做一项；涉及检索质量的改动（7-9）每项先在 50 题评测集上跑基线对比再合入。

已知通道坑（Git Bash，2026-08-22 实测）：conda 注入的 `SSL_CERT_FILE` 指向不存在的路径会让 ollama 包 import 失败，跑测试须加 `env -u SSL_CERT_FILE`；`tests/app/test_streamlit_app.py::test_app_simple_non_streaming` 在本通道既有失败（Streamlit 读 home 目录超时，干净树同样失败），与代码改动无关。
