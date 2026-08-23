# DocPilot 检索评测报告

> **边界声明**：本结果仅适用于当前评测集与当前 Chroma 库配置，样本量 N=50，不构成统计显著性结论，不可跨数据集比较。
>
> 语料规模：10 份文档 / 56 个 chunk；k=3。混合切分：AcmeTech PDF 用硬编码章节标题切分，9 份 markdown 用显式配置的 heading 标题集切分（8 份 `##` 级、VertexAI 用 `###` 级）。markdown 无页码，其 chunk_id 的 `p1` 为占位符（PDF 的 `p` 是真实页码），会影响引用展示。

## 运行配置

- 评测集: `evals/eval_set.jsonl`（共 50 条）
- Chroma 库: `./chroma_db_qwen3_semantic_chunks_v2`
- 语料: 10 份文档 / 56 个 chunk
- Top-k: 3
- 生成时间: 2026-08-23 20:58:14
- 计入指标: 35 条（in_corpus 25 + multi_hop 10）；排除: 待标注 0 条、out_corpus 15 条（仅观察）

## 指标汇总

**总体**（in_corpus + multi_hop）：

| 指标 | 分子/分母 | 值 |
|---|---|---|
| Recall@1 | 31/35 | 88.6% |
| Recall@3 | 35/35 | 100.0% |
| MRR | 32.667/35 | 0.933 |

**分 type**（不混合平均）：

| type | n | Recall@1 | Recall@3 | MRR | full-coverage@3 |
|---|---|---|---|---|---|
| in_corpus | 25 | 22/25 = 88.0% | 25/25 = 100.0% | 23.167/25 = 0.927 | —（单期望 chunk，同 Recall@k） |
| multi_hop | 10 | 9/10 = 90.0% | 10/10 = 100.0% | 9.500/10 = 0.950 | 8/10 = 80.0% |

> **口径说明**：Recall@3 对 multi_hop 的判定是「至少命中一个期望 chunk」，与 multi_hop「需跨两 chunk 才能作答」的定义不符，仅供对照；full-coverage@3（k 内全部期望 chunk 命中）才是 multi_hop 的严格口径。另外 k=3 对 multi_hop 天然不利——双期望 chunk 需同时挤进前 3 位，两个 type 用同一 k 比较时必须附带此说明。

## 逐题明细

| id | 问题 | 期望 chunk_id | 实际 Top-k chunk_id | 命中 | 名次 | 覆盖 |
|---|---|---|---|---|---|---|
| Q1 | What are AcmeTech's mission and values? | AcmeTech_Employee_Handbook-p1-c2 | AcmeTech_Employee_Handbook-p1-c2<br>AcmeTech_Employee_Handbook-p2-c5<br>AcmeTech_Employee_Handbook-p3-c9 | ✅ | 1 | 1/1 |
| Q2 | How many days per week may employees work remotel… | AcmeTech_Employee_Handbook-p1-c4 | AcmeTech_Employee_Handbook-p1-c4<br>AcmeTech_Employee_Handbook-p2-c6<br>AcmeTech_Employee_Handbook-p1-c3 | ✅ | 1 | 1/1 |
| Q3 | What are AcmeTech's core working hours? | AcmeTech_Employee_Handbook-p1-c4 | AcmeTech_Employee_Handbook-p1-c4<br>AcmeTech_Employee_Handbook-p1-c2<br>AcmeTech_Employee_Handbook-p1-c3 | ✅ | 1 | 1/1 |
| Q4 | How many weeks of paid parental leave are provide… | AcmeTech_Employee_Handbook-p2-c6 | AcmeTech_Employee_Handbook-p2-c6<br>Weekly_Maintenance_Run-p1-c2<br>Weekly_Maintenance_Run-p1-c1 | ✅ | 1 | 1/1 |
| Q5 | What should employees do about suspicious emails? | AcmeTech_Employee_Handbook-p2-c7 | AcmeTech_Employee_Handbook-p2-c7<br>Daily_Sentinel-p1-c4<br>AcmeTech_Employee_Handbook-p2-c5 | ✅ | 1 | 1/1 |
| Q11 | How many days of paid time off (PTO) do employees… | AcmeTech_Employee_Handbook-p2-c6 | AcmeTech_Employee_Handbook-p2-c6<br>AcmeTech_Employee_Handbook-p1-c4<br>Weekly_Maintenance_Run-p1-c1 | ✅ | 1 | 1/1 |
| Q12 | What are AcmeTech's regular office hours? | AcmeTech_Employee_Handbook-p1-c3 | AcmeTech_Employee_Handbook-p3-c9<br>AcmeTech_Employee_Handbook-p1-c4<br>AcmeTech_Employee_Handbook-p1-c3 | ✅ | 3 | 1/1 |
| Q13 | How can employees contact IT support? | AcmeTech_Employee_Handbook-p3-c9 | AcmeTech_Employee_Handbook-p3-c9<br>AcmeTech_Employee_Handbook-p2-c7<br>RAG_Assistant-p1-c2 | ✅ | 1 | 1/1 |
| Q14 | If an employee uses their full annual PTO and all… | AcmeTech_Employee_Handbook-p2-c6 | AcmeTech_Employee_Handbook-p2-c6<br>Weekly_Maintenance_Run-p1-c1<br>AcmeTech_Employee_Handbook-p1-c4 | ✅ | 1 | 1/1 |
| Q15 | An employee wants to work from home and also need… | AcmeTech_Employee_Handbook-p1-c4 | AcmeTech_Employee_Handbook-p1-c4<br>AcmeTech_Employee_Handbook-p1-c3<br>AcmeTech_Employee_Handbook-p2-c7 | ✅ | 1 | 1/1 |
| Q16 | How much combined time off can a new parent take … | AcmeTech_Employee_Handbook-p2-c6 | AcmeTech_Employee_Handbook-p2-c6<br>Dependency_Upgrades-p1-c1<br>AcmeTech_Employee_Handbook-p1-c4 | ✅ | 1 | 1/1 |
| Q17 | If an employee receives a suspicious email while … | AcmeTech_Employee_Handbook-p2-c7<br>AcmeTech_Employee_Handbook-p3-c9 | Daily_Sentinel-p1-c4<br>AcmeTech_Employee_Handbook-p2-c7<br>AcmeTech_Employee_Handbook-p1-c4 | ✅ | 2 | 1/2 |
| Q18 | What's the maximum number of days staff are allow… | AcmeTech_Employee_Handbook-p1-c4 | AcmeTech_Employee_Handbook-p1-c4<br>Weekly_Maintenance_Run-p1-c1<br>Weekly_Maintenance_Run-p1-c5 | ✅ | 1 | 1/1 |
| Q19 | How much paid leave do new parents get after the … | AcmeTech_Employee_Handbook-p2-c6 | AcmeTech_Employee_Handbook-p2-c6<br>Dependency_Upgrades-p1-c1<br>AcmeTech_Employee_Handbook-p1-c4 | ✅ | 1 | 1/1 |
| Q20 | If I receive a phishing-looking message, what am … | AcmeTech_Employee_Handbook-p2-c7 | Daily_Sentinel-p1-c4<br>Weekly_Maintenance_Run-p1-c3<br>AcmeTech_Employee_Handbook-p2-c7 | ✅ | 3 | 1/1 |
| Q21 | Which file records the fully resolved dependency … | Dependency_Upgrades-p1-c2 | Dependency_Upgrades-p1-c2<br>Weekly_Maintenance_Run-p1-c10<br>Dependency_Upgrades-p1-c3 | ✅ | 1 | 1/1 |
| Q22 | After editing version pins in pyproject.toml, wha… | Dependency_Upgrades-p1-c3 | Dependency_Upgrades-p1-c3<br>Dependency_Upgrades-p1-c2<br>Dependency_Upgrades-p1-c7 | ✅ | 1 | 1/1 |
| Q23 | I don't have any LLM API keys — can I still live-… | Dependency_Upgrades-p1-c4 | Dependency_Upgrades-p1-c4<br>Weekly_Maintenance_Run-p1-c7<br>Weekly_Maintenance_Run-p1-c9 | ✅ | 1 | 1/1 |
| Q24 | What range of Python versions does this project d… | Dependency_Upgrades-p1-c8 | Dependency_Upgrades-p1-c8<br>Dependency_Upgrades-p1-c2<br>Dependency_Upgrades-p1-c3 | ✅ | 1 | 1/1 |
| Q25 | The biweekly parity gate decides run-or-skip base… | Weekly_Maintenance_Run-p1-c1 | Weekly_Maintenance_Run-p1-c1<br>Weekly_Maintenance_Run-p1-c2<br>Weekly_Maintenance_Run-p1-c5 | ✅ | 1 | 1/1 |
| Q26 | How long must an issue sit unanswered by the othe… | Weekly_Maintenance_Run-p1-c6 | Weekly_Maintenance_Run-p1-c6<br>Weekly_Maintenance_Run-p1-c3<br>Daily_Sentinel-p1-c3 | ✅ | 1 | 1/1 |
| Q27 | When the daily sentinel has nothing urgent to rep… | Daily_Sentinel-p1-c1 | Daily_Sentinel-p1-c1<br>Daily_Sentinel-p1-c4<br>Daily_Sentinel-p1-c3 | ✅ | 1 | 1/1 |
| Q28 | Does a Dependabot security alert that has no PR y… | Daily_Sentinel-p1-c3 | Daily_Sentinel-p1-c3<br>Daily_Sentinel-p1-c1<br>Weekly_Maintenance_Run-p1-c5 | ✅ | 1 | 1/1 |
| Q29 | Why does the Vertex AI doc recommend stable model… | VertexAI-p1-c3 | VertexAI-p1-c3<br>VertexAI-p1-c4<br>VertexAI-p1-c2 | ✅ | 1 | 1/1 |
| Q30 | What's the minimum role I should grant the servic… | VertexAI-p1-c4 | VertexAI-p1-c4<br>VertexAI-p1-c2<br>VertexAI-p1-c3 | ✅ | 1 | 1/1 |
| Q31 | What do I need to set so the GitHub MCP agent act… | GitHub_MCP_Agent-p1-c2 | GitHub_MCP_Agent-p1-c2<br>GitHub_MCP_Agent-p1-c3<br>GitHub_MCP_Agent-p1-c4 | ✅ | 1 | 1/1 |
| Q32 | What's the URL pattern for running a specific age… | AGUI-p1-c1 | AGUI-p1-c2<br>AGUI-p1-c1<br>AGUI-p1-c4 | ✅ | 2 | 1/1 |
| Q33 | 流式回答结束时，客户端靠什么标志判断已经完事？ | architecture-p1-c4 | architecture-p1-c4<br>AGUI-p1-c4<br>Daily_Sentinel-p1-c4 | ✅ | 1 | 1/1 |
| Q34 | How are files in the privatecredentials/ folder k… | File_Based_Credentials-p1-c1 | File_Based_Credentials-p1-c1<br>File_Based_Credentials-p1-c2<br>File_Based_Credentials-p1-c3 | ✅ | 1 | 1/1 |
| Q35 | How do I continue an existing AG-UI conversation … | AGUI-p1-c3<br>AGUI-p1-c4 | AGUI-p1-c4<br>AGUI-p1-c3<br>AGUI-p1-c2 | ✅ | 1 | 2/2 |
| Q36 | The weekly run is draft-only with exactly one exc… | Weekly_Maintenance_Run-p1-c3<br>Weekly_Maintenance_Run-p1-c6 | Weekly_Maintenance_Run-p1-c3<br>Weekly_Maintenance_Run-p1-c6<br>Weekly_Maintenance_Run-p1-c2 | ✅ | 1 | 2/2 |
| Q37 | If the deployed Streamlit app goes down, what mak… | Daily_Sentinel-p1-c3<br>Daily_Sentinel-p1-c4 | Daily_Sentinel-p1-c4<br>Daily_Sentinel-p1-c3<br>Weekly_Maintenance_Run-p1-c7 | ✅ | 1 | 2/2 |
| Q38 | What does the maintenance run do on an off-week, … | Weekly_Maintenance_Run-p1-c1<br>Weekly_Maintenance_Run-p1-c2 | Weekly_Maintenance_Run-p1-c2<br>Weekly_Maintenance_Run-p1-c1<br>Weekly_Maintenance_Run-p1-c3 | ✅ | 1 | 2/2 |
| Q39 | After bumping version pins, what command regenera… | Dependency_Upgrades-p1-c3<br>Dependency_Upgrades-p1-c4 | Dependency_Upgrades-p1-c3<br>Weekly_Maintenance_Run-p1-c9<br>Dependency_Upgrades-p1-c6 | ✅ | 1 | 1/2 |
| Q40 | Which environment variable does the Vertex SDK re… | VertexAI-p1-c2<br>File_Based_Credentials-p1-c2 | File_Based_Credentials-p1-c2<br>VertexAI-p1-c4<br>VertexAI-p1-c2 | ✅ | 1 | 2/2 |

## 失败案例（未命中）

无。所有计入指标的题目均命中。

## 待标注排除

本次无待标注条目，50 条全部计入各自分组。

## out_corpus 观察（不计入指标）

| id | 问题 | 检索返回条数 | Top-1 返回（观察） |
|---|---|---|---|
| Q6 | What is AcmeTech's stock ticker? | 3 | AcmeTech_Employee_Handbook-p3-c8 |
| Q7 | Who is AcmeTech's CEO? | 3 | AcmeTech_Employee_Handbook-p3-c9 |
| Q8 | In what year was AcmeTech founded? | 3 | AcmeTech_Employee_Handbook-p1-c2 |
| Q9 | Where is AcmeTech's headquarters? | 3 | AcmeTech_Employee_Handbook-p3-c9 |
| Q10 | What was AcmeTech's revenue last year? | 3 | AcmeTech_Employee_Handbook-p3-c9 |
| Q41 | How do I get an API key for the Gemini Developer … | 3 | VertexAI-p1-c4 |
| Q42 | What is the default chunk size used by create_chr… | 3 | RAG_Assistant-p1-c1 |
| Q43 | What rate limits apply to the AG-UI endpoints? | 3 | AGUI-p1-c1 |
| Q44 | How do I rotate the GitHub Personal Access Token … | 3 | GitHub_MCP_Agent-p1-c2 |
| Q45 | How do I configure Ollama as the LLM provider for… | 3 | Weekly_Maintenance_Run-p1-c9 |
| Q46 | How much does a Google Cloud service account for … | 3 | VertexAI-p1-c4 |
| Q47 | Who is on call to respond when the daily sentinel… | 3 | Daily_Sentinel-p1-c3 |
| Q48 | How are secrets rotated in production when using … | 3 | VertexAI-p1-c6 |
| Q49 | What weather API does the chatbot agent's weather… | 3 | AGUI-p1-c3 |
| Q50 | What is the best recipe for chocolate chip cookie… | 3 | AGUI-p1-c3 |
