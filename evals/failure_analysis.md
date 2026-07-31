# DocPilot 检索失败归因报告（Day 16）

> 适用对象：`evals/eval_set.jsonl` 50 题在 `chroma_db_qwen3_semantic_chunks_v2`
> （10 文档 / 56 chunk）上的 Baseline（k=3）。每例均逐案打开原文核查，
> 顺序固定：先验标注 → 再看实际 Top-3 原文 → 归入六类之一 → 给修复方向（本次不实施）。

## 失败六分类（2026-07-31 由四类扩充）

1. **切分问题**：答案被切断在两个 chunk 之间，任何单个 chunk 都不完整（指**非设计**的断裂）。
2. **嵌入语义不匹配**：问题与文档用词体系不同（缩写 vs 全称、中文 vs 英文术语）。
3. **问法表达差异**：问题是口语化提问，文档是书面陈述句。
4. **标注错**：期望 id 本身就填错了。
5. **检索器排序特性**：用词体系一致、切分完整、标注无误，仍未命中。典型表现为模型偏好某类表层语义（question / support / urgent），压过精确词面重叠。
6. **多意图单向量**：单个查询含两个主题，压成一个向量后落在两者中间，被「表层同时提到两个主题词」的 chunk 抢走。属查询侧问题。

规则：案例与所有现有类别对不上时，新增一类并写清判据，不硬塞。

---

## 例 1：Q12 → 检索器排序特性

**问题原文**：What are AcmeTech's regular office hours?

**期望 chunk + 原文摘录**：
`AcmeTech_Employee_Handbook-p1-c3` — "Work Hours & Attendance — Our regular office hours are 9:00 AM to 5:00 PM, Monday through Friday."
答案逐字在 chunk 内，标注**正确**（第一步排除标注错）。

**实际 Top-3**（两库一致：9-chunk 与 56-chunk 结果相同，排除跨文档干扰）：
1. `p3-c9` Contact & Support："If you have questions or need support, reach out: - HR..."
2. `p3-c8` Employee Benefits："comprehensive benefits package, including: - Health Insurance..."
3. `p1-c2` Company Mission："Our mission at AcmeTech is to develop cutting-edge software..."

**自检索诊断实验**（先定层再归类）：
query 取期望 chunk 原文首句 `Work Hours & Attendance — Our regular office hours are 9:00 AM to 5:00 PM`：

| 名次 | chunk_id | score（距离，越小越近） |
|---|---|---|
| #1 | **AcmeTech_Employee_Handbook-p1-c3（自身）** | 0.2761 |
| #2 | AcmeTech_Employee_Handbook-p1-c4 | 1.0808 |
| #3 | AcmeTech_Employee_Handbook-p2-c6 | 1.0937 |

向量健全性：p1-c3 在库 ✓；stored vector norm = 1.001460（≈1，正常）；cosine(库存向量, 同文新算向量) = 1.000000（入库无损）。

对照：原始 Q12 问题检索时 p1-c3 仅排 **#6**（cos=0.4105），而 p3-c9 以 cos=0.6706 居首。

**结论**：自检索排第 1 → 索引/embedding 层正常 → 按判定规则归**检索器排序特性**。
问题嵌入被拉向 "questions/support" 表层语义（p3-c9 开头即 "If you have questions or need support"），压过与 p1-c3 的逐字重叠；该偏好在两种候选池规模下稳定复现，是模型固有排序特性。

**修复方向（本次不做）**：混合检索（BM25 + 向量）引入词面匹配分；或单测该 case 去掉 query instruction 后的排名变化。

---

## 例 2：Q39 → 多意图单向量（Q36 同类对照）

**问题原文**：After bumping version pins, what command regenerates the lockfile, and which environment variable lets the live e2e test run without provider credentials?

**期望 chunk + 原文摘录**：
- `Dependency_Upgrades-p1-c3` — "4. Re-resolve: `uv lock --upgrade`. Read the conflict messages carefully..." ✓
- `Dependency_Upgrades-p1-c4` — "The service ships a fake model (`USE_FAKE_MODEL=true`) ... without any provider credentials." ✓
标注**正确**。

**实际 Top-3**：
1. `Weekly_Maintenance_Run-p1-c10`（Phase F — Dependency refresh）："Use the dependency-refresh skill (playbook: docs/Dependency_Upgrades.md). Safe bumps in one PR... Run the full verification ladder including the fake-model live e2e"
2. `Dependency_Upgrades-p1-c3`（期望 1/2，hit@2）
3. `Weekly_Maintenance_Run-p1-c9`（Phase E — Model catalog refresh）

**归类：多意图单向量**。判据：查询含两个主题（锁文件命令 + fake-model 环境变量），压成一个向量后落在两者中间；`p1-c10` 一个 chunk 表层同时覆盖两个主题词（"dependency-refresh" 与 "fake-model live e2e"），对混合向量的契合度高于任一单主题期望 chunk，抢走第 1。这是查询侧问题——multi_hop 跨 chunk 是评测集的设计如此，**不是切分缺陷**，故不归切分问题；同域同术语，排除嵌入语义不匹配；书面技术问句，排除问法差异。

**同类对照 Q36**（期望 c3+c6，实际 c3 hit@1、c6 未进 Top-3）：c6 标题逐字含 "the one pre-authorized write"，却被 c2/c1（日历/奇偶周调度主题）挤出 Top-3——同一机制：双主题混合向量偏向只覆盖单一主题词的 chunk。

**修复方向（本次不做）**：multi_hop 题先做子查询拆分（每半题各检一次再合并），或对 multi_hop 单独放大 k。

---

## 例 3：Q20 → 问法表达差异

**问题原文**：If I receive a phishing-looking message, what am I supposed to do?

**期望 chunk + 原文摘录**：
`AcmeTech_Employee_Handbook-p2-c7` — "IT & Security Guidelines — Employees must: ... - Report any suspicious emails or activity to IT immediately."
答案逐字在，标注**正确**。

**实际 Top-3**：
1. `Daily_Sentinel-p1-c4`："If something IS urgent — End the session with a short alert message..."
2. `Weekly_Maintenance_Run-p1-c3`："Ground rules — Draft-only, with exactly one exception..."
3. `AcmeTech_Employee_Handbook-p2-c7`（期望，hit@3）

**归类：问法表达差异**。判据：同证据的正式问法 Q5（"What should employees do about suspicious emails?"）在新库仍 hit@1，Q20 是其口语化 paraphrase（"phishing-looking message"、"what am I supposed to do"）——唯一变量是问法语域，排名即从 @1 掉到 @3，是"口语化提问 vs 书面陈述句"的直接对照；用词体系相同（security/report/alert 同域），排除嵌入语义不匹配；答案单 chunk 完整，排除切分问题。旧库（无同域竞争者）中 Q20 曾 hit@1——语域差异提供缺口，同域新语料完成抢占，两因素叠加。

**修复方向（本次不做）**：paraphrase 鲁棒性单独成轨评测（同证据题正式/口语双版本对照）；后续可用查询改写弥补语域差。

---

## 归因分布与后续

| 题号 | 归类 | 修复方向（未实施） |
|---|---|---|
| Q12 | 检索器排序特性 | 混合检索 / 测 query instruction 影响 |
| Q39（+Q36 对照） | 多意图单向量 | 子查询拆分 / multi_hop 单独放大 k |
| Q20 | 问法表达差异 | paraphrase 双版本对照轨 |

三例覆盖六类中的三类，均附对照题（Q3/Q5/Q36）支撑排他性论证。
