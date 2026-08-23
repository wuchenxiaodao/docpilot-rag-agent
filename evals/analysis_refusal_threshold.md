# 检索层拒答阈值可行性分析（roadmap #9）

**结论：当前信号下检索层拒答阈值不可行，已搁置；拒答保留在生成层。**

## 缘起

roadmap #9 原计划「rerank + 拒答阈值」，目标是让检索层能在库外问题
上拒答（不把无关 chunk 喂给模型）。代码注释（`src/agents/tools.py`
`database_search_func`）早先观察到 Top-1 分数资料内最低 0.4042 低于
资料外最高 0.4685，两组重叠。#7 混合检索 + #8 多意图拆分后重新验证。

## 方法

`scripts/analyze_score_distribution.py` 复刻 `HybridRetriever._retrieve_one`
的 RRF 融合但保留分数，对 50 题逐题记录 Top-1 分数、Top-1 与 Top-2 的
margin，按 type 分组看分布是否可分。脚本可复现：

```sh
.venv/Scripts/python.exe scripts/analyze_score_distribution.py
```

## 结果（post-#7/#8，RRF 融合分数）

### Top-1 绝对分数

| 组 | n | min | max | mean | median |
|---|---|---|---|---|---|
| 资料内（in_corpus + multi_hop） | 35 | 0.0164 | 0.0279 | 0.0275 | 0.0279 |
| 资料外（out_corpus） | 15 | 0.0164 | 0.0279 | 0.0266 | 0.0272 |

**完全重叠**：资料内 min = 资料外 max = 0.0279。RRF 分数的天花板是
`1/(k+1) + w_b/(k+1) ≈ 0.0279`（vector 与 BM25 都把同一 chunk 排第 1），
地板是 `1/(k+1) ≈ 0.0164`（只有一个信号命中）。分数衡量的是「两个信号
是否一致」，不是「答案是否真在库里」。

### Top-1 与 Top-2 的 margin

| 组 | min | max |
|---|---|---|
| 资料内 | 0.0001 | 0.0114 |
| 资料外 | 0.0000 | 0.0108 |

同样重叠。margin 几乎都是 0（top-1 与 top-2 分数贴在一起，因为候选都
落在 0.0279 或 0.0164 两个点上）。

### 最佳阈值扫描

- 绝对分数 `top1 < 0.0273` 拒答：TP=34, **FN=1**（Q33 资料内被误拒）,
  FP=7, TN=8, F1=0.89。FN=1 即拒绝一道真有答案的题——不可接受。
- margin `< 0.0001` 拒答：TP=35, FN=0, FP=14, TN=1, F1=0.83。
  几乎不拒答，无用。

### 不可分离的案例

- Q33（资料内，中文「流式回答结束时客户端靠什么标志判断」）Top-1=0.0164
- Q50（资料外，巧克力曲奇食谱）Top-1=0.0164

两题分数完全相同，任何阈值都无法分开。

## 根因

RRF 融合分数是「vector 与 BM25 是否一致」的代理，不是「query 与
passage 是否语义相关」的直接度量。当 query 与某 chunk 共享词面
（如 "AcmeTech" 出现在 query 和 handbook 各 chunk 里），BM25 给高分；
vector 也因公司名嵌入相近给高分；两者一致 → RRF 顶格 0.0279——但
chunk 里其实没答 CEO/总部/营收。这是「相似 ≠ 相关」的经典问题，
独立嵌入无法解决，需 cross-encoder（query+passage 联合编码）或 LLM
判分。两者都要新模型，违反「不引入新依赖」约束。

## 已有的拒答：生成层

`evals/refusal_report_20260823_122841.md` 显示生成层（模型看上下文后
自行判断）在【真正库外】的题上 **8/8 全拒答正确**：巧克力曲奇、股票
代码、CEO、总部、成立年份、营收、天气 API。模型层拒答已满足需求。

## 数据质量问题：out_corpus 标签陈旧

v2 语料加入了仓库自身文档（VertexAI、GitHub MCP、AG-UI、架构、
凭据管理、create_chroma_db 等），故 15 道 out_corpus 题里有 7 道其实
已可答（Q41 VertexAI 鉴权、Q42 chunk size、Q43 AG-UI、Q44 GitHub PAT
轮换、Q45 Ollama 配置、Q47 on-call、Q48 secrets 轮换）——模型正确
作答是特性不是 bug。生成层「拒答正确率 8/15=53.3%」的分母被这些
陈旧标签拉低；按真正库外计是 8/8=100%。

## 处置

- **检索层拒答阈值**：搁置。证据如上，RRF/margin 分数不可分。需
  cross-encoder reranker（新模型，违反约束）才可行，留待未来放开
  依赖时再做。
- **rerank**：同理，无新模型时 RRF 已是可用信号的最佳组合，额外
  启发式（heading 匹配、首句匹配等） speculative 且易退化，不做。
- **生成层拒答**：保持现状（已 8/8 真正库外正确），不调。
- **out_corpus 标签**：检索评测不计算 out_corpus 的 Recall（仅观察），
  故陈旧标签不影响检索指标；拒答评测的分母问题记录在此，不强行改
  冻结评测集。
