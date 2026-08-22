# Qwen3.6 本地部署与切换收尾手册（本机 Windows）

> 状态基线（2026-08-23 00:30）：代码/配置/测试/检索回归已全部完成并推送（至 `22bb600`）。
> 剩余工作等两个后台下载完成：Ollama 二进制（`%LOCALAPPDATA%\Programs\OllamaDl`，8 分片并行）
> 与 qwen3.6:35b-a3b 模型（`~/OllamaModelDl/model_pipeline.sh`，14 分片并行，完成后自动
> 校验 sha256 并安装到 `~/.ollama/models`，成功标志：`~/OllamaModelDl/DONE` 存在且
> pipeline.log 末尾为 `MODEL INSTALLED`）。

## 前置状态检查

```bash
# 模型是否就绪
ls ~/OllamaModelDl/DONE && tail -3 ~/OllamaModelDl/pipeline.log
# 二进制是否下完（对比 1460302386 字节，dl_status.log 出现 ASSEMBLED）
tail -3 "$LOCALAPPDATA/Programs/OllamaDl/dl_status.log"
```

两者未齐：只报告进度，不做其他事。

## 第 1 步：安装并启动 Ollama

```bash
D="$LOCALAPPDATA/Programs/Ollama"
mkdir -p "$D" && cd "$LOCALAPPDATA/Programs/OllamaDl" && unzip -q -o ollama-windows-amd64.zip -d "$D"
"$D/ollama.exe" list          # 应列出 qwen3.6:35b-a3b（读 ~/.ollama/models 的手工布局）
nohup "$D/ollama.exe" serve > /tmp/ollama_serve.log 2>&1 &
sleep 5
curl -s http://localhost:11434/v1/models | head -20
```

注意：`serve` 必须一直挂着（后台任务）；若 `list` 看不到模型，检查
`~/.ollama/models/manifests/registry.ollama.ai/library/qwen3.6/35b-a3b` 与
`blobs/sha256-f5ee307a*`（23.9GB）是否存在。

## 第 2 步：端到端冒烟（tool calling + SSE）

```bash
cd /c/Users/wang/Projects/agent-service-toolkit
env -u SSL_CERT_FILE .venv/Scripts/python.exe src/run_service.py &   # 后台起服务
sleep 15 && curl -s http://localhost:8080/info | head -5              # default_model 应为 openai-compatible
# 非流式 + 工具调用（Q12 是已知难例，答案在 Employee Handbook p1-c3）
curl -s -X POST http://localhost:8080/rag-assistant/invoke \
  -H "Content-Type: application/json" \
  -d '{"message":"What are AcmeTech'\''s regular office hours?"}'
# SSE 流式
curl -s -N -X POST http://localhost:8080/rag-assistant/stream \
  -H "Content-Type: application/json" \
  -d '{"message":"How many days per week may employees work remotely?","stream_tokens":true}' | head -30
```

首次调用会把 23.9GB 模型载入显存+内存（8GB 显存放不下，部分层走 CPU），首个 token
可能要等数分钟——这不是故障。若超 10 分钟无响应才排查 /tmp/ollama_serve.log。

## 第 3 步：拒答正确率评测（换模型前后对比表的数据来源）

```bash
cd /c/Users/wang/Projects/agent-service-toolkit
env -u SSL_CERT_FILE .venv/Scripts/python.exe scripts/eval_refusal.py
# 25 题 × 每题 30-120s 推理，预计 30-60 分钟；报告写入 evals/refusal_report_*.md（gitignore 内）
```

## 第 4 步：README 与路线图收尾

- README 技术栈注明：生成模型 Qwen3.6（qwen3.6:35b-a3b，本地 Ollama / OpenAI 兼容端点）、
  Embedding Qwen3-Embedding-0.6B（未变）。
- 追加换模型前后对比表：检索三项与基线一致（91.4% / 97.1% / 0.938，报告
  evals/report_model_swap_regression.md）；拒答正确率用第 3 步结果（旧模型无数据，标注 N/A）。
- 勾选 docs/improvement-roadmap.md 第 1 项（注明日期与模型名）。

## 第 5 步：提交推送

```bash
git add -A && git commit -m "feat: switch generation model to local Qwen3.6 35b-a3b (roadmap #1)"
git -c http.version=HTTP/1.1 push origin HEAD
```

## 边界

- 不改 embedding/Chroma 相关代码；`.env` 不入库。
- 若冒烟发现 tool calling 或流式异常，先查 `/tmp/ollama_serve.log` 与服务日志，
  带 exit code 与日志片段停下报告，不要盲目改代码。
- 全部完成后创建标记文件 `~/OllamaModelDl/POSTMODEL_DONE`（防重复执行），并在报告中
  注明「可删除定时任务」。
