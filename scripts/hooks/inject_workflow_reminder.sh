#!/usr/bin/env bash
# UserPromptSubmit：每次用户提问时注入 Git 工作约定，保证跨会话不忘。
# 输出严格 JSON，仅含 additionalContext 一个键。
cat <<'EOF'
{"additionalContext": "Git 工作约定（用户要求）：1) 开始大型修改前，先确保当前改动已提交（快照钩子会自动做，也可手动提交）；2) 每完成 docs/improvement-roadmap.md 中的一个模块，必须创建正式 commit 并推送到 origin（https://github.com/wuchenxiaodao/docpilot-rag-agent）；3) commit message 需说明完成的是哪个模块。"}
EOF
exit 0
