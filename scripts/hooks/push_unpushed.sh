#!/usr/bin/env bash
# Stop（回合结束）：把当前分支已提交但未推送的 commit 推送到 origin
# （https://github.com/wuchenxiaodao/docpilot-rag-agent）。
# 只推送已提交的内容，从不自动提交；任何异常都以 0 退出，不阻塞会话。
set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT" 2>/dev/null || exit 0
[ -d .git ] || exit 0

LOG=.git/zcode-hooks.log
branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null) || exit 0
[ -n "$branch" ] && [ "$branch" != "HEAD" ] || exit 0

# 分支还没有上游时，首次推送并建立跟踪
if ! git rev-parse --abbrev-ref "@{u}" >/dev/null 2>&1; then
  if git push -u origin "$branch" >>"$LOG" 2>&1; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] pushed new branch $branch to origin" >> "$LOG"
  fi
  exit 0
fi

if [ "$(git rev-parse HEAD)" != "$(git rev-parse '@{u}')" ]; then
  if git push origin "HEAD:$branch" >>"$LOG" 2>&1; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] pushed $branch to origin" >> "$LOG"
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] PUSH FAILED on $branch (see output above)" >> "$LOG"
  fi
fi
exit 0
