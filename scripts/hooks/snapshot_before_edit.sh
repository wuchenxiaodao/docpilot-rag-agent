#!/usr/bin/env bash
# PreToolUse(Edit|Write)：大改动开始前，把工作区未提交的改动做一次快照提交。
# 10 分钟节流，避免每次编辑都产生噪音提交。永不阻塞编辑：任何异常都以 0 退出。
set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT" 2>/dev/null || exit 0
[ -d .git ] || exit 0

STAMP=".git/zcode-snapshot-stamp"
THROTTLE=600  # 秒

now=$(date +%s)
last=$(cat "$STAMP" 2>/dev/null || echo 0)
case "$last" in (*[!0-9]*|"") last=0;; esac
[ $((now - last)) -ge "$THROTTLE" ] || exit 0

if [ -z "$(git status --porcelain)" ]; then
  date +%s > "$STAMP"
  exit 0
fi

git add -A -- . 2>/dev/null
if git commit -m "chore(snapshot): auto-save working tree before edits" --no-verify -q 2>/dev/null; then
  date +%s > "$STAMP"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] snapshot commit created" >> .git/zcode-hooks.log
fi
exit 0
