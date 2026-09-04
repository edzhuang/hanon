#!/usr/bin/env bash
# Sample N pieces per brief in hanon/prompt.py, all briefs in parallel.
# Usage: scripts/sample_all.sh [N]   (default 32 → 256 pieces over 8 briefs)
set -u
cd "$(dirname "$0")/.."
N="${1:-32}"
mkdir -p out/logs
i=0
# Read briefs into an array first: a `| while read` loop runs in a subshell, and
# `wait` in the parent would not see jobs started there.
IFS=$'\n' read -r -d '' -a BRIEFS < <(uv run python -c "from hanon.prompt import PROMPTS; print('\n'.join(PROMPTS))" && printf '\0')
for brief in "${BRIEFS[@]}"; do
  i=$((i+1))
  PYTHONUNBUFFERED=1 uv run scripts/play.py "$brief" -n "$N" > "out/logs/brief_$i.log" 2>&1 &
done
wait
echo "all briefs done" > out/logs/DONE
