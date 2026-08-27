#!/usr/bin/env bash
# PID가 끝날 때까지 기다린다 — improve-loop SKILL 6.5 규칙 1.
#
# **왜 이 스크립트가 있나.** `while pgrep -f <패턴>` 로 기다리면 패턴이 자기 명령줄에도
# 들어 있어 조건이 영원히 참이 된다. 실제로 그래서 무한 루프에 빠졌고, 종료하지 않았으니
# 완료 알림도 없었고, GPU 4장이 12시간 놀았다. 숫자 PID는 자기 자신을 매칭할 수 없다.
#
#   사용:  bash .claude/skills/improve-loop/waitfor.sh <PID> [간격초] [설명]
#
# **먼저 6.5 규칙 0을 확인한다** — 긴 작업은 `nohup &` 대신 그 명령 자체를 백그라운드
# 태스크로 띄우는 것이 낫다. 그러면 종료가 곧 알림이라 이 스크립트조차 필요 없다.
set -uo pipefail

PID=${1:-}
INTERVAL=${2:-30}
LABEL=${3:-"PID $PID"}

if [ -z "$PID" ]; then
    echo "사용: waitfor.sh <PID> [간격초] [설명]" >&2
    exit 2
fi
if ! [[ "$PID" =~ ^[0-9]+$ ]]; then
    echo "PID는 숫자여야 한다 (패턴 매칭은 자기 자신을 세므로 금지): '$PID'" >&2
    exit 2
fi
if ! kill -0 "$PID" 2>/dev/null; then
    echo "PID $PID 는 이미 죽었다 — 기다릴 것이 없다"
    exit 0
fi

STARTED=$(date +%s)
while kill -0 "$PID" 2>/dev/null; do
    sleep "$INTERVAL"
done
ELAPSED=$(( $(date +%s) - STARTED ))

echo "=== 끝났다: $LABEL  (대기 ${ELAPSED}초) ==="
date
echo
echo "GPU:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "부하: $(cut -d' ' -f1-3 /proc/loadavg)"
echo
echo "다음: 자원이 비었으면 대기열 최상단을 올린다 (SKILL 3절)"
