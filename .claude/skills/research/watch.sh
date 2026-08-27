#!/usr/bin/env bash
# 유휴 감시자 — improve-loop SKILL "완료 감지".
#
# 실험을 띄운 **직후** 백그라운드로 이걸 건다. 자원이 비면 즉시 종료하면서 요약을 찍는데,
# 그 종료가 완료 알림이 되어 팀장을 깨운다. 이것이 없으면 GPU가 밤새 놀아도 아무도 모른다.
#
#   사용:  bash .claude/skills/improve-loop/watch.sh [확인간격초] [유휴판정횟수]
#   기본:  60초 간격, 연속 3회 유휴면 종료 (= 3분)
set -uo pipefail

INTERVAL=${1:-60}
NEED_IDLE=${2:-3}
LOG_ROOT=/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2026_stella/log

running_trainings() { ps -eo comm,args | awk '$1 ~ /^python/ && /stella\.train\.train/' | wc -l; }
running_decode() {
    ps -eo comm,args |
        awk '$1 ~ /^python/ && /scripts\/(eval_decode|tune_decoder|dump_predictions)/' | wc -l
}
busy_gpus() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
        awk '$1 > 2000' | wc -l
}

idle=0
while true; do
    train=$(running_trainings); decode=$(running_decode); gpu=$(busy_gpus)
    if [ "$train" -eq 0 ] && [ "$decode" -eq 0 ] && [ "$gpu" -eq 0 ]; then
        idle=$((idle + 1))
    else
        idle=0
    fi
    [ "$idle" -ge "$NEED_IDLE" ] && break
    sleep "$INTERVAL"
done

echo "=== 자원이 비었다 — 다음 실험을 올려라 ==="
date
echo "GPU 4슬롯 전부 유휴, 학습 0개, D 트랙 0개"
echo
echo "=== 최근 끝난 실행 (10분 이내) ==="
find "$LOG_ROOT" -maxdepth 2 -name metrics.csv -mmin -600 -printf '%TH:%TM  %h\n' 2>/dev/null |
    sort | tail -8
echo
echo "다음: .venv/bin/python scripts/summarize_runs.py --last 8 --tail 3"
