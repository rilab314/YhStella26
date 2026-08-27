#!/usr/bin/env bash
# 자원 원장 — improve-loop SKILL 3절.
# 이 기계는 사람이 같이 쓴다. 대화형 반응성이 실험 처리량보다 우선한다.
# GPU 슬롯 4 / CPU 부하 상한 16(코어의 절반) / RAM 160 GB / 내 프로세스는 전부 nice 15.
set -uo pipefail

LOAD_CAP=16
MEM_CAP=160

echo "=== GPU 슬롯 (4개) ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits |
    awk -F', ' '{s=($3>2000)?"사용중":"비어있음"; printf "  GPU%s  %3s%%  %6s / %s MiB  %s\n",$1,$2,$3,$4,s}'

echo "=== CPU / 메모리 (사람과 공유) ==="
read -r one five fifteen < <(awk '{print $1, $2, $3}' /proc/loadavg)
printf "  부하 %s (1분) / %s (5분) / %s (15분)   코어 %s   **상한 %s**\n" \
    "$one" "$five" "$fifteen" "$(nproc)" "$LOAD_CAP"
free -g | awk -v cap=$MEM_CAP '/^Mem:/ {
    printf "  메모리 %s / %s GB 사용 (상한 %s)  가용 %s GB\n", $3, $2, cap, $7
    if ($3 > cap) print "  !! 메모리 초과 — 실험 하나를 중단한다"
}'
awk -v load="$one" -v cap=$LOAD_CAP 'BEGIN {
    room = int(cap - load); if (room < 0) room = 0; if (room > 8) room = 8
    printf "  => D 트랙 워커 여유: %d  (학습 중이면 4 이하)\n", room
    if (load > cap) print "  !! 과부하 — ① renice 15 다시 걸기 ② D 트랙 중단 ③ U arm 하나 줄이기"
}'

echo "=== 우선순위 점검 (내 프로세스는 전부 nice 15 여야 한다) ==="
bad=$(ps -eo nice,comm,args | awk '$2 ~ /^python/ && $1 < 10 && /stella\.train\.train|scripts\//' | wc -l)
if [ "$bad" -gt 0 ]; then
    echo "  !! nice 가 낮은 프로세스 $bad 개 — 아래를 실행한다"
    echo '  for pat in "stella.train.train" "run_experiments" "eval_decode" "tune_decoder" "pt_data_worker"; do'
    echo '    for pid in $(pgrep -f "$pat"); do renice -n 15 -p $pid >/dev/null 2>&1; done; done'
else
    echo "  OK — 대화형 작업이 우선권을 갖는다"
fi

echo "=== 돌고 있는 학습 ==="
ps -eo comm,args | awk '$1 ~ /^python/ && /stella\.train\.train/' |
    sed -E 's/.*--config +([^ ]+).*--tag +([^ ]+).*/  \1  tag=\2/' | sort -u | grep . || echo "  (없음)"

echo "=== 돌고 있는 D 트랙 ==="
decode=$(ps -eo comm,args | awk '$1 ~ /^python/ && /scripts\/(eval_decode|tune_decoder|dump_predictions)/' |
    sed -E 's#.*scripts/([a-z_]+)\.py.*#  \1#' | sort | uniq -c)
[ -n "$decode" ] && echo "$decode" || echo "  (없음)"
