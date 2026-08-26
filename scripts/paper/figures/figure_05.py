"""Fig. 5 — 순서 없는 슬롯 배정 (III-C, 1단, 3패널 세로). **미구현 — 설계만 적어 둔다.**

## 무엇을 그리나

    (a) 고정 각도 섹터   같은 셀에서 선을 조금 회전시킨 두 경우 —
                        타깃 채널 인덱스가 바뀌는 것을 표시
    (b) 고정 prev/next   같은 기하인데 주석 방향이 반대면 타깃이 뒤바뀌는 것
    (c) ours            2×2 비용 행렬과 두 순열의 총비용, argmin 표시.
                        그 아래 작은 곡선 — 슬롯 배정이 얼마나 흔들리는지의 학습 곡선
                        (코드 이름 `match_ambiguity`)

## 왜 필요한가

"슬롯이 순서가 없다"는 말 자체는 추상적이다. (a)(b)가 **문제**를, (c)가 **해법과 그 해법이
실제로 작동한 증거**를 한 그림에 담는다. 배정 흔들림 곡선이 있어야 "매칭을 넣었더니 배정이
안정됐다"가 주장이 아니라 **관측**이 된다.

## 제작 방식

    (a) 개념도 + Fig. 2(c) 스크립트가 뽑은 실측 채널 인덱스를 숫자로 병기
    (b) 개념도 (그리기 도구)
    (c) 위쪽 2×2 비용 행렬은 개념도, **아래쪽 곡선은 실측** — 이 스크립트가 만드는 부분

## (c) 곡선 — ★ 실측 완료

    파일    docs/figures/fig5c_match_ambiguity.png (이미 있다)
    출처    F01(전체 데이터·현행 설정) 실행의 metrics.csv, val 곡선
    값      0.126 (에폭 0) → 0.0004 (에폭 3). 로그 축에서도 3에폭 만에 바닥.
    주의    옛 설계안이 인용하던 "0.503 → 0.099(에폭 22)"는 **더 이른 실험 단계의 다른
            설정** 값이다. 실측이 더 낫다 — 배정은 "서서히 안정"이 아니라 **거의 즉시 수렴**.

이 스크립트는 그 곡선을 **재현 가능하게 다시 만드는 것**이 목적이다. 현재 저장소의 PNG 는
세션 스크래치에서 만든 것이라 재현 스크립트가 없다 — 그 구멍을 메운다.

    입력   log/{실행}/metrics.csv        ← stella/eval/runlog.py 로 읽는다
    계산   에폭별 `val/conn/match_ambiguity`
    출력   figure_05/figure_05c.png · .pdf · .csv (에폭, 값)
    축     y 로그 스케일 (0.0004 를 선형 축에 그리면 바닥에 붙어 안 보인다)

## 상태

**(c) 실측 완료 · 재현 스크립트 미구현.** (a)(b) 개념도는 원고 그림 작업에서 만든다.
"""

# from stella.paper.figure_base import PlotFigure
#
#
# class MatchAmbiguityCurve(PlotFigure):
#     """(c) 아래쪽 곡선: 슬롯 배정 흔들림이 에폭에 따라 어떻게 떨어지는지."""
#
#     name = "figure_05"
#     METRIC_KEY = "val/conn/match_ambiguity"
#     RUN_KEYWORD = "F01"     # 전체 데이터·현행 설정 실행
#     LOG_SCALE = True
#     width_in, height_in = 3.5, 2.0   # 1단
#
#     def collect(self):
#         raise NotImplementedError("설계는 위 docstring 참고")
