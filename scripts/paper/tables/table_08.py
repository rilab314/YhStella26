"""Table VIII — 구조·디코더 ablation + 클래스별 성능 (VII-D·VII-E, 2단).
**미구현 — 설계만 적어 둔다.**

## 무엇이 들어가나 — 블록 두 개를 한 표에

**상단: 구조 민감도**

    백본 3종        ConvNeXtV2-B / SwinV2 / DINOv3-sat ViT-L
    어텐션 층 수     0 / 2 / 6
    윈도우 크기 w    7 / 9
    열              인스턴스 F1 · 최대 GPU 메모리 · 스텝당 ms

**하단: 클래스별 성능**

    클래스 11종 × (GT 수, 예측 수, F1, 커버리지, 조각 수, **중복 FP / 헛detection FP**)

## 왜 필요한가

**상단은 "구조는 표준이다"를 뒷받침한다.** 백본을 바꿔도 표현의 이득이 남는다는 것이 요점이지
**최고 백본을 찾는 것이 목적이 아니다.** 직전 논문 심사에서 **R2-2c 가 "백본 비교는 SOTA
비교가 아니다"** 라고 못 박은 바 있다. 캡션에 그 취지를 쓴다.

**하단의 FP 분해가 실용적 논지다.** 틀린 예측을 두 가지로 가른다.

    중복 FP      이미 맞힌 선 위에 하나 더 그린 것 — 지우면 되는 값싼 오류
    헛detection  아무것도 없는 곳에 그린 것 — 사람이 확인해야 하는 비싼 오류

HD map 을 사람이 보정하는 관점에서 **이 둘의 비용이 완전히 다르다.** 같은 F1 이라도
중복이 대부분이면 실용적으로 훨씬 낫다.

## 두 regime 이 보여야 한다

클래스 11종은 성격이 둘로 갈린다.

    길고 뚜렷한 선   중앙선·차선 경계 — 길고 수가 적다
    짧고 붐비는 선   횡단보도·정지선 — 짧고 한 타일에 여럿 있다

**두 무리의 F1 차이가 VIII-1(짧은 선 문제)의 근거**다. 표에서 클래스를 이 순서로 묶어
정렬하면 그 대비가 눈에 보인다. Table IV(클래스별 평균 길이)와 나란히 읽히게 만든다.

## 지면이 모자라면 상단을 보충자료로 옮긴다

§6.2 감축안 3번 — 상단(구조 ablation)을 보충자료로 보내고 본문에는 2문장 요약만 둔다
(0.16쪽 절약). 하단(클래스별)은 남긴다.

## 값의 출처

    상단   실행 6~8개의 log/{실행}/metrics.csv  ← stella/eval/runlog.py
           메모리·ms 는 같은 기계·같은 배치에서 잰다
    하단   최종 채택 실행의 예측 캐시를 클래스별로 평가
           중복/헛detection 분해는 **정의를 스크립트에 명시한다** —
           중복 = 같은 클래스의 다른 예측선과 자기 길이의 80% 이상이 8 px 안에서
           나란히 가는 것 (Table IX 의 "중복 선 비율"과 같은 정의)
    split  **test**

## 규격

상단은 **U 규격**, 하단은 **F 규격**이다. **한 표에 두 규격이 들어간다** —
블록이 분리돼 있고 서로 비교하지 않으므로 허용되지만, 각 블록 제목에 규격을 쓴다.

## 출력

    table_08/table_08.csv · table_08.md
    (두 블록이라 CSV 는 table_08_arch.csv · table_08_class.csv 로 나눈다)

## 상태

**상단 착수 가능 · 하단은 최종 실행 대기.** `A1` 라운드(백본·윈도우·격자 6종)가 상단을 채운다.
"""

# from stella.paper.table_base import PaperTable
#
#
# class ArchitectureAndClassTable(PaperTable):
#     """상단 구조 민감도(U 규격) + 하단 클래스별 성능(F 규격)."""
#
#     name = "table_08"
#     SPLIT = "test"
#     ARCH_AXES = ("backbone", "num_attn_layers", "window_size")
#     ARCH_METRICS = ("f1", "peak_mem_gb", "step_ms")
#     CLASS_METRICS = ("n_gt", "n_pred", "f1", "coverage", "frag", "fp_redundant", "fp_spurious")
#     REDUNDANT_OVERLAP = 0.80    # 자기 길이의 이 비율 이상이 겹치면 중복
#     REDUNDANT_GAP_PX = 8.0      # 이 거리 안에서 나란히 가면 겹친 것
#
#     def collect(self):
#         raise NotImplementedError("설계는 위 docstring 참고")
