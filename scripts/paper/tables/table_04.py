"""Table IV — 데이터셋과 클래스 (VI-A, 2단). **미구현 — 설계만 적어 둔다.**

## 무엇이 들어가나

클래스 11종 × 다섯 열.

    category_id | 클래스 이름 | 전체 인스턴스 수 | train 셀 수 비중 | 평균 길이(칸 / m)

표 아래 각주로:

    GSD 0.2550 m/px · 타일 768 px = 195.8 m
    split  train 8,979 / val 1,218 / test 2,457 (SEED-MAP **v1.2**)
    제외 규칙  geometry_type = POLYGON (노면 기호라 선이 아니다)
               category_id 599 · 5011 · None (전체 차선의 0.2%)

## 왜 필요한가

클래스별 성능(Table VIII 하단)을 해석하려면 **빈도와 길이 분포가 먼저 있어야 한다.**
"희소 클래스라 못 찾는다"와 "짧아서 못 찾는다"는 **다른 진단**이고, 이 표가 그것을 가른다.
그리고 이 논문의 VIII-1(짧은 선 문제)은 이 표의 "평균 길이" 열 없이는 근거가 없다.

## ★ v1.2 로 바뀌었다 — 다시 재야 한다

이 표는 **데이터셋 통계**라서, v1.1 → v1.2 로 바꾼 것이 직접 영향을 준다.

    train  8,979장 그대로 (v1.2 는 train 타일을 하나도 지우지 않았다)
    val    1,282 → **1,218** (−64)
    test   2,567 → **2,457** (−110)

빠진 이유: 그 검증·시험 타일이 학습 타일과 화면이 겹쳐 있었다. 타일은 196 m 정사각형인데
80~100 m 간격으로 놓여 있어 이웃끼리 절반 넘게 겹친다 — split 경계에서 픽셀이 새고 있었다.

**"전체 인스턴스 수" 열은 세 split 합계이므로 값이 바뀐다.** train 만 쓰는 열
("train 셀 수 비중", "평균 길이")은 사실상 그대로다.

## 값의 출처

    scripts/stat_labels.py 가 이미 이 통계를 낸다. 그것을 함수로 불러 쓰고,
    표 형식만 이 스크립트가 만든다. **같은 숫자를 두 곳에서 계산하지 않는다.**

    클래스 이름·category_id 매핑은 stella/data/types.py 의 `CATEGORY_ID_TO_LABEL`·
    `CLASS_NAMES` 가 단일 출처다. 스크립트가 이름을 따로 적지 않는다.

## 각주의 GSD 값에 주의

설계 문서 어딘가에 **0.1278 m/px** 로 적힌 곳이 남아 있다(README 등).
**0.2550 이 맞는 값**이고, 이 표의 미터 환산·타일 크기·좌표 변환 정확도가 전부 거기 걸려 있다.
표를 만들 때 각주의 값과 Table III 의 값이 같은지 스크립트가 검사한다.

## 출력

    table_04/table_04.csv · table_04.md

## 상태

**착수 전.** 실험이 필요 없다 — v1.2 데이터셋으로 지금 바로 만들 수 있다.
"""

# from stella.paper.table_base import PaperTable
#
#
# class DatasetClassTable(PaperTable):
#     """클래스별 빈도·길이 분포. scripts/stat_labels.py 의 통계를 표로 만든다."""
#
#     name = "table_04"
#     COLUMNS = ("category_id", "클래스", "인스턴스 수", "train 셀 비중", "평균 길이(칸)",
#                "평균 길이(m)")
#     GSD_M_PER_PX = 0.2550
#     GRID_STRIDE_PX = 4
#     SPLITS = ("train", "val", "test")
#
#     def collect(self):
#         raise NotImplementedError("설계는 위 docstring 참고")
