"""Fig. 9 — baseline 정성 비교 (VII-A, 2단, 4행×5열). **미구현 — 설계만 적어 둔다.**

## 무엇을 그리나

한 행 = 한 장면. 열 다섯 개.

    원본 | 원본+GT | 면 기반(LaneStitch 후처리) | GTE 헤드(같은 백본) | **STELLA (ours)**

세 예측 열은 **같은 렌더링 규칙**을 쓴다 — 클래스 색 폴리라인 + 양 끝점 흰 테두리 원.

## 왜 필요한가

같은 렌더링이어야 그림에서 보이는 차이가 **알고리즘에서 온 것**으로 읽힌다.
색이나 선 굵기가 방법마다 다르면 그림 자체가 반박당한다.
그리고 끝점 원 덕분에 **"조각남"의 차이가 정성 그림에서 세진다** — baseline 이 한 차선을
네 조각으로 낸 것이 점 개수로 보인다.

## 선별 조건 (AND)

    ΔF1 = F1(ours) − max(baseline들) >= 0.20    우리가 확실히 나은 장면
    F1(ours) >= 0.50                            우리 것도 깨끗해야 한다
    baseline 조각 수 >= 1.5                     조각남의 차이가 있는 장면
    GT 인스턴스 8~25개                           읽을 수 있는 밀도

**주의.** "우리가 이긴 장면만 골랐다"는 비판을 받는다. 그래서 조건과 상한을 논문 캡션에
명시하고, Fig. 11(대표 정성 결과)이 **사분면에서 고르게 뽑아** 균형을 맞춘다.
Fig. 9는 "차이가 무엇인지"를 보이는 그림이고, Fig. 11이 "평소에 어떤지"를 보이는 그림이다.

## 정렬·상한·파일명

    정렬   ΔF1 내림차순
    상한   100장
    이름   {순위:03d}_{타일id}_d{ΔF1×1000:03d}.png

## 입력

    예측 캐시   방법 3종 각각 .../pred_cache/{실행}/test/{stem}.npz
    split       **test** — 논문이 수치를 보고하는 split 과 그림이 어긋나면 안 된다
    baseline    면 기반 · GTE 헤드 두 대조군의 예측이 같은 형식(폴리라인)으로 캐시에 있어야 한다

## 상태

**착수 전 — baseline 2건이 막고 있다.**
① 면 기반(세선화+Douglas–Peucker) baseline, ② **GTE 헤드 대조군**(같은 백본·같은 neck·
같은 학습 예산에서 출력 표현만 GTE 로 바꾼 것). ②는 이 논문의 사활이 걸린 대조군이고
Table V·VII 도 같은 실행을 쓴다.
"""

# from figure_base import SampleFigure
#
#
# class BaselineComparisonFigure(SampleFigure):
#     """세 방법의 예측을 같은 규칙으로 그리고 ΔF1 순으로 고른다."""
#
#     name = "figure_09"
#     split = "test"
#     DELTA_F1_MIN = 0.20
#     OURS_F1_MIN = 0.50
#     BASELINE_FRAG_MIN = 1.5     # baseline 이 조각을 내고 있는 장면
#     GT_MIN, GT_MAX = 8, 25
#     METHODS = ("seg_skeldp", "gte_head", "stella")   # 열 순서
#
#     def build_figure(self, stem, sample, pred):
#         raise NotImplementedError("설계는 위 docstring 참고")
