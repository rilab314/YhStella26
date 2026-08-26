"""Table III — 파라미터 전량 명세 (V절, 2단). **미구현 — 설계만 적어 둔다.**

## 무엇이 들어가나

    기호 | 값 | **미터 환산** | 의미 | 어디서 정했나

    모델    격자 간격 s, 격자 한 변 L, 슬롯 수 R, 토큰 수 K, d_model, 층 수, 윈도우 w, n_max
    디코더  radius, align_thresh, opp_thresh, end_thresh, exist_thresh, w_dist, w_opp,
            min_class_prob, purity_thresh, end_extend, max_turn_deg, min_points, merge_gap

## 왜 필요한가

직전 논문 심사에서 **R3-3b 가 13개 항목을 열거하며 요구한 것이 정확히 이 표**다.
그리고 미터 병기가 **R2-3(GSD 명시)에 동시에 답한다**.

부수 효과가 하나 더 있다 — **의사코드에 값을 넣지 않고 기호만 쓸 수 있게 된다.**
Algorithm 1·2 가 짧아지고, 값의 단일 출처가 이 표 하나로 생긴다.

## 미터 환산

    GSD = 0.2550 m/px      1픽셀이 지상에서 몇 미터인가
    타일 768 px = 195.8 m
    격자 한 칸 4 px = 1.02 m

거리 단위 파라미터에만 미터 열을 채우고, 문턱·가중치 같은 무차원 값은 `–` 로 둔다.
**0 과 "해당 없음"이 같은 칸에 보이면 안 된다.**

## "어디서 정했나" 열이 이 표의 정직성이다

세 등급으로 적는다.

    설계    문서에서 정한 값 (R=2, D=2)
    실측    데이터를 재서 정한 값 (n_max 9,500 — 실측 최대 8,909)
    스윕    실험으로 고른 값 (radius 5, w_dist 0.072, min_points 8, merge_gap 24)

**설계 문서의 값과 실측이 다르면 실측을 따른다**는 것이 이 프로젝트의 규칙이고,
그 규칙의 결과가 이 열에 그대로 드러난다. 숨기지 않는 것이 낫다 —
`w_dist` 는 설계 문서가 "폐기"로 적어 둔 값인데 스윕에서 되살아나 F1 을 33% 올렸다.

## 값의 출처 — config 에서 뽑는다

`configs/base.py`(F 규격)의 dataclass 를 직접 읽는다. **손으로 적지 않는다** —
이 표가 논문이 보고하는 결과를 만든 그 설정과 어긋나면 표의 존재 이유가 사라진다.

기호·의미·"어디서 정했나"는 코드에서 나오지 않으므로 이 스크립트 안의 상수 표에 둔다.
**config 에 있는데 상수 표에 없는 파라미터가 나오면 경고를 찍는다** — 새 파라미터를
추가하고 표에 넣는 것을 잊는 사고를 막는다.

## 출력

    table_03/table_03.csv · table_03.md

## 상태

**착수 전.** 실험이 필요 없다. 다만 **실험이 끝나 파라미터가 확정된 뒤에 다시 돌려야** 한다.
"""

# from stella.paper.table_base import PaperTable
#
#
# class ParameterSpecTable(PaperTable):
#     """config 에서 값을 읽고, 기호·의미·근거는 상수 표에서 붙인다."""
#
#     name = "table_03"
#     COLUMNS = ("기호", "값", "미터", "의미", "어디서 정했나")
#     GSD_M_PER_PX = 0.2550
#     CONFIG_MODULE = "configs.base"
#     # SPEC = [("s", "data.grid_stride", "격자 간격", "설계", True), ...]
#     #        (기호, config 점 경로, 의미, 근거 등급, 미터 환산 여부)
#
#     def collect(self):
#         raise NotImplementedError("설계는 위 docstring 참고")
