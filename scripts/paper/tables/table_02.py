"""Table II — 기호와 출력 계약 (III절, 1단). **미구현 — 설계만 적어 둔다.**

## 무엇이 들어가나

모델 출력과 GT 를 **나란히** 놓아 짝을 보인다. 열 다섯 개.

    필드 | dtype·shape | 범위 | 원점 | 감독 방식

    모델 출력                     GT
    heatmap_logit    ←→   (히트맵 타깃)
    class_logit      ←→   class_map
    self_coord       ←→   coord_map
    end_logit        ←→   end_map
    exist_logit      ←→   (conn_dirs 의 유효 슬롯 수)
    conn_dir         ←→   conn_dirs

## 왜 필요한가

"GT 와 출력이 같은 모양"이라는 이 논문 주장의 **정밀한 형태**이고, 재현에 필요한 최소 명세다.
그림(Fig. 4)이 그것을 보여 주고, 이 표가 그것을 못 박는다.

## 값의 출처 — 코드에서 뽑는다

shape·dtype 을 손으로 적으면 config 를 바꿨을 때 표가 조용히 틀린 값을 갖게 된다.

    모델 출력  stella/model/stella.py 의 `ModelOutput` — 실제 텐서를 한 번 만들어 shape 을 읽는다
    GT        stella/data/types.py 의 출력 계약 + stella/data/encode.py 의 인코더 출력
    범위·원점  코드에서 자동으로 나오지 않는다 — 이 스크립트 안의 상수 표에 둔다
              (예: `self_coord` 는 [0, 1), 원점은 셀 좌상단)
    감독 방식  어느 손실이 그 필드를 보는가. stella/loss/criterion.py 의 조립을 보고 적는다

**shape 은 실측, 나머지는 상수.** 어느 것이 어느 쪽인지 CSV 에 열을 두어 표시한다 —
나중에 이 표를 고칠 사람이 "이 값은 코드에서 나오나 손으로 쓰나"를 알아야 한다.

## 좌표 규약을 각주로 단다

    셀 인덱스만 (i, j) = (행, 열). 그 외 모든 2차원 벡터는 (x, y) 순서.
    셀 내 좌표의 원점 = 셀 좌상단, 연결 방향의 원점 = 셀 중심(자기 노드 점).
    인코더는 픽셀의 면적 중심(+0.5)을 쓰고, 디코더는 출력 직전에 −0.5 해서 라벨 좌표계로 돌린다.

이 각주가 없으면 재현하는 사람이 반드시 반 픽셀 어긋난다.

## 출력

    table_02/table_02.csv · table_02.md

## 상태

**착수 전.** 실험이 필요 없다 — config 하나만 있으면 지금 만들 수 있다.
"""

# from stella.paper.table_base import PaperTable
#
#
# class OutputContractTable(PaperTable):
#     """모델 출력과 GT 를 짝지어 놓은 명세표. shape 은 코드에서, 의미는 상수에서."""
#
#     name = "table_02"
#     COLUMNS = ("필드", "dtype·shape", "범위", "원점", "감독 방식")
#     # PAIRS = [("heatmap_logit", None), ("class_logit", "class_map"), ...]
#
#     def collect(self):
#         raise NotImplementedError("설계는 위 docstring 참고")
