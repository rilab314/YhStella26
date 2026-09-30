"""Table I — 표현 방식 비교 매트릭스 (II절 끝, 2단).

**실험 결과가 아니라 저자의 문헌 판단이다.** 그래서 내용을 이 파일의 상수(`ROWS`)로 두고 코드는
렌더링만 한다 — "표를 고치려면 어느 파일을 고치나"의 답이 하나로 남는다
(`docs/6_paper_assets.md` 15.4).
내용의 단일 출처는 `docs/paper_outline.md` §4 Table I 이다. **둘이 다르면 설계안을 먼저 고치고
여기로 옮긴다.**

## 왜 필요한가

Related Work 를 문단으로만 쓰면 독자가 "그래서 뭐가 다른가"를 스스로 조립해야 한다. 이 표는
**기여 주장을 열 이름으로 번역한 것**이다 — "순서 모호성 처리"와 "인스턴스 정체성" 두 열이
우리만 다르다는 사실이 표 한 장에서 끝난다.

## "후처리" 열은 유무가 아니라 **무엇을 할 수 있는가**로 채운다

우리도 후처리가 있으므로 "없음"이라고 쓰면 거짓이다. 설계안은 세 등급(정체성을 만든다 /
연결을 만든다 / 중복만 줄인다)으로 적기로 정했는데, **설계안의 표 본문은 아직 옛 표기**
(NMS · 무거움 · 경량 · 중간)로 남아 있다. 그 칸을 세 등급으로 바꾸는 것은 문헌 판단이라
저자가 정한다 — 바꾸면 설계안과 이 상수를 함께 고친다.

**칸을 고칠 때는 `../materials/papers/` 의 PDF 를 다시 확인한다** — 검색 요약으로 채우지 않는다.

사용:
    python scripts/tables/table_01.py
"""

from table_base import PaperTable

COLUMNS = (
    "방법",
    "출력 1차 단위",
    "인스턴스 정체성",
    "연결성 표현",
    "순서 모호성 처리",
    "후처리",
    "클래스",
    "인스턴스 수",
)
# 설계안 §4 Table I 을 그대로 옮겼다. 강조 표시(굵게)는 원고 조판에서 붙인다.
ROWS = (
    (
        "CLRNet / CLRerNet",
        "앵커 라인",
        "모델",
        "(없음)",
        "시점 가정으로 고정",
        "NMS",
        "○",
        "고정 소수",
    ),
    (
        "MapTR / MapTRv2",
        "질의 폴리라인",
        "모델",
        "(없음)",
        "등가 순열군",
        "없음",
        "○",
        "질의 수로 고정",
    ),
    (
        "Seg + watershed (OpenSatMap)",
        "픽셀",
        "후처리",
        "연결 성분",
        "—",
        "무거움",
        "○",
        "가변",
    ),
    (
        "Sat2Graph (GTE)",
        "셀",
        "없음(전역 그래프)",
        "셀당 6슬롯 (edgeness, dx, dy)",
        "각도 섹터 고정",
        "경량",
        "✕",
        "—",
    ),
    (
        "SAM-Road / DeH4R",
        "정점",
        "없음(전역 그래프)",
        "쌍(pair) 분류기",
        "—",
        "경량",
        "✕",
        "—",
    ),
    (
        "STELLA (ours)",
        "셀",
        "모델(사슬)",
        "셀당 R=2 단위방향",
        "셀 단위 최소비용 배정",
        "경량",
        "○",
        "가변",
    ),
)


class RepresentationMatrix(PaperTable):
    name = "table_01"
    columns = COLUMNS

    def collect(self) -> list[dict]:
        return [dict(zip(COLUMNS, row, strict=True)) for row in ROWS]


def main() -> None:
    RepresentationMatrix().run()


if __name__ == "__main__":
    main()
