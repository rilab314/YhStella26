"""Table II — 모델 출력 명세 (III-A, 1단).

**모델이 내놓는 값만** 싣는다. 열은 넷 — 슬롯 종류 · 필드 · 역할 · 범위
(`docs/paper_outline.md` §4 Table II). GT 필드는 싣지 않는다 — 출력과 데이터 인코딩을 한 표에
섞으면 III-A 가 헷갈린다는 지적을 받았다.

## 값의 출처 — 뜻은 상수, 이름과 개수는 코드

역할·범위는 코드에서 나오지 않으므로 이 파일의 상수로 둔다. 대신 **코드와 어긋날 수 있는 것은
코드에서 읽는다.**

    필드 이름    `stella.model.stella.ModelOutput` 에 그 필드가 실제로 있는지 확인한다
    R (연결 슬롯 수)   config `model.num_conn_slots`
    C (종류 수)       config `data.num_classes` (배경 포함)

필드가 코드에서 사라지거나 이름이 바뀌면 **표를 내지 않고 멈춘다** — 표가 조용히 틀린
명세를 싣는 것보다 낫다.

## 좌표 규약은 각주로 단다

셀 인덱스만 (행, 열), 그 밖의 2차원 벡터는 (x, y). `coord` 의 원점은 칸 좌상단, `dir` 의 원점은
그 칸의 self 점이다. **원점이 필드마다 다르다는 점이 재현할 때 가장 자주 틀리는 곳이다.**

사용:
    python scripts/tables/table_02.py
"""

from dataclasses import fields

from table_base import PaperTable

from stella.config_io import load_config
from stella.model.stella import ModelOutput

COLUMNS = ("슬롯 종류", "필드", "역할", "범위", "코드 필드")
# (슬롯 종류, 논문 필드 이름, 역할, 범위, ModelOutput 필드).
# 슬롯 종류·역할의 {R}·{C} 는 config 값으로 채운다.
SPEC = (
    (
        "히트맵 (격자 전 칸)",
        "heatmap",
        "이 칸에 선이 지나는가 — 노드 후보를 고르는 데 쓴다",
        "확률 [0,1]",
        "heatmap_logit",
    ),
    (
        "self 슬롯 (칸당 1개)",
        "class",
        "어느 종류의 선인가 — 배경을 포함해 C={C}종",
        "C차원 확률분포",
        "class_logit",
    ),
    (
        "self 슬롯 (칸당 1개)",
        "coord",
        "칸 안 어디를 지나는가 — 원점은 칸 좌상단",
        "[0,1)²",
        "self_coord",
    ),
    (
        "self 슬롯 (칸당 1개)",
        "end",
        "이 칸이 사슬의 끝인가 — 디코더의 정지·연장 판단에 쓴다",
        "확률 [0,1]",
        "end_logit",
    ),
    (
        "연결 슬롯 (칸당 R={R}개, 순서 없음)",
        "exist",
        "이 방향으로 선이 실제로 이어지는가",
        "확률 [0,1]",
        "exist_logit",
    ),
    (
        "연결 슬롯 (칸당 R={R}개, 순서 없음)",
        "dir",
        "어느 방향으로 이어지는가 — 원점은 그 칸의 self 점",
        "단위벡터, ‖d‖=1",
        "conn_dir",
    ),
)


class OutputSpec(PaperTable):
    name = "table_02"
    columns = COLUMNS

    def __init__(self, config: str):
        self.cfg = load_config(config, [])

    def collect(self) -> list[dict]:
        check_fields()
        sizes = {"R": self.cfg.model.num_conn_slots, "C": self.cfg.data.num_classes}
        return [
            {
                "슬롯 종류": slot.format(**sizes),
                "필드": field,
                "역할": role.format(**sizes),
                "범위": value_range,
                "코드 필드": code,
            }
            for slot, field, role, value_range, code in SPEC
        ]


def check_fields() -> None:
    """표에 적은 필드가 모델 출력에 실제로 있어야 한다."""
    present = {item.name for item in fields(ModelOutput)}
    missing = [code for *_, code in SPEC if code not in present]
    if missing:
        raise SystemExit(f"[table_02] ModelOutput 에 없는 필드: {missing} — 표를 먼저 고친다")


def main() -> None:
    OutputSpec("configs.base").run()


if __name__ == "__main__":
    main()
