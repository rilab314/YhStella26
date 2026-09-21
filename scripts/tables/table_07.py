"""Table VII — 백본·구조 민감도 + 종류별 성능 (VI-E · VI-F, 2단).

**표현을 고정한 채 주변 설정만 흔든다** — 표현 축을 흔드는 Table VI 와 역할이 다르다.
한 표에 블록 두 개라 파일도 둘이다(`table_07_upper` · `table_07_lower`).

## 상단 — 구조 민감도 (U 규격: train 3,000장 · val 700장 · 10에폭)

전체 데이터로 열세 번 돌릴 수 없어 작은 규격으로 비교했다. **절대값을 Table III 과 나란히
읽으면 안 된다** — 캡션에 규격을 밝힌다. 각 블록 안에서 첫 행 대비 변화만 읽는다.

- **백본 블록**: 크기가 아니라 계열이 중요하다 — Swin-L(195 M)은 채택 백본(196 M)과 크기가
  같은데 크게 진다. "큰 모델이라 이겼다"는 반박을 막는 줄이다. 최고 백본을 찾는 것이 목적이
  아니다(직전 심사 R2-2c) — 캡션에 그 취지를 쓴다.
- **구조 블록**: 어텐션 창 7칸이면 충분하고, 층을 줄이면 실제로 손해다(효율 근거로 못 쓴다).
- **연결 슬롯 3개는 여기 싣지 않는다** — 같은 축을 Table VI 가 F 규격으로 싣는다(−24.3%).

값의 출처는 5단계 요약 파일(`eval_methods.py --method u_*`) — 모든 행을 **현행 디코더로 다시
디코딩한** 값이다. 학습 당시 디코더로 잰 옛 판정은 쓰지 않는다.

**최대 GPU 메모리 · 스텝 시간 열은 아직 비어 있다** — 같은 기계·같은 배치로 따로 재야 한다.

## 하단 — 종류별 성능 (최종 모델 · 전량)

정답 선 수를 함께 싣는다 — 자전거도로는 27장 149개뿐이라 그 칸의 점수는 표본이 적어 흔들린다.
독자가 그 칸을 믿을지 판단할 수 있게 한다.

**거짓 양성(FP)을 두 갈래로 가른다** — HD map 을 사람이 고치는 관점에서 비용이 다르다.

    중복 FP    같은 종류의 정답 선 위에 이미 올라가 있는 여분 — 지우면 되는 값싼 오류
    헛검출 FP   같은 종류의 어떤 정답 선 위에도 없는 것 — 사람이 확인해야 하는 비싼 오류

값의 출처는 `eval_breakdown.py` — **종류마다 그 종류의 예측·정답만 남겨** 채점한다. 그래서
coverage·조각 수·FP 두 갈래가 "같은 종류끼리"의 값이고, 전체 행(전 종류 합산 채점)과 정의가
조금 다르다(전체 행의 중복 FP 는 다른 종류의 정답 위에 올라간 것도 센다). 각주로 적는다.

사용:
    python scripts/tables/table_07.py --split val
"""

import argparse

from table_base import (
    BLANK,
    TABLE_ROOT,
    PaperTable,
    change,
    fresh,
    number,
    read_breakdown,
    read_summary,
    write_csv,
    write_markdown,
)

# (블록, 논문 행 이름, 방법). 블록마다 첫 행이 기준이다. 방법 이름은 `eval_methods.py` 의 METHODS.
UPPER_ROWS = (
    ("백본", "ConvNeXtV2-B (기준)", "u_cnxb"),
    ("백본", "ConvNeXtV2-L (채택)", "u_cnxl"),
    ("백본", "DINOv3-sat ViT-L", "u_dinov3"),
    ("백본", "DINOv3-sat ViT-L (학습률 ×0.2)", "u_dinov3_lr02"),
    ("백본", "Swin-B", "u_swinb"),
    ("백본", "Swin-L", "u_swinl"),
    ("백본", "ConvNeXtV2-B (동결)", "u_freeze"),
    ("구조", "대조군 (창 7 · 층 6)", "u_ref"),
    ("구조", "어텐션 창 9칸", "u_win9"),
    ("구조", "어텐션 층 4개", "u_deep4"),
    ("구조", "어텐션 층 3개", "u_deep3"),
    ("구조", "백본 HRNet-W48 (고해상도 유지)", "u_hrnet"),
)
UPPER_COLUMNS = ("블록", "조건", "inst F1", "ΔF1", "타일 수", "최대 메모리(GB)", "스텝(ms)")

FINAL_METHOD = "stella"
# 종류 이름 (코드 이름 → 논문 표기). 표의 행 순서는 F1 내림차순으로 정한다.
CLASS_LABELS = {
    "center_line": "중앙선",
    "u_turn_zone_line": "유턴구역선",
    "lane_line": "차선",
    "bus_only_lane": "버스전용차로",
    "edge_line": "길가장자리구역선",
    "path_change_restriction_line": "진로변경제한선",
    "no_parking_stopping_line": "주정차금지선",
    "guiding_line": "유도선",
    "stop_line": "정지선",
    "bicycle_lane": "자전거도로",
}
LOWER_COLUMNS = (
    "종류",
    "코드 이름",
    "정답 선 수",
    "예측 선 수",
    "inst F1",
    "coverage",
    "frag",
    "중복 FP",
    "헛검출 FP",
)


class Sensitivity(PaperTable):
    """두 블록을 한 폴더에 낸다 — 폴더 단위 덮어쓰기 규약을 지키려고 `run` 을 한 번만 부른다."""

    name = "table_07"

    def __init__(self, split: str):
        self.split = split

    def run(self) -> None:
        out_dir = fresh(TABLE_ROOT / self.name)
        for suffix, columns, rows in (
            ("upper", UPPER_COLUMNS, self.collect_upper()),
            ("lower", LOWER_COLUMNS, self.collect_lower()),
        ):
            stem = f"{self.name}_{suffix}"
            write_csv(out_dir / f"{stem}.csv", columns, rows)
            write_markdown(out_dir / f"{stem}.md", columns, rows)
            self.columns = columns
            self.report_missing(rows)
        print(f"[{self.name}] 저장 {out_dir}")

    def collect_upper(self) -> list[dict]:
        rows, bases = [], {}
        for block, label, method in UPPER_ROWS:
            payload = read_summary(method, "val")
            f1 = payload.get("summary", {}).get("f1")
            is_base = block not in bases
            bases.setdefault(block, f1)
            rows.append(
                upper_row(block, label, payload, "기준" if is_base else change(f1, bases[block]))
            )
        return rows

    def collect_lower(self) -> list[dict]:
        per_class = read_breakdown(FINAL_METHOD, self.split).get("per_class", {})
        names = sorted(
            (name for name in CLASS_LABELS if name in per_class),
            key=lambda name: per_class[name]["f1"],
            reverse=True,
        )
        return [*(lower_row(name, per_class[name]) for name in names), self.total_row(per_class)]

    def total_row(self, per_class: dict) -> dict:
        """전체 행 — 점수는 주 비교표와 같은 전 종류 합산 채점, 개수는 종류별 합."""
        payload = read_summary(FINAL_METHOD, self.split)
        scores = payload.get("summary", {})
        counts = {key: sum(item[key] for item in per_class.values()) for key in ("n_gt", "n_pred")}
        return {
            "종류": "전체",
            "코드 이름": "all",
            "정답 선 수": str(counts["n_gt"]) if per_class else BLANK,
            "예측 선 수": str(counts["n_pred"]) if per_class else BLANK,
            "inst F1": number(scores.get("f1")),
            "coverage": number(scores.get("coverage")),
            "frag": number(scores.get("frag")),
            "중복 FP": count(scores.get("fp_redundant")),
            "헛검출 FP": count(scores.get("fp_spurious")),
        }


def upper_row(block: str, label: str, payload: dict, delta: str) -> dict:
    return {
        "블록": block,
        "조건": label,
        "inst F1": number(payload.get("summary", {}).get("f1")),
        "ΔF1": delta if payload else BLANK,
        "타일 수": str(payload.get("tiles", BLANK)),
    }


def lower_row(name: str, scores: dict) -> dict:
    return {
        "종류": CLASS_LABELS[name],
        "코드 이름": name,
        "정답 선 수": str(scores["n_gt"]),
        "예측 선 수": str(scores["n_pred"]),
        "inst F1": number(scores["f1"]),
        "coverage": number(scores["coverage"]),
        "frag": number(scores["frag"]),
        "중복 FP": count(scores["fp_redundant"]),
        "헛검출 FP": count(scores["fp_spurious"]),
    }


def count(value) -> str:
    return BLANK if value is None else str(int(value))


def main() -> None:
    args = parse_args()
    Sensitivity(args.split).run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", help="하단은 7단계 이후 test 로 낸다")
    return parser.parse_args()


if __name__ == "__main__":
    main()
