"""표 생성기의 공통 뼈대 (design 15.4절 · 6단계).

## 왜 표도 스크립트로 만드나

표를 손으로 옮겨 적으면 **실험을 다시 돌릴 때마다 옮겨 적어야 하고, 한 번은 반드시 틀린다.**
그리고 심사에서 "이 값은 어디서 나왔나"에 답할 수 없다. 표마다 스크립트를 두면 그 답이
곧 스크립트 경로다. **사람이 판단해 채우는 표(Table I)도 내용을 스크립트 안 상수로 둔다.**

## 값의 출처는 하나다

결과 표는 **5단계 진입점(`scripts/eval_methods.py`)이 남긴 요약 파일만** 읽는다.

    paper/score/{방법}_{split}/summary.json    전체·종류별 지표
    paper/score/{방법}_{split}/per_tile.csv    타일별 지표 (그림이 읽는다)

표가 `metrics.csv` 를 직접 파싱하면 판정 스크립트와 값이 갈라진다 — 그 경로를 막는 것이 요점이다.

## 설계와 다른 점 하나 — pandas 를 쓰지 않는다

설계 문서는 `pandas.DataFrame` 을 전제했지만 **이 환경에 pandas 가 없다**(09-16 확인).
잠금 환경을 건드리지 않으려고 표준 라이브러리 `csv` 와 dict 목록으로 같은 일을 한다.

## 규약

1. **두 파일을 낸다** — `.csv`(기계용) · `.md`(원고에 붙여 넣는 용도). 숫자는 같은 값이다.
2. **빈칸은 `–`(en dash)** — 0 과 "아직 안 쟀음"이 같은 칸에 보이면 안 된다.
3. **값이 없으면 빈 표를 내고 무엇이 없는지 화면에 찍는다.** 스크립트가 죽으면 안 된다.
4. **폴더 단위로 지우고 다시 만든다** — 지난 실행의 파일이 섞이지 않게.
"""

import csv
import json
import shutil
from pathlib import Path

DATA_ROOT = Path("/data/ongoing/2026_stella")
# `scripts/figures/figure_base.py` 에도 같은 값이 있다. scripts/ 는 패키지가 아니라 서로
# import 하지 못한다 — 경로를 바꾸면 두 곳을 함께 고친다.
PAPER_ROOT = DATA_ROOT / "paper"
TABLE_ROOT = PAPER_ROOT / "table"
SCORE_ROOT = PAPER_ROOT / "score"
BLANK = "–"
# 잴 수는 있는데 **정의상 값이 없는** 칸(예: 순서를 고정한 조건의 배정 모호도).
# "아직 안 쟀음"(`BLANK`)과 구별해야 빈칸 점검이 거짓 경보를 내지 않는다.
UNDEFINED = "—"


class PaperTable:
    """하위 클래스는 `columns` 와 `collect()` 만 채운다."""

    name = "table_00"
    columns: tuple[str, ...] = ()

    def run(self) -> None:
        rows = self.collect()
        out_dir = fresh(TABLE_ROOT / self.name)
        self.save(out_dir, rows)
        self.report_missing(rows)

    def collect(self) -> list[dict]:
        raise NotImplementedError

    def save(self, out_dir: Path, rows: list[dict]) -> None:
        write_csv(out_dir / f"{self.name}.csv", self.columns, rows)
        write_markdown(out_dir / f"{self.name}.md", self.columns, rows)
        print(f"[{self.name}] 저장 {out_dir} · {len(rows)}행")

    def report_missing(self, rows: list[dict]) -> None:
        """빈칸이 어디인지 화면에 찍는다 — 표가 완성됐는지 눈으로 확인하는 유일한 수단이다."""
        missing = {
            column: sum(1 for row in rows if row.get(column, BLANK) == BLANK)
            for column in self.columns
        }
        empty = {name: count for name, count in missing.items() if count}
        if not empty:
            print(f"[{self.name}] 빈칸 없음")
            return
        parts = " · ".join(f"{name} {count}칸" for name, count in empty.items())
        print(f"[{self.name}] 아직 빈 칸 — {parts}")


def fresh(path: Path) -> Path:
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    head = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, BLANK)) for column in columns) + " |" for row in rows
    ]
    path.write_text("\n".join([head, rule, *body]) + "\n", encoding="utf-8")


def read_summary(method: str, split: str) -> dict:
    """5단계 요약 파일. 없으면 빈 dict — 표는 빈칸으로 남고 죽지 않는다(규약 3)."""
    path = SCORE_ROOT / f"{method}_{split}" / "summary.json"
    if not path.exists():
        print(f"[table_base] 요약 없음: {path}")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_breakdown(method: str, split: str) -> dict:
    """종류별 분해 · 중복 선 비율(`scripts/eval_breakdown.py`). 없으면 빈 dict."""
    path = SCORE_ROOT / f"{method}_{split}" / "breakdown.json"
    if not path.exists():
        print(f"[table_base] 분해 없음: {path}")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def change(value, base) -> str:
    """기준 대비 **절대 차이**(F1 점수 그대로). 표마다 부호·자릿수를 맞추려고 한 곳에 둔다.

    퍼센트로 쓰지 않는다(교수 지시 09-30) — F1 자체가 비율이라 "+5.4%"가 점수 5.4점인지
    상대 변화인지 헷갈리고, 누적값으로도 읽힌다.
    """
    if value is None or base is None:
        return BLANK
    return f"{float(value) - float(base):+.4f}"


def number(value, digits: int = 4) -> str:
    """빈칸 기호와 자릿수를 한 곳에서 정한다 — 표마다 다르게 반올림하지 않기 위해서다."""
    if value is None:
        return BLANK
    return f"{float(value):.{digits}f}"
