"""인스턴스 파일 왕복과 타일별 지표 (5단계 — 출력 형식 통일).

**왕복이 값을 바꾸지 않아야** 그림이 읽은 선과 표가 채점한 선이 같다고 말할 수 있다.
그리고 **타일별 합계가 전체 채점과 같아야** 그림 선별 기준과 논문 표가 어긋나지 않는다.
"""

import numpy as np
import pytest

from stella.config_io import load_config
from stella.eval.instance_io import load_instances, save_instances
from stella.eval.per_tile import ROW_FIELDS, TileScorer, write_rows


def instances() -> list[dict]:
    return [
        {"class": 1, "points": np.array([[0.0, 0.0], [4.0, 0.0], [8.0, 0.5]], np.float32)},
        {"class": 5, "points": np.array([[2.0, 9.0], [2.5, 13.0]], np.float32)},
    ]


def test_roundtrip_keeps_values(tmp_path):
    save_instances(tmp_path / "tile.npz", instances())
    loaded = load_instances(tmp_path / "tile.npz")
    assert [item["class"] for item in loaded] == [1, 5]
    for before, after in zip(instances(), loaded):
        assert np.allclose(before["points"], after["points"])


def test_roundtrip_handles_empty(tmp_path):
    """예측이 하나도 없는 타일도 있다 — 그 파일에서 죽으면 안 된다."""
    save_instances(tmp_path / "empty.npz", [])
    assert load_instances(tmp_path / "empty.npz") == []


def test_tile_rows_have_fixed_fields(tmp_path):
    cfg = load_config("configs.base", [])
    scorer = TileScorer(cfg)
    row = scorer.row("tile_a", instances(), instances())
    assert set(ROW_FIELDS) <= set(row)
    assert row["n_gt"] == 2 and row["n_pred"] == 2 and row["n_class"] == 2
    assert row["f1"] == pytest.approx(1.0)  # 예측이 정답과 같으면 만점이다
    write_rows(tmp_path / "per_tile.csv", [row])
    assert (tmp_path / "per_tile.csv").read_text(encoding="utf-8").startswith("tile,")


def test_total_matches_sum_of_tiles():
    """타일마다 따로 잰 것과 전체 누적이 같은 자를 쓰는지 — 두 장으로 확인한다."""
    cfg = load_config("configs.base", [])
    scorer = TileScorer(cfg)
    rows = [scorer.row(f"tile_{k}", instances(), instances()) for k in range(2)]
    summary = scorer.summary()
    assert summary["f1"] == pytest.approx(1.0)
    assert all(row["f1"] == pytest.approx(1.0) for row in rows)
