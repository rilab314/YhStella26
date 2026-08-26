"""조각 병합·예측 캐시 (가설 백로그 · D 트랙).

병합은 **잘못 이으면 조각남보다 나쁘다**(환각 FP가 된다). 그래서 "이어야 할 것을 잇는다"와
"이으면 안 되는 것을 안 잇는다"를 같은 무게로 시험한다.
"""

import numpy as np
import torch
from helpers import gt_model_output

from configs.exp_synthetic import get_config
from stella.builder import build_instance
from stella.data.encode import ChainEncoder
from stella.decode.cache import load_prediction, save_prediction
from stella.decode.postprocess import ChainMerger

IMAGE = 256
STRIDE = 4


def piece(label: int, points: list) -> dict:
    return {"class": label, "points": np.array(points, dtype=np.float32), "score": 1.0}


def test_collinear_fragments_merge_into_one():
    merger = ChainMerger(gap=8.0, align_cos=0.8)
    merged, removed = merger(
        [
            piece(3, [[10.0, 50.0], [40.0, 50.0]]),
            piece(3, [[45.0, 50.0], [80.0, 50.0]]),
        ]
    )
    assert removed == 1
    assert len(merged) == 1
    assert merged[0]["points"][:, 0].min() == 10.0
    assert merged[0]["points"][:, 0].max() == 80.0


def test_different_classes_do_not_merge():
    merger = ChainMerger(gap=8.0, align_cos=0.8)
    merged, removed = merger(
        [
            piece(3, [[10.0, 50.0], [40.0, 50.0]]),
            piece(5, [[45.0, 50.0], [80.0, 50.0]]),
        ]
    )
    assert removed == 0 and len(merged) == 2


def test_perpendicular_fragments_do_not_merge():
    """끝점은 가깝지만 서로를 향하지 않는다 — T접합에서 본선과 곁가지를 붙이면 안 된다."""
    merger = ChainMerger(gap=8.0, align_cos=0.8)
    merged, _ = merger(
        [
            piece(3, [[10.0, 50.0], [40.0, 50.0]]),
            piece(3, [[44.0, 50.0], [44.0, 90.0]]),
        ]
    )
    assert len(merged) == 2


def test_far_fragments_do_not_merge():
    merger = ChainMerger(gap=4.0, align_cos=0.8)
    merged, _ = merger(
        [
            piece(3, [[10.0, 50.0], [40.0, 50.0]]),
            piece(3, [[60.0, 50.0], [80.0, 50.0]]),
        ]
    )
    assert len(merged) == 2


def test_three_fragments_merge_in_order():
    merger = ChainMerger(gap=8.0, align_cos=0.8)
    merged, removed = merger(
        [
            piece(3, [[45.0, 50.0], [80.0, 50.0]]),
            piece(3, [[10.0, 50.0], [40.0, 50.0]]),
            piece(3, [[85.0, 50.0], [120.0, 50.0]]),
        ]
    )
    assert removed == 2 and len(merged) == 1
    xs = merged[0]["points"][:, 0]
    assert np.all(np.diff(xs) > 0)  # 이어 붙인 뒤에도 한 방향으로 진행한다


def test_disabled_merger_is_identity():
    merger = ChainMerger(gap=0.0, align_cos=0.8)
    items = [piece(3, [[10.0, 50.0], [40.0, 50.0]]), piece(3, [[45.0, 50.0], [80.0, 50.0]])]
    merged, removed = merger(items)
    assert removed == 0 and merged is items


def test_prediction_cache_roundtrip(tmp_path):
    """희소 캐시는 노드 셀의 값을 fp16 정밀도로 그대로 복원해야 한다."""
    cfg = get_config()
    cfg.data.image_size = IMAGE
    encoder = ChainEncoder(
        image_size=IMAGE, grid_stride=STRIDE, num_classes=12, max_degree=2, supersample=1
    )
    target = encoder.encode([piece(3, [[30.0, 100.0], [220.0, 100.0]])])
    targets = {k: torch.from_numpy(v).unsqueeze(0) for k, v in target.items()}
    output = gt_model_output(targets, cfg.data.num_classes, cfg.model.num_conn_slots)
    instances = [{"class": 3, "points": np.array([[30.0, 100.0], [220.0, 100.0]], np.float32)}]
    save_prediction(tmp_path / "a.npz", output[0], instances)
    shape = {"grid_size": IMAGE // STRIDE, "num_classes": 12, "num_slots": 2}
    loaded, back = load_prediction(tmp_path / "a.npz", shape)
    assert bool((loaded.node_mask == output.node_mask[0]).all())
    mask = output.node_mask[0]
    assert torch.allclose(loaded.self_coord[mask], output.self_coord[0][mask], atol=1e-3)
    assert torch.allclose(loaded.conn_dir[mask], output.conn_dir[0][mask], atol=1e-3)
    assert back[0]["class"] == 3


def test_cached_prediction_decodes_same_as_direct(tmp_path):
    """캐시를 거쳐도 디코딩 결과가 같아야 한다 — D 트랙 전체가 여기에 의존한다."""
    cfg = get_config()
    cfg.data.image_size = IMAGE
    encoder = ChainEncoder(
        image_size=IMAGE, grid_stride=STRIDE, num_classes=12, max_degree=2, supersample=1
    )
    target = encoder.encode([piece(3, [[30.0, 100.0], [220.0, 100.0]])])
    targets = {k: torch.from_numpy(v).unsqueeze(0) for k, v in target.items()}
    output = gt_model_output(targets, cfg.data.num_classes, cfg.model.num_conn_slots)
    save_prediction(tmp_path / "a.npz", output[0], [])
    shape = {"grid_size": IMAGE // STRIDE, "num_classes": 12, "num_slots": 2}
    loaded, _ = load_prediction(tmp_path / "a.npz", shape)
    decoder = build_instance(cfg.decode, cfg)
    direct = decoder(output[0])
    cached = decoder(loaded)
    assert len(direct) == len(cached) == 1
    assert np.allclose(direct[0]["points"], cached[0]["points"], atol=0.05)


def test_cache_round_trip_keeps_fg_logit(tmp_path):
    """캐시가 `fg_logit`을 잃으면 **D 트랙에서 전경 헤드가 통째로 사라진다.**

    실제로 그렇게 당했다: `fg_logit`을 `ModelOutput`에는 넣고 희소 캐시 키에는 안 넣어서,
    캐시를 거친 예측은 전부 0이 됐다. `sigmoid(0)=0.5`라 `fg_thresh=0.5` 경로에서
    **정점이 하나도 안 남아 f1이 0**이 됐다. 게이트는 기본값(`fg_thresh=-1`)으로만 돌아 못 잡는다.
    """
    cfg = get_config()
    cfg.data.image_size = IMAGE
    encoder = ChainEncoder(
        image_size=IMAGE, grid_stride=STRIDE, num_classes=12, max_degree=2, supersample=1
    )
    target = encoder.encode([piece(3, [[30.0, 100.0], [220.0, 100.0]])])
    targets = {k: torch.from_numpy(v).unsqueeze(0) for k, v in target.items()}
    output = gt_model_output(targets, cfg.data.num_classes, cfg.model.num_conn_slots)
    save_prediction(tmp_path / "a.npz", output[0], [])
    shape = {"grid_size": IMAGE // STRIDE, "num_classes": 12, "num_slots": 2}
    loaded, _ = load_prediction(tmp_path / "a.npz", shape)

    node = output[0].node_mask
    assert torch.count_nonzero(output[0].fg_logit[node]) > 0  # GT 주입은 전경을 확신한다
    assert torch.allclose(loaded.fg_logit[node], output[0].fg_logit[node], atol=0.05)

    cfg.decode.fg_thresh = 0.5  # 이진 헤드로 거르는 경로 — 캐시를 거쳐도 같아야 한다
    decoder = build_instance(cfg.decode, cfg)
    direct, cached = decoder(output[0]), decoder(loaded)
    assert len(direct) == len(cached) == 1
    assert np.allclose(direct[0]["points"], cached[0]["points"], atol=0.05)


def test_old_cache_without_fg_still_loads(tmp_path):
    """`fg` 키가 없는 **옛 캐시**도 읽혀야 한다 — 그 모델엔 전경 헤드가 없었다."""
    import numpy as _np

    cfg = get_config()
    cfg.data.image_size = IMAGE
    encoder = ChainEncoder(
        image_size=IMAGE, grid_stride=STRIDE, num_classes=12, max_degree=2, supersample=1
    )
    target = encoder.encode([piece(3, [[30.0, 100.0], [220.0, 100.0]])])
    targets = {k: torch.from_numpy(v).unsqueeze(0) for k, v in target.items()}
    output = gt_model_output(targets, cfg.data.num_classes, cfg.model.num_conn_slots)
    save_prediction(tmp_path / "a.npz", output[0], [])
    with _np.load(tmp_path / "a.npz") as data:  # fg 키만 뺀 옛 형식으로 다시 쓴다
        kept = {k: data[k] for k in data.files if k != "fg"}
    _np.savez_compressed(tmp_path / "old.npz", **kept)
    shape = {"grid_size": IMAGE // STRIDE, "num_classes": 12, "num_slots": 2}
    loaded, _ = load_prediction(tmp_path / "old.npz", shape)
    assert torch.count_nonzero(loaded.fg_logit) == 0  # 없으면 0으로 남는다
    assert len(build_instance(cfg.decode, cfg)(loaded)) == 1  # 기본 경로는 멀쩡하다


def test_dedup_removes_contained_line_and_never_adds():
    """포함된 짧은 선은 사라지고, 인스턴스 수는 **줄기만 한다** — 이 단계의 계약이다."""
    import numpy as np

    from stella.decode.dedup import DuplicateResolver

    resolver = DuplicateResolver(
        overlap_high=6.0,
        overlap_low=3.0,
        min_free_len=8.0,
        bridge_gap=0.0,
        min_diverge_len=0.0,
        join_gap=6.0,
        step=2.0,
        keep_ratio=0.35,
        mode="ratio",
    )
    long_line = np.stack([np.arange(0, 200, 4.0), np.zeros(50)], axis=1)
    inside = np.stack([np.arange(40, 120, 4.0), np.full(20, 1.5)], axis=1)  # 1.5px 옆 = 중복
    out, stats = resolver(
        [
            {"class": 3, "points": long_line, "score": 1.0},
            {"class": 3, "points": inside, "score": 1.0},
        ]
    )
    assert len(out) == 1
    assert stats["dedup_dropped"] == 1


def test_dedup_never_bridges_a_gap():
    """**떨어진 두 선은 잇지 않는다.** 겹치지 않으면 둘 다 그대로 남는다."""
    import numpy as np

    from stella.decode.dedup import DuplicateResolver

    resolver = DuplicateResolver(
        overlap_high=6.0,
        overlap_low=3.0,
        min_free_len=4.0,
        bridge_gap=0.0,
        min_diverge_len=0.0,
        join_gap=6.0,
        step=2.0,
        keep_ratio=0.35,
        mode="ratio",
    )
    first = np.stack([np.arange(0, 80, 4.0), np.zeros(20)], axis=1)
    far = np.stack([np.arange(200, 280, 4.0), np.zeros(20)], axis=1)  # 120px 떨어짐
    out, stats = resolver(
        [
            {"class": 3, "points": first, "score": 1.0},
            {"class": 3, "points": far, "score": 1.0},
        ]
    )
    assert len(out) == 2
    assert stats["dedup_joined"] == 0


def test_dedup_threshold_stays_below_true_neighbour_spacing():
    """중복 문턱은 **진짜 이웃 선 간격보다 좁아야** 한다 — 넓으면 진짜 선을 지운다.

    실측(08-20, GT 200장): 같은 클래스 이웃까지의 중앙 거리가 6px 이내인 진짜 선이 3.4%,
    4px 이내 1.6%, 3px 이내 0.8% 다. 6px 로 잡았더니 GT 주입 천장이 0.946 -> 0.908 로
    깎였다 — 이중선처럼 원래 붙어 있는 선을 지운 것이다. 제거 대상 중복은 간격 중앙 1.8px.
    """
    from configs.base import get_config

    decode = get_config().decode
    assert 0.0 < decode.dedup_high <= 4.0
    assert decode.dedup_low < decode.dedup_high
    # 붙이기 반경이 한 셀(4px) 남짓을 넘으면 **떨어진 선을 끌어오게** 된다 — 계약 위반이다.
    assert decode.dedup_join_gap <= 8.0


def test_dedup_keeps_partially_free_line_whole():
    """자유 구간이 충분하면 **자르지 않고 원본 그대로** 둔다 — 자르기는 진짜 선을 깎았다.

    실측(08-20): 자르는 판은 f1 +5.5% 지만 recall −3.9% · GT 주입 천장 0.946 -> 0.908 이다.
    정답에는 중복이 없으므로 그 4%는 진짜 선을 지운 것이다.
    """
    import numpy as np

    from stella.decode.dedup import DuplicateResolver

    resolver = DuplicateResolver(
        overlap_high=6.0,
        overlap_low=3.0,
        min_free_len=8.0,
        bridge_gap=0.0,
        min_diverge_len=0.0,
        join_gap=6.0,
        step=2.0,
        keep_ratio=0.35,
        mode="ratio",
    )
    base = np.stack([np.arange(0, 200, 4.0), np.zeros(50)], axis=1)
    half = np.stack([np.arange(100, 300, 4.0), np.full(50, 1.5)], axis=1)  # 절반만 겹친다
    out, _ = resolver(
        [
            {"class": 3, "points": base, "score": 1.0},
            {"class": 3, "points": half, "score": 1.0},
        ]
    )
    assert len(out) == 2
    assert any(len(item["points"]) == len(half) for item in out)  # 원본이 그대로 남았다


def _ends_resolver(**overrides):
    """머리·꼬리만 자르는 방식의 정리기. 문턱은 위 시험들과 같게 둔다."""
    from stella.decode.dedup import DuplicateResolver

    params = dict(
        overlap_high=6.0,
        overlap_low=3.0,
        min_free_len=8.0,
        bridge_gap=0.0,
        min_diverge_len=0.0,
        join_gap=6.0,
        step=2.0,
        keep_ratio=0.0,
        mode="ends",
    )
    return DuplicateResolver(**(params | overrides))


def test_dedup_ends_mode_never_fragments_a_crossing():
    """가운데서 겹치는 선은 **한 개 그대로** 남는다 — 자르면 진짜 선이 둘로 갈라진다.

    비율 게이트(keep_ratio) 없이도 성립해야 한다. 그것이 "머리·꼬리만 자른다"의 요점이다.
    """
    import numpy as np

    base = np.stack([np.arange(0, 200, 4.0), np.zeros(50)], axis=1)
    crossing = np.stack([np.full(50, 100.0), np.arange(-100, 100, 4.0)], axis=1)
    out, stats = _ends_resolver()(
        [
            {"class": 3, "points": base, "score": 1.0},
            {"class": 3, "points": crossing, "score": 1.0},
        ]
    )
    assert len(out) == 2  # 조각으로 갈라지지 않았다
    assert stats["dedup_dropped"] == 0


def test_dedup_ends_mode_trims_only_the_overlapping_tail():
    """꼬리만 겹친 선은 **짧아지되 하나로** 남는다 — 인스턴스 수는 그대로다."""
    import numpy as np

    base = np.stack([np.arange(0, 200, 4.0), np.zeros(50)], axis=1)
    tail = np.stack([np.arange(100, 300, 4.0), np.full(50, 1.5)], axis=1)  # 앞 절반이 겹친다
    out, _ = _ends_resolver()(
        [
            {"class": 3, "points": base, "score": 1.0},
            {"class": 3, "points": tail, "score": 1.0},
        ]
    )
    assert len(out) == 2
    trimmed = min(out, key=lambda item: _length(item["points"]))
    assert _length(trimmed["points"]) < _length(tail) - 50.0  # 겹친 앞부분이 잘렸다


def test_dedup_ends_mode_still_drops_a_contained_line():
    """완전히 포함된 선은 이 방식에서도 사라진다 — 자유 지점이 하나도 없다."""
    import numpy as np

    base = np.stack([np.arange(0, 200, 4.0), np.zeros(50)], axis=1)
    inside = np.stack([np.arange(40, 120, 4.0), np.full(20, 1.5)], axis=1)
    out, stats = _ends_resolver()(
        [
            {"class": 3, "points": base, "score": 1.0},
            {"class": 3, "points": inside, "score": 1.0},
        ]
    )
    assert len(out) == 1
    assert stats["dedup_dropped"] == 1


def _length(points) -> float:
    import numpy as np

    return float(np.linalg.norm(np.diff(np.asarray(points), axis=0), axis=1).sum())
