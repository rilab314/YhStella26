"""디코더 검증 — GT를 모델 출력 형식으로 주입하면 폴리라인이 복원되는지 (impl_plan M12)."""

import numpy as np
import torch
from helpers import gt_model_output

from configs.exp_synthetic import get_config
from stella.builder import build_instance
from stella.data.encode import ChainEncoder
from stella.data.synthetic import SyntheticDataset
from stella.data.types import collate_fn
from stella.eval import geometry

IMAGE = 256
STRIDE = 4


def make_cfg():
    cfg = get_config()
    cfg.data.image_size = IMAGE
    return cfg


def encode_scene(instances) -> dict:
    encoder = ChainEncoder(
        image_size=IMAGE, grid_stride=STRIDE, num_classes=12, max_degree=2, supersample=1
    )
    target = encoder.encode(instances)
    return {key: torch.from_numpy(value).unsqueeze(0) for key, value in target.items()}


def decode_gt(cfg, decoder, instances) -> list[dict]:
    targets = encode_scene(instances)
    output = gt_model_output(targets, cfg.data.num_classes, cfg.model.num_conn_slots)
    return decoder(output[0])


def covered_fraction(gt_points: np.ndarray, decoded: list[dict], rho: float = 3.0) -> float:
    sampled, tangent = geometry.resample(gt_points, 1.0)
    distance = np.full(sampled.shape[0], np.inf)
    for item in decoded:
        candidate = geometry.gated_distance(sampled, tangent, item["points"], np.cos(np.pi / 3))
        distance = np.minimum(distance, candidate)
    return float((distance <= rho).mean())


def test_empty_output_decodes_to_empty_list():
    """학습 초기 — 임계값을 넘는 정점이 하나도 없어도 죽지 않는다."""
    cfg = make_cfg()
    decoder = build_instance(cfg.decode, cfg)
    side, slots, classes = 64, cfg.model.num_conn_slots, cfg.data.num_classes
    from stella.model.stella import ModelOutput

    output = ModelOutput(
        heatmap_logit=torch.full((side, side), -10.0),
        node_mask=torch.zeros((side, side), dtype=torch.bool),
        class_logit=torch.zeros((side, side, classes)),
        self_coord=torch.zeros((side, side, 2)),
        end_logit=torch.zeros((side, side)),
        exist_logit=torch.zeros((side, side, slots)),
        conn_dir=torch.zeros((side, side, slots, 2)),
    )
    assert decoder(output) == []


def test_straight_line_is_recovered_with_end_extension():
    cfg = make_cfg()
    decoder = build_instance(cfg.decode, cfg)
    points = np.array([[30.0, 100.0], [220.0, 100.0]], dtype=np.float32)
    decoded = decode_gt(cfg, decoder, [{"class": 3, "points": points}])
    assert len(decoded) == 1
    assert decoded[0]["class"] == 3
    assert covered_fraction(points, decoded) > 0.98
    assert decoded[0]["points"][:, 1].std() < 0.5  # 수평선이므로 y가 거의 일정
    xs = decoded[0]["points"][:, 0]
    assert xs.min() < 34.0 and xs.max() > 216.0  # 끝 연장이 끝칸 미채움 길이를 복원한다


def test_two_classes_stay_separate():
    cfg = make_cfg()
    decoder = build_instance(cfg.decode, cfg)
    first = np.array([[20.0, 60.0], [230.0, 60.0]], dtype=np.float32)
    second = np.array([[20.0, 180.0], [230.0, 180.0]], dtype=np.float32)
    decoded = decode_gt(
        cfg, decoder, [{"class": 3, "points": first}, {"class": 5, "points": second}]
    )
    assert sorted(item["class"] for item in decoded) == [3, 5]


def test_t_junction_keeps_main_line_uncut():
    """본선은 접합 셀에서 잘리지 않고, 곁가지는 자기 끝 셀에서 정지한다 (10.3절)."""
    cfg = make_cfg()
    decoder = build_instance(cfg.decode, cfg)
    main = np.array([[10.0, 120.0], [246.0, 120.0]], dtype=np.float32)
    stem = np.array([[128.0, 30.0], [128.0, 120.0]], dtype=np.float32)
    decoded = decode_gt(cfg, decoder, [{"class": 3, "points": main}, {"class": 5, "points": stem}])
    per_class = sorted((item["class"], len(item["points"])) for item in decoded)
    assert [label for label, _ in per_class] == [3, 5]  # 클래스마다 사슬 하나
    assert covered_fraction(main, decoded) > 0.98


def test_x_crossing_gives_one_chain_per_line():
    """교차 셀을 잃은 선도 반경 2 건너뛰기로 한 사슬로 복원된다 (6.4절·10.3절)."""
    cfg = make_cfg()
    decoder = build_instance(cfg.decode, cfg)
    down = np.array([[20.0, 20.0], [236.0, 236.0]], dtype=np.float32)
    up = np.array([[20.0, 236.0], [236.0, 20.0]], dtype=np.float32)
    decoded = decode_gt(cfg, decoder, [{"class": 3, "points": down}, {"class": 5, "points": up}])
    assert sorted(item["class"] for item in decoded) == [3, 5]
    assert covered_fraction(down, decoded) > 0.95
    assert covered_fraction(up, decoded) > 0.95


def test_short_line_becomes_three_point_polyline():
    """3칸짜리 선 = 1셀 사슬 + 양방향 끝 연장 -> 3점 폴리라인 (결정 31)."""
    cfg = make_cfg()
    decoder = build_instance(cfg.decode, cfg)
    points = np.array([[100.0, 50.0], [111.0, 50.0]], dtype=np.float32)
    decoded = decode_gt(cfg, decoder, [{"class": 3, "points": points}])
    assert len(decoded) == 1
    assert len(decoded[0]["points"]) == 3


def test_synthetic_scene_is_mostly_recovered():
    cfg = make_cfg()
    decoder = build_instance(cfg.decode, cfg)
    dataset = SyntheticDataset(
        split="val",
        image_size=IMAGE,
        grid_stride=STRIDE,
        num_classes=12,
        max_degree=2,
        encode_supersample=1,
        augment=False,
        limit=4,
    )
    fractions, gt_count, decoded_count = [], 0, 0
    for index in range(4):
        batch = collate_fn([dataset[index]])
        decoded = decoder(gt_model_output(batch, cfg.data.num_classes, cfg.model.num_conn_slots)[0])
        assert decoded, "디코딩 결과가 비었다"
        gt_count += len(batch["instances"][0])
        decoded_count += len(decoded)
        for inst in batch["instances"][0]:
            fractions.append(covered_fraction(np.asarray(inst["points"]), decoded))
    assert float(np.mean(fractions)) > 0.9, f"평균 복원율 {np.mean(fractions):.3f}"
    assert decoded_count < gt_count * 1.2, f"조각 과다: {decoded_count} / GT {gt_count}"


def test_low_purity_chain_is_discarded():
    """argmax 클래스가 반반 섞인 사슬은 순도 검사에서 통째로 버려진다 (10.3절)."""
    cfg = make_cfg()
    points = np.array([[30.0, 100.0], [220.0, 100.0]], dtype=np.float32)
    targets = encode_scene([{"class": 3, "points": points}])
    output = gt_model_output(targets, cfg.data.num_classes, cfg.model.num_conn_slots)
    positive = (targets["class_map"] > 0).nonzero(as_tuple=False)
    mixed = torch.full((len(positive), cfg.data.num_classes), -5.0)
    mixed[::2, 3], mixed[::2, 5] = 2.0, 1.8  # 절반은 3이 근소하게 이긴다
    mixed[1::2, 3], mixed[1::2, 5] = 1.8, 2.0  # 나머지 절반은 5가 이긴다
    output.class_logit[targets["class_map"] > 0] = mixed
    strict = build_instance(cfg.decode, cfg)
    assert strict(output[0]) == []  # 순도 ~0.5 <= 0.6 -> 전 시드 실패
    cfg.decode.purity_thresh = 0.3
    lenient = build_instance(cfg.decode, cfg)
    assert len(lenient(output[0])) == 1  # 순도 하한을 낮추면 한 사슬로 살아난다


def test_simplify_removes_collinear_points():
    cfg = make_cfg()
    cfg.decode.simplify_tol = 1.0
    decoder = build_instance(cfg.decode, cfg)
    points = np.array([[30.0, 100.0], [220.0, 100.0]], dtype=np.float32)
    decoded = decode_gt(cfg, decoder, [{"class": 3, "points": points}])
    assert len(decoded) == 1
    assert len(decoded[0]["points"]) == 2  # 직선이므로 양 끝만 남는다
    assert covered_fraction(points, decoded) > 0.98
