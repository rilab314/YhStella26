"""디코더 검증 — GT를 모델 출력 형식으로 주입하면 폴리라인이 복원되는지 (impl_plan M6)."""

import numpy as np
import torch

from configs.exp_synthetic import get_config
from stella.builder import build_instance
from stella.data.encode import GridEncoder
from stella.data.synthetic import SyntheticDataset
from stella.data.types import collate_fn
from stella.eval import geometry
from stella.loss.conn import derive_branches
from stella.model.stella import ModelOutput

IMAGE = 256
STRIDE = 4


def make_cfg():
    cfg = get_config()
    cfg.data.image_size = IMAGE
    return cfg


def gt_output(targets: dict, cfg) -> ModelOutput:
    """GT를 7.1절 출력 계약 형태로 주입한다 (완벽한 예측)."""
    shape = targets["class_map"].shape
    slots, classes = cfg.model.num_conn_slots, cfg.data.num_classes
    positive = targets["class_map"] > 0
    output = ModelOutput(
        heatmap_logit=torch.where(positive, 10.0, -10.0),
        node_mask=positive.clone(),
        class_logit=torch.zeros((*shape, classes)),
        self_coord=torch.zeros((*shape, 2)),
        exist_logit=torch.full((*shape, slots), -10.0),
        conn_dir=torch.zeros((*shape, slots, 2)),
        t_logit=torch.full((*shape, slots), -10.0),
    )
    cells = positive.nonzero(as_tuple=False)
    gt_dir, gt_t, valid = derive_branches(targets, cells)
    output.self_coord[positive] = targets["coord_map"][positive]
    output.class_logit[positive] = (
        torch.nn.functional.one_hot(targets["class_map"][positive], classes).float() * 20.0
    )
    output.conn_dir[positive] = gt_dir[:, :slots]
    output.exist_logit[positive] = torch.where(valid[:, :slots], 10.0, -10.0)
    output.t_logit[positive] = torch.where(gt_t[:, :slots] > 0, 10.0, -10.0)
    return output


def encode_scene(instances):
    encoder = GridEncoder(
        image_size=IMAGE, grid_stride=STRIDE, num_classes=12, max_degree=3, supersample=1
    )
    target = encoder.encode(instances)
    return {key: torch.from_numpy(value).unsqueeze(0) for key, value in target.items()}


def covered_fraction(gt_points: np.ndarray, decoded: list[dict], rho: float = 3.0) -> float:
    sampled, tangent = geometry.resample(gt_points, 1.0)
    distance = np.full(sampled.shape[0], np.inf)
    for item in decoded:
        candidate = geometry.gated_distance(sampled, tangent, item["points"], np.cos(np.pi / 3))
        distance = np.minimum(distance, candidate)
    return float((distance <= rho).mean())


def test_straight_line_is_recovered():
    cfg = make_cfg()
    decoder = build_instance(cfg.decode, cfg)
    points = np.array([[30.0, 100.0], [220.0, 100.0]], dtype=np.float32)
    targets = encode_scene([{"class": 3, "points": points}])
    decoded = decoder(gt_output(targets, cfg)[0])
    assert len(decoded) == 1
    assert decoded[0]["class"] == 3
    assert covered_fraction(points, decoded) > 0.98
    assert decoded[0]["points"][:, 1].std() < 0.5  # 수평선이므로 y가 거의 일정


def test_two_classes_stay_separate():
    cfg = make_cfg()
    decoder = build_instance(cfg.decode, cfg)
    first = np.array([[20.0, 60.0], [230.0, 60.0]], dtype=np.float32)
    second = np.array([[20.0, 180.0], [230.0, 180.0]], dtype=np.float32)
    targets = encode_scene([{"class": 3, "points": first}, {"class": 5, "points": second}])
    decoded = decoder(gt_output(targets, cfg)[0])
    assert sorted(item["class"] for item in decoded) == [3, 5]


def test_synthetic_scene_is_mostly_recovered():
    cfg = make_cfg()
    decoder = build_instance(cfg.decode, cfg)
    dataset = SyntheticDataset(
        split="val",
        image_size=IMAGE,
        grid_stride=STRIDE,
        num_classes=12,
        max_degree=3,
        encode_supersample=1,
        augment=False,
        limit=4,
    )
    fractions = []
    for index in range(4):
        batch = collate_fn([dataset[index]])
        decoded = decoder(gt_output(batch, cfg)[0])
        assert decoded, "디코딩 결과가 비었다"
        for inst in batch["instances"][0]:
            fractions.append(covered_fraction(np.asarray(inst["points"]), decoded))
    assert float(np.mean(fractions)) > 0.9, f"평균 복원율 {np.mean(fractions):.3f}"


def test_simplify_removes_collinear_points():
    """simplify_tol > 0 이면 RDP로 직선 구간의 중간점을 지운다 (10.5절 5번)."""
    cfg = make_cfg()
    cfg.decode.simplify_tol = 1.0
    decoder = build_instance(cfg.decode, cfg)
    points = np.array([[30.0, 100.0], [220.0, 100.0]], dtype=np.float32)
    targets = encode_scene([{"class": 3, "points": points}])
    decoded = decoder(gt_output(targets, cfg)[0])
    assert len(decoded) == 1
    assert len(decoded[0]["points"]) == 2  # 직선이므로 양 끝만 남는다
    assert covered_fraction(points, decoded) > 0.98
