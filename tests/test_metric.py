"""인스턴스 CCQ 지표 검증 (design M7·11절)."""

import numpy as np

from configs.exp_synthetic import get_config
from stella.builder import build_instance
from stella.eval import runlog


def make_metric(**overrides):
    cfg = get_config()
    for key, value in overrides.items():
        setattr(cfg.eval, key, value)
    return build_instance(cfg.eval, cfg)


def line(x0: float, x1: float, y: float, label: int = 3) -> dict:
    return {"class": label, "points": np.array([[x0, y], [x1, y]], dtype=np.float32)}


def test_exact_reproduction_scores_one():
    metric = make_metric()
    truth = [line(10, 500, 100), line(10, 500, 300, label=5)]
    metric.update([dict(item) for item in truth], truth)
    result = metric.compute()
    assert float(result["f1"]) == 1.0
    assert float(result["precision"]) == 1.0 and float(result["recall"]) == 1.0
    assert float(result["coverage"]) == 1.0 and float(result["correctness"]) == 1.0
    assert float(result["frag"]) == 1.0
    assert float(result["rms"]) < 1e-6


def test_missing_prediction_is_false_negative():
    metric = make_metric()
    metric.update([], [line(10, 500, 100)])
    result = metric.compute()
    assert float(result["recall"]) == 0.0
    assert float(result["coverage"]) == 0.0


def test_fragmented_prediction_gives_tp_and_redundant_fp():
    """조각난 예측: 대표 조각 하나가 TP, 나머지는 GT 위에 있으므로 redundant FP."""
    metric = make_metric()
    truth = [line(0, 400, 100)]
    pieces = [line(0, 260, 100), line(270, 400, 100)]
    metric.update(pieces, truth)
    result = metric.compute()
    assert float(result["f1"]) > 0.0
    assert float(result["fp_redundant"]) == 1.0
    assert float(result["fp_spurious"]) == 0.0
    assert float(result["frag"]) == 2.0  # GT 하나를 두 조각이 덮는다
    assert float(result["coverage"]) > 0.95  # 커버리지는 조각남을 벌하지 않는다


def test_line_drawn_off_the_gt_is_spurious_fp():
    metric = make_metric()
    metric.update([line(10, 500, 400)], [line(10, 500, 100)])
    result = metric.compute()
    assert float(result["fp_spurious"]) == 1.0
    assert float(result["fp_redundant"]) == 0.0
    assert float(result["precision"]) == 0.0


def test_hallucinated_bridge_lowers_correctness():
    """실제로 끊긴 곳을 이어버린 예측은 그 구간이 어떤 버퍼에도 안 들어가 C2가 떨어진다."""
    metric = make_metric()
    truth = [line(0, 150, 100), line(450, 600, 100)]
    metric.update([line(0, 600, 100)], truth)
    result = metric.compute()
    assert float(result["correctness"]) < 0.6
    assert float(result["fp_spurious"]) == 1.0


def test_short_offset_is_still_matched_within_buffer():
    metric = make_metric(buffer_rho=12.0)
    metric.update([line(10, 500, 106)], [line(10, 500, 100)])
    result = metric.compute()
    assert float(result["f1"]) == 1.0
    assert 5.5 < float(result["rms"]) < 6.5


def test_lane_switch_is_not_a_true_positive():
    """옆 차선으로 갈아탄 예측은 매칭되면 안 된다.

    C2를 '모든 GT의 합집합'까지의 거리로 재던 시절에는 이것이 TP였다 — 갈아타도 매 순간
    **어떤** 선 위엔 있기 때문이다. 실측(08-14): 차선 간격 중앙값이 11.8 px 인데 버퍼가
    12 px 여서 이웃 선이 늘 버퍼 안에 있었고, 매칭 자격을 얻은 예측의 17%가 이 모양이었다.
    지금은 **GT 선 하나** 위에 머무는 비율로 잰다.
    """
    metric = make_metric()
    truth = [line(0, 400, 100), line(0, 400, 112)]  # 12 px 간격 = 실측 중앙값
    switched = {
        "class": 3,
        "points": np.array([[0, 100], [200, 100], [210, 112], [400, 112]], dtype=np.float32),
    }
    metric.update([switched], truth)
    result = metric.compute()
    assert float(result["f1"]) == 0.0  # 어느 쪽 선도 주장하지 못한다
    assert float(result["correctness"]) < 0.6  # 절반만 한 선 위에 있다


def test_single_lane_prediction_survives_the_narrow_buffer():
    """갈아탐을 잡겠다고 좁힌 버퍼가 정상 예측까지 죽이면 안 된다 (위 검사의 짝)."""
    metric = make_metric()
    truth = [line(0, 400, 100), line(0, 400, 112)]
    metric.update([line(0, 400, 101), line(0, 400, 111)], truth)
    result = metric.compute()
    assert float(result["f1"]) == 1.0


def test_class_mismatch_is_not_matched():
    metric = make_metric()
    metric.update([line(10, 500, 100, label=7)], [line(10, 500, 100, label=3)])
    result = metric.compute()
    assert float(result["f1"]) == 0.0
    assert float(result["fp_redundant"]) == 1.0  # 위치는 맞으니 spurious 는 아니다


def test_per_class_keys_only_for_present_classes():
    metric = make_metric()
    truth = [line(10, 500, 100, label=3), line(10, 500, 300, label=9)]
    metric.update([dict(item) for item in truth], truth)
    result = metric.compute()
    assert "f1/lane_line" in result and "f1/stop_line" in result
    assert "f1/bicycle_lane" not in result
    assert float(result["f1_macro"]) == 1.0


def test_best_checkpoint_ignores_last_ckpt(tmp_path):
    """`last.ckpt` 는 최종 에폭이 아니다 — 점수가 가장 좋은 에폭 파일을 골라야 한다.

    Lightning 2.6.5 는 상위 k 저장이 일어난 에폭에만 `last.ckpt` 를 갱신한다. F02(40에폭)에서
    그 파일이 29에폭에 멈춰 있었다. 이름을 믿고 집으면 조용히 다른 모델을 채점하게 된다.
    """
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    rows = ["epoch,val/inst/f1"] + [f"{e},{0.30 + 0.01 * e}" for e in range(4)]
    (run / "metrics.csv").write_text("\n".join(rows), encoding="utf-8")
    for name in ("epoch001.ckpt", "epoch002.ckpt", "last.ckpt"):
        (run / "checkpoints" / name).write_bytes(b"")
    assert runlog.best_checkpoint(run).name == "epoch002.ckpt"


def test_best_checkpoint_without_scores_returns_none(tmp_path):
    """점수를 못 읽으면 None — 부르는 쪽이 예전 방식으로 물러설 수 있어야 한다."""
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "epoch000.ckpt").write_bytes(b"")
    assert runlog.best_checkpoint(run) is None
