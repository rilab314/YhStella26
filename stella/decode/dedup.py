"""중복 정리 — 같은 차선 위에 겹쳐 그려진 선을 지우거나 끝점에서 잇는다 (design 10.5절).

**실측이 이 단계를 요구했다.** 예측 선의 18.5%가 같은 클래스의 다른 선과 **1.8 px 간격으로
나란히** 그려진다(겹친 쌍의 91%가 한 셀=4 px 이내). 이웃 차선(간격 11.8 px)이 아니라 **같은
차선 위의 이중 그리기**다. 원인은 모델이 정답보다 넓은 띠를 전경으로 부르고(선택 셀이 정답
셀의 4배) 그 띠에서 두 줄의 정점이 나오는 것이다. 디코더는 정점을 한 번씩만 쓰지만
**"이미 그린 선"이라는 개념이 없어** 나란한 중복을 막지 못한다.

**이 단계가 할 수 있는 것과 없는 것 (계약)**

    한다:   포함된 짧은 선을 지운다 · 겹친 구간은 한쪽만 남기고 두 선을 끝점에서 잇는다
            · Y 갈래는 최대 곡률이 낮은 쪽을 남긴다
    안 한다: **떨어진 두 선을 잇지 않는다**(간격을 메우지 않는다) · 없던 인스턴스를 만들지
            않는다 · 클래스를 바꾸지 않는다

그래서 인스턴스 수는 **줄기만 하고 늘지 않는다.** 이는 주장이 아니라 알고리즘의 성질이다 —
연결은 **이미 겹쳐 있던 두 선 사이에서만** 일어난다.

거리 판정(이력 문턱·짧은 끊김 잇기)은 직전 연구 LaneStitch 의 벡터화 후처리에서 가져왔고,
**간격을 메우는 단계를 뺐다.**
"""

import numpy as np

from stella.eval import geometry

EPS = 1e-9
# 이어 붙인 결과의 최대 꺾임각이 원래 조각들보다 이만큼 넘게 나빠지면 갈래를 잘못 고른 것이다.
BRANCH_SLACK_DEG = 20.0


class DuplicateResolver:
    """같은 클래스 선들의 중복을 정리한다. `overlap_high <= 0` 이면 무동작."""

    @classmethod
    def from_cfg(cls, module_cfg, cfg, **kwargs) -> "DuplicateResolver":
        return cls(
            overlap_high=module_cfg.overlap_high,
            overlap_low=module_cfg.overlap_low,
            min_free_len=module_cfg.min_free_len,
            bridge_gap=module_cfg.bridge_gap,
            min_diverge_len=module_cfg.min_diverge_len,
            join_gap=module_cfg.join_gap,
            step=module_cfg.step,
            keep_ratio=module_cfg.keep_ratio,
            **kwargs,
        )

    def __init__(
        self,
        *,
        overlap_high: float,
        overlap_low: float,
        min_free_len: float,
        bridge_gap: float,
        min_diverge_len: float,
        join_gap: float,
        step: float,
        keep_ratio: float,
    ):
        self.overlap_high = overlap_high
        self.overlap_low = overlap_low
        self.min_free_len = min_free_len
        self.bridge_gap = bridge_gap
        self.min_diverge_len = min_diverge_len
        self.join_gap = join_gap
        self.step = step
        self.keep_ratio = keep_ratio

    def __call__(self, instances: list[dict]) -> tuple[list[dict], dict]:
        """정리된 인스턴스와 진단 카운터를 낸다. 클래스별로 따로 처리한다."""
        if self.overlap_high <= 0.0 or len(instances) < 2:
            return instances, {"dedup_dropped": 0, "dedup_joined": 0}
        out, stats = [], {"dedup_dropped": 0, "dedup_joined": 0}
        for label in sorted({int(item["class"]) for item in instances}):
            group = [item for item in instances if int(item["class"]) == label]
            out.extend(self._resolve_group(group, stats))
        return out, stats

    def _resolve_group(self, group: list[dict], stats: dict) -> list[dict]:
        """길이 내림차순으로 훑으며 이미 확정된 선과 겹치는 구간을 뺀다."""
        order = sorted(group, key=lambda it: _arc_length(it["points"]), reverse=True)
        kept: list[dict] = []
        for item in order:
            pieces = self._free_pieces(item["points"], [k["points"] for k in kept])
            if not pieces:
                stats["dedup_dropped"] += 1
                continue
            if not kept:
                kept.extend(dict(item, points=piece) for piece in pieces)
                continue
            if self._survives_whole(item, pieces):
                kept.append(item)  # 상당 부분이 자유롭다 -> **자르지 않고 원본 그대로** 남긴다
                continue
            self._absorb(kept, item, pieces, stats)
        return kept

    def _survives_whole(self, item: dict, pieces: list[np.ndarray]) -> bool:
        """자유 구간이 원래 길이의 `keep_ratio` 이상이면 **중복이 아니다** — 원본을 그대로 둔다.

        자르기는 위험하다. 실측(08-20): 자르는 판을 쓰면 f1 이 +5.5% 오르지만 recall 이 −3.9%,
        GT 주입 천장이 0.946 -> 0.908 로 내려갔다. 정답에는 중복이 없으므로 그 4%는 **진짜 선을
        지운 것**이다. 미리 정한 판정 기준("재현율을 떨어뜨리면 안 된다")이 그 판을 기각했다.
        """
        if self.keep_ratio <= 0.0:
            return False
        total = _arc_length(item["points"])
        free = sum(_arc_length(piece) for piece in pieces)
        return total > EPS and free / total >= self.keep_ratio

    def _absorb(self, kept: list[dict], item: dict, pieces: list[np.ndarray], stats: dict) -> None:
        """자유 구간을 **겹쳤던 선의 끝에** 붙인다. 붙일 곳이 없으면 새 선으로 남긴다.

        **조각을 버리지 않는다.** 한때 가장 긴 조각만 남겼더니 recall 이 3.6% 떨어지고 GT 주입
        천장이 0.946 -> 0.925 로 내려갔다 — 겹침에 잘린 나머지 조각이 진짜 선이었기 때문이다.
        """
        for piece in pieces:
            target, end = _nearest_end(kept, piece, self.join_gap)
            joined = None if target is None else _join(kept[target]["points"], piece, end)
            if joined is None:
                kept.append(dict(item, points=piece))
                continue
            kept[target] = dict(kept[target], points=joined)
            stats["dedup_joined"] += 1

    def _free_pieces(self, points, refs: list[np.ndarray]) -> list[np.ndarray]:
        """기준선들에서 가로 거리가 먼 구간만 남긴다. 남는 것이 없으면 '포함'이다."""
        pts = _resample(np.asarray(points, dtype=np.float64), self.step)
        if pts.shape[0] < 2:
            return []
        if not refs:
            return [pts]
        distance = np.full(pts.shape[0], np.inf)
        for ref in refs:
            distance = np.minimum(distance, _point_to_polyline(pts, ref))
        free = self._hysteresis(distance, pts)
        free = _bridge(pts, free, self.bridge_gap)
        pieces = [pts[s:e] for s, e in _runs(free) if e - s >= 2]
        return [p for p in pieces if _arc_length(p) >= self.min_free_len]

    def _hysteresis(self, distance: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """두 문턱 — 약한 이탈은 **강한 이탈이 일정 길이 이상 지속될 때만** 자유로 인정한다.

        점 하나가 튀어서 구간 전체를 살리면 중복이 조각으로 남는다.
        """
        strong, weak = distance > self.overlap_high, distance > self.overlap_low
        free = np.zeros(distance.shape[0], dtype=bool)
        for start, end in _runs(weak):
            if _sustained(strong[start:end], pts[start:end], self.min_diverge_len):
                free[start:end] = True
        return free


def _sustained(strong: np.ndarray, pts: np.ndarray, min_len: float) -> bool:
    if min_len <= 0.0:
        return bool(strong.any())
    return any(_arc_length(pts[s:e]) >= min_len for s, e in _runs(strong))


def _bridge(pts: np.ndarray, free: np.ndarray, gap: float) -> np.ndarray:
    """자유 구간 사이의 **짧은** 겹침 끊김을 메운다 (양 끝은 건드리지 않는다)."""
    if gap <= 0.0:
        return free
    filled = free.copy()
    for start, end in _runs(~free):
        if 0 < start and end < free.shape[0] and _arc_length(pts[start - 1 : end + 1]) <= gap:
            filled[start:end] = True
    return filled


def _nearest_end(kept: list[dict], piece: np.ndarray, join_gap: float):
    """자유 구간을 붙일 대상과 그 끝(0=앞, -1=뒤). `join_gap` 안에 없으면 (None, None).

    **간격을 메우지 않는다는 계약이 여기서 지켜진다** — 이미 겹쳐 있던 선의 끝에서
    `join_gap`(한 셀 남짓) 안에 있을 때만 붙는다. 떨어진 선을 끌어오지 않는다.
    """
    best, target, end = join_gap, None, None
    for index, item in enumerate(kept):
        for tip, side in ((item["points"][0], 0), (item["points"][-1], -1)):
            for near in (piece[0], piece[-1]):
                distance = float(np.linalg.norm(tip - near))
                if distance < best:
                    best, target, end = distance, index, side
    return target, end


def _join(base: np.ndarray, piece: np.ndarray, end: int) -> np.ndarray | None:
    """끝점에서 잇는다. **Y 갈래면 최대 곡률이 낮은 쪽을 남긴다** — 못 이으면 None.

    이어 붙인 결과의 최대 꺾임각이 원래 두 조각의 최대값보다 크게 나빠지면, 그 이음은
    갈래를 잘못 고른 것이다. 차선표시는 급격히 꺾이지 않는다(라벨 실측 꺾임각 90% 분위수 5도).
    """
    tip = base[0] if end == 0 else base[-1]
    forward = (
        piece
        if float(np.linalg.norm(tip - piece[0])) <= float(np.linalg.norm(tip - piece[-1]))
        else piece[::-1]
    )
    joined = np.vstack([forward[::-1], base]) if end == 0 else np.vstack([base, forward])
    if _max_turn(joined) > max(_max_turn(base), _max_turn(forward)) + BRANCH_SLACK_DEG:
        return None
    return joined


def _max_turn(points: np.ndarray) -> float:
    """연속한 두 스텝 사이의 방향 변화 최대값(도). 점이 3개 미만이면 0."""
    delta = np.diff(points, axis=0)
    norm = np.linalg.norm(delta, axis=1, keepdims=True)
    unit = np.divide(delta, norm, out=np.zeros_like(delta), where=norm > EPS)
    if unit.shape[0] < 2:
        return 0.0
    cosine = np.clip((unit[:-1] * unit[1:]).sum(axis=1), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)).max())


def _point_to_polyline(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """각 점에서 폴리라인까지의 최소 거리 (M,)."""
    poly = np.asarray(poly, dtype=np.float64)
    if poly.shape[0] < 2:
        return np.full(pts.shape[0], np.inf)
    start, span = poly[:-1], poly[1:] - poly[:-1]
    length2 = np.maximum((span**2).sum(axis=1), EPS)
    rel = pts[:, None, :] - start[None, :, :]
    t = np.clip((rel * span[None, :, :]).sum(axis=2) / length2[None, :], 0.0, 1.0)
    projected = start[None, :, :] + t[:, :, None] * span[None, :, :]
    return np.linalg.norm(pts[:, None, :] - projected, axis=2).min(axis=1)


def _runs(mask: np.ndarray):
    """참인 구간의 (시작, 끝) 목록."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2], edges[1::2]))


def _resample(points: np.ndarray, step: float) -> np.ndarray:
    if points.shape[0] < 2 or step <= 0.0:
        return points
    return geometry.resample(points, step)[0]


def _arc_length(points) -> float:
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _longest(pieces: list[np.ndarray]) -> np.ndarray:
    return max(pieces, key=_arc_length)
