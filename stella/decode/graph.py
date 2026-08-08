"""객체 생성(디코딩) — 셀 단위 예측을 폴리라인 객체로 (impl_plan 10절, 9·10차 개정).

    ① 정점 추출 -> ② 사슬 확장 (시드에서 양방향) -> ③ 후처리(병합·단순화)

핵심 확인은 **"서로가 서로의 점을 향하는가"**다: 내 확장 슬롯 방향 c와 후보의 되가리킴
슬롯 방향 n이 마주보면 c . n -> -1. 인코딩(6.4절 사슬)과 같은 모양으로 한 노드씩
확장하며, 구 GraphDecoder의 전역 그래프·상호 최선 확인·경로 절단은 폐기했다 —
간선 소실 2.3%가 성분 수 1.8배로 증폭되던 구조였다.

① 은 `vertices.py`, ③ 은 `postprocess.py`, 진단 카운터는 `stats.py`에 있다.
좌표 규약: 내부 계산은 전부 **격자 단위**다. 반환 직전에만 픽셀로 바꾼다 —
격자 좌표 p는 픽셀 p * s - 0.5 에 대응한다(인코더가 픽셀 면적 중심 +0.5를 쓰기 때문).
"""

from dataclasses import fields as dataclass_fields

import numpy as np

from stella.decode.postprocess import ChainMerger, simplify_polyline
from stella.decode.stats import ChainStats
from stella.decode.vertices import VertexExtractor

PIXEL_CENTER_SHIFT = 0.5
# 동률 해소용 미세 거리 항. 일직선 위에서는 한 칸 뒤와 두 칸 뒤가 정렬·마주봄 모두
# 동률이라(둘 다 같은 선의 셀이라 되가리킴 슬롯도 있다) 가까운 쪽을 골라야 정점을
# 건너뛰지 않는다. 계획이 배제한 것은 "정렬 나쁜 가까운 셀을 끌어들이는" 크기의
# 거리 항(w_dist = 0.3)이고, 이 값은 반경 2에서 최대 0.004라 동률만 가른다 (10.3절).
DISTANCE_TIEBREAK = 1e-3
ALIGN_MODES = ("cosine", "perp")


class ChainDecoder:
    @classmethod
    def from_cfg(cls, module_cfg, cfg, **kwargs) -> "ChainDecoder":
        params = {
            f.name: getattr(module_cfg, f.name)
            for f in dataclass_fields(module_cfg)
            if f.name not in ("path", "name")
        }
        return cls(grid_stride=cfg.data.grid_stride, grid_size=cfg.data.grid_size, **params)

    def __init__(
        self,
        *,
        grid_stride: int,
        grid_size: int,
        heatmap_thresh: float,
        exist_thresh: float,
        end_thresh: float,
        radius: int,
        align_thresh: float,
        opp_thresh: float,
        w_opp: float,
        min_class_prob: float,
        purity_thresh: float,
        end_extend: float,
        min_points: int,
        simplify_tol: float,
        seed_mode: str,
        stop_needs_nocand: bool,
        merge_gap: float,
        merge_align: float,
        align_mode: str,
        perp_thresh: float,
    ):
        if align_mode not in ALIGN_MODES:
            raise ValueError(f"align_mode 는 {ALIGN_MODES} 중 하나여야 한다: {align_mode}")
        self.grid_stride = grid_stride
        self.exist_thresh = exist_thresh
        self.end_thresh = end_thresh
        self.align_thresh = align_thresh
        self.opp_thresh = opp_thresh
        self.w_opp = w_opp
        self.min_class_prob = min_class_prob
        self.purity_thresh = purity_thresh
        self.end_extend = end_extend
        self.min_points = min_points
        self.simplify_tol = simplify_tol
        self.stop_needs_nocand = stop_needs_nocand
        self.align_mode = align_mode
        self.perp_thresh = perp_thresh
        self.extractor = VertexExtractor(
            grid_size=grid_size,
            heatmap_thresh=heatmap_thresh,
            radius=radius,
            seed_mode=seed_mode,
            end_thresh=end_thresh,
        )
        self.merger = ChainMerger(gap=merge_gap, align_cos=merge_align)
        self.stats = ChainStats()

    def __call__(self, output) -> list[dict]:
        vertices = self.extractor(output)
        if vertices["point"].shape[0] == 0:
            self.stats.add_vertices(0, 0)
            return []
        chains = self._grow_chains(vertices)
        instances = [self._to_instance(vertices, *chain) for chain in chains]
        merged, removed = self.merger([item for item in instances if item is not None])
        self.stats.add_merge(removed)
        return merged

    # --- ② 사슬 확장 (10.3절) ----------------------------------------------------

    def _grow_chains(self, vertices: dict) -> list[tuple]:
        """시드마다 양방향으로 확장하고, 순도 검사를 통과한 사슬만 남긴다."""
        total, slots = vertices["exist"].shape
        used = np.zeros(total, dtype=bool)
        slot_used = np.zeros((total, slots), dtype=bool)
        failed = np.zeros(total, dtype=bool)
        chains = []
        for seed in self.extractor.seed_order(vertices):
            if used[seed] or failed[seed]:
                continue
            chain = self._grow_from_seed(vertices, used, slot_used, failed, int(seed))
            if chain is not None:
                chains.append(chain)
                self.stats.add_chain(len(chain[0]))
        self.stats.add_vertices(total, int(used.sum()))
        return chains

    def _grow_from_seed(self, vertices, used, slot_used, failed, seed) -> tuple | None:
        """시드의 슬롯 두 개를 따라 양방향 확장 -> 순도 검사 -> 끝 연장.

        R > 2 ablation에서는 존재 확률 상위 2개 슬롯만 시드 방향으로 쓴다 —
        사슬은 정의상 양방향뿐이다.
        """
        label = int(vertices["label"][seed])
        used[seed] = True
        touched = [seed]
        forward, backward = np.argsort(-vertices["exist"][seed], kind="stable")[:2]
        head = self._expand(vertices, used, slot_used, seed, int(forward), label, touched)
        tail = self._expand(vertices, used, slot_used, seed, int(backward), label, touched)
        chain = [*reversed(tail), seed, *head]
        purity = float(np.mean(vertices["label"][chain] == label))
        if purity <= self.purity_thresh:  # 정점·슬롯을 되돌리고 시드만 실패로 남긴다
            used[touched] = False
            slot_used[touched] = False
            failed[seed] = True
            self.stats.add_reject()
            return None
        head_ext, tail_ext = self._extensions(vertices, slot_used, chain)
        return chain, head_ext, tail_ext, label

    def _expand(self, vertices, used, slot_used, start, slot, label, touched) -> list[int]:
        """한 노드씩 단방향 확장. 정지 사유는 전부 stats에 기록한다.

        고리 폐쇄(시작 정점 복귀)는 시작 정점이 이미 사용 상태라 후보에서 빠져
        "후보 없음" 정지에 흡수된다. 스텝마다 미사용 정점을 하나 소비하므로 무한 루프가 없다.
        """
        path: list[int] = []
        vertex, k = start, slot
        while True:
            if k is None or slot_used[vertex, k]:
                return self._stop(path, "slotused")
            if vertices["exist"][vertex, k] <= self.exist_thresh:
                return self._stop(path, "exist")
            found = self._best_candidate(vertices, used, slot_used, vertex, k, label)
            if found is None:
                return self._stop(path, "nocand")
            vertex, k = self._step(vertices, used, slot_used, (vertex, k), found, touched, path)
            if self._should_stop_at_end(vertices, used, slot_used, vertex, k, label):
                return self._stop(path, "end")

    def _stop(self, path: list[int], reason: str) -> list[int]:
        self.stats.add_stop(reason)
        return path

    def _should_stop_at_end(self, vertices, used, slot_used, vertex, k, label) -> bool:
        """끝 확률로 멈춘다. `stop_needs_nocand`면 이어갈 후보까지 없을 때만 멈춘다 (A4)."""
        if vertices["end_prob"][vertex] <= self.end_thresh:
            return False
        if not self.stop_needs_nocand or k is None or slot_used[vertex, k]:
            return True
        return self._best_candidate(vertices, used, slot_used, vertex, k, label) is None

    def _step(self, vertices, used, slot_used, current, found, touched, path):
        """후보를 사슬에 붙이고 (다음 정점, 계속 확장할 슬롯)을 돌려준다."""
        vertex, k = current
        target, back = found
        slot_used[vertex, k] = slot_used[target, back] = True
        used[target] = True
        touched.append(target)
        path.append(target)
        return target, self._next_slot(vertices, slot_used, target, back)

    def _next_slot(self, vertices, slot_used, vertex, back) -> int | None:
        """되가리킴 슬롯의 반대쪽 활성 슬롯 — 되가리킴 방향과 가장 반대인 것 (R = 2면 남은 하나)."""
        usable = ~slot_used[vertex] & (vertices["exist"][vertex] > self.exist_thresh)
        if not usable.any():
            return None
        dots = vertices["dir"][vertex] @ vertices["dir"][vertex, back]
        return int(np.where(usable, dots, np.inf).argmin())

    def _best_candidate(self, vertices, used, slot_used, vertex, k, label) -> tuple | None:
        """반경 안 미사용 정점 중 게이트(정렬·마주봄·사슬 클래스 확률)를 통과한 비용 최소."""
        nearby = vertices["neighbors"][vertex]
        nearby = nearby[nearby >= 0]
        nearby = nearby[~used[nearby]]
        nearby = nearby[vertices["class_prob"][nearby, label] >= self.min_class_prob]
        if nearby.size == 0:
            return None
        heading = vertices["dir"][vertex, k]
        delta = vertices["point"][nearby] - vertices["point"][vertex]
        distance = np.linalg.norm(delta, axis=-1)
        align = (delta / np.maximum(distance, 1e-9)[:, None]) @ heading
        opposite, back = self._facing_slots(vertices, slot_used, nearby, heading)
        cost = self._candidate_cost(align, distance, opposite)
        best = int(cost.argmin())
        if not np.isfinite(cost[best]):
            return None
        return int(nearby[best]), int(back[best])

    def _candidate_cost(self, align, distance, opposite) -> np.ndarray:
        """게이트를 통과하지 못한 후보는 inf.

        `cosine` 게이트는 각도만 보므로 **먼 후보에게 관대하다** — 45도 안이면 두 칸 떨어진
        엉뚱한 선도 통과한다. `perp` 게이트는 예측 방향 직선에서의 수직 이탈(셀 단위)을 보므로
        거리에 비례해 엄격해진다. 기하적으로 이쪽이 "예측한 선 위에 있는가"에 가깝다 (백로그 A6).
        """
        facing = -opposite >= self.opp_thresh
        shared = self.w_opp * (1.0 + opposite) + DISTANCE_TIEBREAK * distance
        if self.align_mode == "perp":
            perpendicular = distance * np.sqrt(np.maximum(1.0 - align**2, 0.0))
            allowed = facing & (align > 0.0) & (perpendicular <= self.perp_thresh)
            return np.where(allowed, perpendicular + shared, np.inf)
        allowed = facing & (align >= self.align_thresh)
        return np.where(allowed, (1.0 - align) + shared, np.inf)

    def _facing_slots(self, vertices, slot_used, nearby, heading) -> tuple[np.ndarray, np.ndarray]:
        """후보별 되가리킴 슬롯: 활성·미사용 슬롯 중 c . n 이 가장 작은 것 (마주봄 최대)."""
        dots = vertices["dir"][nearby] @ heading  # (M, R)
        usable = (vertices["exist"][nearby] > self.exist_thresh) & ~slot_used[nearby]
        dots = np.where(usable, dots, np.inf)  # 쓸 슬롯이 없으면 inf -> 마주봄 게이트에서 탈락
        back = dots.argmin(axis=1)
        return dots[np.arange(nearby.size), back], back

    def _extensions(self, vertices, slot_used, chain) -> tuple[list, list]:
        """끝 셀에서 남은 활성 슬롯(끝방향) 쪽으로 연장점을 하나 추가한다 (결정 31).

        1셀 사슬(3칸짜리 선)은 남은 슬롯 두 개로 양쪽에 연장해 3점 폴리라인이 된다.
        """
        if len(chain) == 1:
            points = self._extend_points(vertices, slot_used, chain[0])
            return points[:1], points[1:2]
        head = self._extend_points(vertices, slot_used, chain[0])
        tail = self._extend_points(vertices, slot_used, chain[-1])
        return head[:1], tail[:1]

    def _extend_points(self, vertices, slot_used, vertex) -> list[np.ndarray]:
        if vertices["end_prob"][vertex] <= self.end_thresh:
            return []
        usable = ~slot_used[vertex] & (vertices["exist"][vertex] > self.exist_thresh)
        origin = vertices["point"][vertex]
        slots = np.flatnonzero(usable)
        return [origin + vertices["dir"][vertex, k] * self.end_extend for k in slots]

    # --- ③ 후처리 (10.4절) -------------------------------------------------------

    def _to_instance(self, vertices, chain, head_ext, tail_ext, label) -> dict | None:
        """폴리라인 클래스 = 사슬 클래스(시드의 클래스) — 순도 검사가 다수결 일치를 보증한다."""
        points = np.array([*head_ext, *vertices["point"][chain], *tail_ext])
        if points.shape[0] < self.min_points:
            return None
        pixels = points * self.grid_stride - PIXEL_CENTER_SHIFT
        if self.simplify_tol > 0:
            pixels = simplify_polyline(pixels, self.simplify_tol)
        return {
            "class": label,
            "points": pixels.astype(np.float32),
            "score": float(vertices["score"][chain].mean()),
        }
