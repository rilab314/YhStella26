from typing import List, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np
import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from model.dto import LineString, LineNode


class LineStringInstanceGenerator(nn.Module):
    @staticmethod
    def build_from_cfg(cfg):
        return LineStringInstanceGenerator()

    def __init__(self, threshold: float = 0.3):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, output) -> List[List[LineString]]:
        """
        output:
            - segm_logit:      (B, H, W, C)
            - center_point:    (B, H, W, 2)
            - left_point:      (B, H, W, 2)
            - right_point:     (B, H, W, 2)
            - left_end_logit:  (B, H, W, 1)
            - right_end_logit: (B, H, W, 1)
        return:
            - List[List[LineString]]  # length B
        """
        segm_logit = output["segm_logit"]  # (B,H,W,C)
        B = segm_logit.shape[0]
        batch_line_strings: List[List[LineString]] = []

        for b in range(B):
            points, class_ids, scores = self._select_by_argmax(segm_logit[b], self.threshold)
            line_strings = self._make_lines(output, points, class_ids, scores, batch_idx=b)
            batch_line_strings.append(line_strings)

        return batch_line_strings

    def _select_by_argmax(self, segm_logit: torch.Tensor, threshold: float):
        """
        segm_logit: (H, W, C)
        return:
            yx_grid:   (N, 2)  -> (y, x) on HxW grid
            class_ids: (N,)
            scores:    (N,)
        """
        H, W, C = segm_logit.shape  # (H,W,C)
        prob = F.softmax(segm_logit, dim=-1)      # (H,W,C)
        scores, clses = prob.max(dim=-1)         # (H,W), (H,W)
        max_scores = F.max_pool2d(scores, kernel_size=3, stride=1, padding=1)
        peak_scores = max_scores.clone()
        peak_scores[max_scores != scores] = 0.0
        mask = (clses != 0) & (peak_scores > threshold)     # (H,W)

        ys, xs = torch.nonzero(mask, as_tuple=True)         # (N,)
        yx_grid = torch.stack([ys, xs], dim=-1)             # (N,2)
        class_ids = clses[mask]                             # (N,)
        sel_scores = scores[mask]                           # (N,)
        # TODO: sort by score
        return yx_grid, class_ids, sel_scores

    def _make_lines(
        self,
        output: dict,
        points: torch.Tensor,
        class_ids: torch.Tensor,
        scores: torch.Tensor,
        batch_idx: int,
    ) -> List[LineString]:
        """
        단일 이미지(batch_idx)에 대해 LineString 리스트를 생성.
        """
        nodes = self._gather_points_from_mask(output, points, class_ids, scores, batch_idx)

        if len(nodes) == 0:
            return []

        neighbors = self._build_graph(nodes)
        raw_lines = self._trace_line_strings(nodes, neighbors)
        filtered_lines = self._post_process_lines(raw_lines)

        line_strings: List[LineString] = []
        for line in filtered_lines:
            if not line:
                continue

            pts = [(int(nodes[nid].x), int(nodes[nid].y)) for nid in line]
            cls_id = nodes[line[0]].cls
            line_strings.append(LineString(points=pts, class_id=cls_id,))
        return line_strings

    def _gather_points_from_mask(
        self,
        output: dict,
        points: torch.Tensor,      # (N, 2)  -> (grid_y, grid_x)
        class_ids: torch.Tensor,   # (N,)
        scores: torch.Tensor,      # (N,)
        batch_idx: int,
    ) -> List[LineNode]:
        """
        peak로 선택된 (grid_y, grid_x) 위치들에 대해:
        - center_point / left_point / right_point (normalized)를 768 좌표계로 변환
        - left_end_lo
        
        git / right_end_logit으로 끝점 여부 판단
        - LineNode 리스트 생성
        """
        center_point = output["center_point"][batch_idx]      # (H,W,2)
        left_point   = output["left_point"][batch_idx]        # (H,W,2)
        right_point  = output["right_point"][batch_idx]       # (H,W,2)
        left_end_logit  = output["left_end_logit"][batch_idx]   # (H,W,1)
        right_end_logit = output["right_end_logit"][batch_idx]  # (H,W,1)

        H, W, _ = center_point.shape
        # TODO
        img_h, img_w = 768.0, 768.0
        num_points = points.shape[0]
        nodes: List[LineNode] = []

        left_end_prob  = torch.sigmoid(left_end_logit[..., 0])   # (H,W)
        right_end_prob = torch.sigmoid(right_end_logit[..., 0])  # (H,W)

        for node_id in range(num_points):
            grid_y = int(points[node_id, 0].item())
            grid_x = int(points[node_id, 1].item())

            cls_id = int(class_ids[node_id].item())
            score = float(scores[node_id].item())

            # TODO: image coordinates -> grid relative coordinates
            c = center_point[grid_y, grid_x]   # (2,)
            l = left_point[grid_y, grid_x]     # (2,)
            r = right_point[grid_y, grid_x]    # (2,)

            cx = float(c[0].item() * img_w)
            cy = float(c[1].item() * img_h)

            lx = float(l[0].item() * img_w)
            ly = float(l[1].item() * img_h)

            rx = float(r[0].item() * img_w)
            ry = float(r[1].item() * img_h)

            is_left_end = bool(left_end_prob[grid_y, grid_x].item() > self.threshold)
            is_right_end = bool(right_end_prob[grid_y, grid_x].item() > self.threshold)

            node = LineNode(
                id=node_id,
                cls=cls_id,
                score=score,
                grid_y=grid_y,
                grid_x=grid_x,
                x=cx,
                y=cy,
                left_x=lx,
                left_y=ly,
                right_x=rx,
                right_y=ry,
                is_left_end=is_left_end,
                is_right_end=is_right_end,
            )
            nodes.append(node)
        return nodes

    def _build_graph(self, nodes: List[LineNode]) -> List[List[int]]:
        """
        nodes: LineNode 리스트
        return:
            neighbors: List[List[int]]
                - neighbors[i] = [node i와 연결된 node id들]
        """
        num_nodes = len(nodes)
        neighbors: List[List[int]] = [[] for _ in range(num_nodes)]

        if num_nodes == 0:
            return neighbors

        cls_to_indices = {}
        for idx, node in enumerate(nodes):
            cls_to_indices.setdefault(node.cls, []).append(idx)

        D_max = 12.0
        D_max_sq = D_max * D_max

        for i, node in enumerate(nodes):
            candidate_indices = cls_to_indices.get(node.cls, [])

            if not node.is_left_end:
                tx, ty = node.left_x, node.left_y
                best_j = -1
                best_dist_sq = float("inf")

                for j in candidate_indices:
                    if j == i:
                        continue
                    nj = nodes[j]
                    dx = nj.x - tx
                    dy = nj.y - ty
                    dist_sq = dx * dx + dy * dy
                    if dist_sq < best_dist_sq:
                        best_dist_sq = dist_sq
                        best_j = j

                if best_j >= 0 and best_dist_sq <= D_max_sq:
                    neighbors[i].append(best_j)

            if not node.is_right_end:
                tx, ty = node.right_x, node.right_y
                best_j = -1
                best_dist_sq = float("inf")

                for j in candidate_indices:
                    if j == i:
                        continue
                    nj = nodes[j]
                    dx = nj.x - tx
                    dy = nj.y - ty
                    dist_sq = dx * dx + dy * dy
                    if dist_sq < best_dist_sq:
                        best_dist_sq = dist_sq
                        best_j = j

                if best_j >= 0 and best_dist_sq <= D_max_sq:
                    neighbors[i].append(best_j)

        return neighbors

    def _trace_line_strings(
        self,
        nodes: List[LineNode],
        neighbors: List[List[int]]
    ) -> List[List[int]]:
        """
        nodes와 neighbors를 이용해 node id 시퀀스(line)들의 리스트를 만든다.
        """
        num_nodes = len(nodes)
        if num_nodes == 0:
            return []

        in_any_line = [False] * num_nodes
        lines: List[List[int]] = []

        def extend_from(start_id: int, prev_id: int, in_current: set) -> List[int]:
            path: List[int] = []
            curr = start_id
            prev = prev_id

            while True:
                if in_any_line[curr]:
                    break
                if curr in in_current:
                    break

                path.append(curr)
                in_current.add(curr)

                node = nodes[curr]

                if node.is_left_end or node.is_right_end:
                    break

                cand_ids = [n_id for n_id in neighbors[curr]
                            if n_id != prev and not in_any_line[n_id]]
                if not cand_ids:
                    break

                best_id = -1
                best_dist_sq = float("inf")
                for n_id in cand_ids:
                    n_node = nodes[n_id]
                    dx = n_node.x - node.x
                    dy = n_node.y - node.y
                    d2 = dx * dx + dy * dy
                    if d2 < best_dist_sq:
                        best_dist_sq = d2
                        best_id = n_id

                if best_id < 0:
                    break

                prev, curr = curr, best_id

            return path

        for seed_id in range(num_nodes):
            if in_any_line[seed_id]:
                continue

            in_current: set = {seed_id}

            seed_neighbors = [n_id for n_id in neighbors[seed_id]
                              if not in_any_line[n_id]]

            left_path: List[int] = []
            right_path: List[int] = []

            if len(seed_neighbors) >= 1:
                left_start = seed_neighbors[0]
                left_path = extend_from(left_start, seed_id, in_current)

            if len(seed_neighbors) >= 2:
                right_start = seed_neighbors[1]
                if right_start not in in_current and not in_any_line[right_start]:
                    right_path = extend_from(right_start, seed_id, in_current)

            line_ids: List[int] = list(reversed(left_path)) + [seed_id] + right_path

            for nid in line_ids:
                in_any_line[nid] = True

            lines.append(line_ids)

        return lines

    def _post_process_lines(self, lines: List[List[int]]) -> List[List[int]]:
        """
        lines: List[List[int]]  # node_id 시퀀스
        return: 필터링 후 lines
        """
        filtered = [line for line in lines if len(line) >= 4]

        seen = set()
        unique_lines = []
        for line in filtered:
            key = tuple(line)
            if key not in seen:
                seen.add(key)
                unique_lines.append(line)

        return unique_lines

    

def debug_draw_peak_arrows(
    output: dict,
    visualizer,
    instance_generator,
    batch_idx: int = 0,
    img_size: int = 768,
    img: np.ndarray = None
):
    """
    디버깅용 peak arrow 시각화 함수 (클래스 외부 독립 함수)
    - instance_generator._select_by_argmax() 사용
    - cls별 색상: visualizer.color_map_bgr 사용
    """

    # segm_logit[b]: (H,W,C)
    segm_logit_b = output["segm_logit"][batch_idx]
    seg_img = visualizer.create_visualization_panel(data=output, mode='output', with_img=img)

    # peak detection
    points, class_ids, scores = instance_generator._select_by_argmax(
        segm_logit_b, instance_generator.threshold
    )

    # 좌표 정보
    center_point = output["center_point"][batch_idx]      # (H,W,2)
    left_point   = output["left_point"][batch_idx]        # (H,W,2)
    right_point  = output["right_point"][batch_idx]       # (H,W,2)

    H, W, _ = center_point.shape
    img_h, img_w = float(img_size), float(img_size)

    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8) if img is None else img

    num_points = points.shape[0]
    frames = []

    for i in range(num_points):
        gy = int(points[i, 0].item())  # grid y
        gx = int(points[i, 1].item())  # grid x
        cls_id = int(class_ids[i].item())

        # cls 색상 가져오기
        color = visualizer.color_map_bgr.get(cls_id, (0, 255, 0))

        c = center_point[gy, gx]
        l = left_point[gy, gx]
        r = right_point[gy, gx]

        cx = int(round(c[0].item() * img_w))
        cy = int(round(c[1].item() * img_h))
        lx = int(round(l[0].item() * img_w))
        ly = int(round(l[1].item() * img_h))
        rx = int(round(r[0].item() * img_w))
        ry = int(round(r[1].item() * img_h))

        if 0 <= cx < img_size and 0 <= cy < img_size:
            if 0 <= lx < img_size and 0 <= ly < img_size:
                cv2.arrowedLine(canvas, (cx, cy), (lx, ly), color, 1, tipLength=0.25)

            if 0 <= rx < img_size and 0 <= ry < img_size:
                cv2.arrowedLine(canvas, (cx, cy), (rx, ry), color, 1, tipLength=0.25)

                frame = cv2.hconcat([seg_img, canvas])
        
        frame = cv2.hconcat([seg_img, canvas])
        frames.append(frame.copy())  # 🔹 비디오용 프레임 저장

    return canvas, frames


def save_frames_as_video(frames: List[np.ndarray], save_path: str, delay_ms: int = 80):
    if not frames:
        return

    h, w, _ = frames[0].shape
    fps = 1000.0 / delay_ms  # 80ms → 12.5 fps

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

    for f in frames:
        writer.write(f)

    hold_sec = 1.0
    repeat_count = int(fps * hold_sec)  # 12.5 * 1s ≈ 13 frames 추가

    last_frame = frames[-1]
    for _ in range(repeat_count):
        writer.write(last_frame)

    writer.release()


def main():
    cfg = CfgNode.from_file("satellite_detr")
    instance_generator = LineStringInstanceGenerator()
    visualizer = TargetLogitVisualizer(cfg.dataset.labels)
    img_dir = '/workspace/SatelliteDet/dataset/satellite_lane/validation/image'

    torch_data_list = [_ for _ in os.listdir('.') if '.pt' in _]

    for i, data_name in enumerate(torch_data_list):
        img = cv2.imread(os.path.join(img_dir, data_name.replace('.pt', '.png')))
        data = torch.load(data_name)

        for key in data.keys():
            data[key] = data[key].unsqueeze(0)

        line_instances = instance_generator(data)

        # c, v = debug_draw_peak_arrows(data, visualizer=visualizer, instance_generator=instance_generator, img=img)

        for i, line_string in enumerate(line_instances[0]):
            for x, y in line_string.points:
                color = visualizer.color_map_bgr[line_string.class_id]
                xi = int(x)
                yi = int(y)
                if 0 <= xi < img.shape[1] and 0 <= yi < img.shape[0]:
                    cv2.circle(img, (xi, yi), 3, color, thickness=-1)
        


        cv2.imwrite(data_name.replace('.pt', '.png'), img)
        # seg_img = visualizer.create_visualization_panel(data=data, mode='output', with_img=img)
        # v.append(cv2.hconcat([seg_img, img]))
        # save_frames_as_video(v, data_name.replace('.pt', '.mp4'), delay_ms=80)


if __name__ == '__main__':
    from util.target_logit_visualizer import TargetLogitVisualizer
    from configs.config import CfgNode
    import cv2

    main()
