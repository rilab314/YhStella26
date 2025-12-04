import cv2
import numpy as np
import torch
import pandas as pd
from typing import Optional, List, Dict, Tuple


class TargetLogitVisualizer:
    def __init__(self, class_ids: List[Dict]):
        """
        클래스 ID와 RGB 색상 정보를 초기화합니다.
        """
        self.class_ids = class_ids
        self.color_map_bgr = {info["id"]: tuple(info["color"])[::-1] for info in class_ids}

    def visualize(
        self,
        output: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        with_img: np.ndarray = None,
    ) -> np.ndarray:
        """
        모델 출력과 GT를 각각 시각화하여 좌우로 병합한 이미지를 생성합니다.
        """
        target_img = self.create_visualization_panel(target, "target", with_img)
        output_img = self.create_visualization_panel(output, "output", with_img)
        return cv2.hconcat([target_img, output_img])

    def create_visualization_panel(
        self,
        data: Dict[str, torch.Tensor],
        mode: str,
        with_img: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        단일 패널(타겟 또는 출력)을 생성합니다.
        """
        if mode == "target":
            segm_label = data["segm_label"].detach().cpu()
            left_end_mask = (data["left_end"].detach().cpu() == 1).numpy()
            right_end_mask = (data["right_end"].detach().cpu() == 1).numpy()
            return self._draw_segmentation_panel(segm_label, left_end_mask, right_end_mask, with_img)

        segm_label = torch.argmax(data["segm_logit"], dim=-1, keepdim=True)
        segm_label_mask = segm_label > 0
        for k, v in data.items():
            if type(v) == torch.Tensor:
                data[k] = v * segm_label_mask.expand_as(v)

        if mode == "output":
            segm_label = segm_label.detach().cpu()
            left_end_mask = (data["left_end_logit"].detach().cpu() >= 0).numpy()
            right_end_mask = (data["right_end_logit"].detach().cpu() >= 0).numpy()
            return self._draw_segmentation_panel(segm_label, left_end_mask, right_end_mask, with_img)

        elif mode == "accuracy":
            segm_logit = torch.softmax(data["segm_logit"], dim=-1)
            segm_max_label_vals, segm_max_label_idxs = torch.max(segm_logit, dim=2, keepdim=True)
            segm_max_label_vals = segm_max_label_vals.detach().cpu().numpy()
            segm_max_label_idxs = segm_max_label_idxs.detach().cpu().numpy()
            return self._draw_segmentation_panel_with_acc(segm_max_label_vals, segm_max_label_idxs, with_img)

        elif mode == "arrow":
            for k, v in data.items():
                if type(v) == torch.Tensor:
                    data[k] = v.detach().cpu().numpy()
            segm_label = segm_label.detach().cpu().numpy()
            center_point = data['center_point']
            left_point = data['left_point']
            right_point = data['right_point']
            return self.visualize_and_show_arrow(segm_label, center_point, left_point, right_point, with_img=with_img)
        
    def _draw_segmentation_panel(
        self,
        segm_label: torch.Tensor,
        left_end_mask: np.ndarray,
        right_end_mask: np.ndarray,
        with_img: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        segm_label과 end 마스크를 이용해 컬러 맵과 마커를 그립니다.
        """
        segm_label_np = segm_label.numpy().squeeze()
        H, W = segm_label_np.shape
        scale = 4
        vis_h, vis_w = H * scale, W * scale

        if with_img is None:
            vis_img = np.zeros((vis_h, vis_w, 3), dtype=np.uint8)
        else:
            vis_img = with_img.copy()
            if vis_img.shape[:2] != (vis_h, vis_w):
                vis_img = cv2.resize(vis_img, (vis_w, vis_h), interpolation=cv2.INTER_LINEAR)

        segm_up = cv2.resize(
            segm_label_np.astype(np.uint8),
            (vis_w, vis_h),
            interpolation=cv2.INTER_NEAREST,
        )

        for class_id, color_bgr in self.color_map_bgr.items():
            if class_id == 0:
                continue
            vis_img[segm_up == class_id] = color_bgr

        return vis_img

    def _draw_segmentation_panel_with_acc(
        self,
        segm_max_label_vals: np.ndarray,
        segm_max_label_idxs: np.ndarray,
        with_img: Optional[np.ndarray] = None,
    ):
        H, W = segm_max_label_idxs.shape[:2]
        scale_up = 4
        vis_h, vis_w = H * scale_up, W * scale_up

        if with_img is None:
            vis_img = np.zeros((vis_h, vis_w, 3), dtype=np.uint8)
        else:
            vis_img = with_img
            if vis_img.shape[:2] != (vis_h, vis_w):
                vis_img = cv2.resize(vis_img, (vis_w, vis_h), interpolation=cv2.INTER_LINEAR)
            if vis_img.dtype != np.uint8:
                vis_img = vis_img.astype(np.uint8)

        segm_up = cv2.resize(
            segm_max_label_idxs.astype(np.int32),
            (vis_w, vis_h),
            interpolation=cv2.INTER_NEAREST,
        )
        if segm_up.ndim == 3:
            segm_up = segm_up[..., 0]

        vals_up = cv2.resize(
            segm_max_label_vals.astype(np.float32),
            (vis_w, vis_h),
            interpolation=cv2.INTER_LINEAR,
        )
        if vals_up.ndim == 3:
            vals_up = vals_up[..., 0]

        for cls_id, color in self.color_map_bgr.items():
            if cls_id == 0:
                continue

            mask = (segm_up == cls_id)
            if not np.any(mask):
                continue
            acc = np.zeros_like(vals_up, dtype=np.float32)
            acc[mask] = vals_up[mask]
            acc_safe = np.where(acc == 0, 1, acc)
            acc_safe = np.clip(acc_safe, 0.0, 1.0)
            acc_safe = np.floor(acc_safe * 5.0) / 5.0

            color_f = np.array(color, dtype=np.uint8)
            scaled = color_f * acc_safe[..., None]
            scaled = np.clip(scaled, 0, 255).astype(np.uint8)

            vis_img[mask] = scaled[mask]

        return vis_img

    def visualize_and_show_arrow(
        self,
        segm_label: torch.Tensor,
        center_point: torch.Tensor,
        left_point: torch.Tensor,
        right_point: torch.Tensor,
        with_img: Optional[np.ndarray] = None,
        skip: int = 4,
        img_size: int = 768,
    ) -> np.ndarray:
        """
        (H,W,9) 텐서/배열을 받아 화살표를 시각화하고 표시.
        0: segm_id(정수 클래스)
        1: left_end_logit
        2: right_end_logit
        3-4: center_point (cx, cy)
        5-6: left_point   (lx, ly)
        7-8: right_point  (rx, ry)
        """ 
        H, W, _ = segm_label.shape
        canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8) if with_img is None else with_img.copy()
        if canvas.shape[0] != img_size or canvas.shape[1] != img_size:
            canvas = cv2.resize(canvas, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

        for y in range(0, H, max(1, int(skip))):
            for x in range(0, W, max(1, int(skip))):
                cls_id = int(segm_label[y, x])
                color = self.color_map_bgr.get(cls_id, (255, 255, 255))
                c_xy_px = (center_point[y, x] * img_size).astype(np.float32)
                l_xy_px = (left_point[y, x] * img_size).astype(np.float32)
                r_xy_px = (right_point[y, x] * img_size).astype(np.float32)
                cls_id = int(segm_label[y, x])
                if cls_id in self.color_map_bgr and cls_id != 0:
                    color = self.color_map_bgr[cls_id]
                    p0 = (int(round(c_xy_px[0])), int(round(c_xy_px[1])))

                    p1 = (int(round(l_xy_px[0])), int(round(l_xy_px[1])))
                    cv2.arrowedLine(canvas, p0, p1, color, 1, tipLength=0.25)

                    p1 = (int(round(r_xy_px[0])), int(round(r_xy_px[1])))
                    cv2.arrowedLine(canvas, p0, p1, color, 1, tipLength=0.25)
        return canvas

    def _to_px(self, x: float, y: float, img_size: int) -> Tuple[int, int]:
        """
        정규화 좌표를 픽셀 좌표로 변환합니다.
        """
        px = int(round(x * (img_size - 1)))
        py = int(round(y * (img_size - 1)))
        px = max(0, min(img_size - 1, px))
        py = max(0, min(img_size - 1, py))
        return px, py

    def _build_index(self, points: np.ndarray, img_size: int) -> Dict[Tuple[int, int], int]:
        """
        (x,y) 픽셀 좌표를 키로 하는 인덱스를 구축합니다.
        """
        idx = {}
        for i in range(points.shape[0]):
            x, y = float(points[i, 1]), float(points[i, 2])
            key = self._to_px(x, y, img_size)
            idx[key] = i
        return idx

    def _follow_and_draw(
        self,
        img: np.ndarray,
        arr: np.ndarray,
        start_idx: int,
        next_lookup: Dict[Tuple[int, int], int],
        skip: int,
        img_size: int,
    ):
        """
        start 포인트부터 next 체인을 따라가며 간격(skip)에 맞춰 화살표를 그리고, 마지막 점은 점으로 표시합니다.
        """
        visited = set()
        i = start_idx
        step = 0
        while True:
            if i in visited:
                break
            visited.add(i)
            cls_id = int(arr[i, 0])
            color = self.color_map_bgr.get(cls_id, (255, 255, 255))
            x, y = float(arr[i, 1]), float(arr[i, 2])
            nx, ny = float(arr[i, 5]), float(arr[i, 6])
            is_end = int(arr[i, 8]) == 1
            p0 = self._to_px(x, y, img_size)
            if is_end:
                cv2.circle(img, p0, radius=4, color=color, thickness=-1, lineType=cv2.LINE_AA)
                break
            p1 = self._to_px(nx, ny, img_size)
            next_idx = next_lookup.get(p1, None)
            if next_idx is None:
                cv2.circle(img, p0, radius=4, color=color, thickness=-1, lineType=cv2.LINE_AA)
                break
            if step % max(1, int(skip)) == 0:
                cv2.arrowedLine(img, p0, p1, color, thickness=2, tipLength=0.25)
            if int(arr[next_idx, 8]) == 1:
                cv2.circle(img, p1, radius=4, color=color, thickness=-1, lineType=cv2.LINE_AA)
                break
            i = next_idx
            step += 1
