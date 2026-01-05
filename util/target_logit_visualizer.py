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
        self.l_r_diff_debug = [0, 0]

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
            return self._draw_segmentation_panel(segm_label, with_img)

        segm_label = data['segm_logit']
        segm_label = torch.argmax(data["segm_logit"], dim=-1, keepdim=True)
        segm_label_mask = segm_label > 0
        for k, v in data.items():
            if type(v) == torch.Tensor:
                data[k] = v * segm_label_mask.expand_as(v)

        if mode == "output":
            segm_label = segm_label.detach().cpu()
            return self._draw_segmentation_panel(segm_label, with_img)

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
        with_img: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        segm_label과 end 마스크를 이용해 컬러 맵과 마커를 그립니다.
        """
        segm_label_np = segm_label.numpy().squeeze()
        H, W = segm_label_np.shape
        scale = 4
        vis_h, vis_w = H * scale, W * scale
        canvas = np.zeros((vis_h, vis_w, 3), dtype=np.uint8) if with_img is None else with_img.copy()

        segm_up = cv2.resize(segm_label_np.astype(np.uint8),(vis_w, vis_h),interpolation=cv2.INTER_NEAREST)
        for class_id, color_bgr in self.color_map_bgr.items():
            if class_id == 0:
                continue
            canvas[segm_up == class_id] = color_bgr
        return canvas

    def _draw_segmentation_panel_with_acc(
        self,
        segm_max_label_vals: np.ndarray,
        segm_max_label_idxs: np.ndarray,
        with_img: Optional[np.ndarray] = None,
    ):
        H, W = segm_max_label_idxs.shape[:2]
        scale = 4
        vis_h, vis_w = H * scale, W * scale

        canvas = np.zeros((vis_h, vis_w, 3), dtype=np.uint8) if with_img is None else with_img.copy()

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

            canvas[mask] = scaled[mask]

        return canvas

    def visualize_and_show_arrow(
        self,
        segm_label: np.ndarray,
        center_point: np.ndarray,
        left_point: np.ndarray,
        right_point: np.ndarray,
        with_img: Optional[np.ndarray] = None,
        skip: int = 4,
        img_size: int = 768,
        scale: int = 4,
    ) -> np.ndarray:
        img = with_img if with_img is not None else np.zeros((768, 768, 3), dtype=np.uint8)

        for class_id, color_bgr in self.color_map_bgr.items():
            if class_id == 0:
                continue

            segm_2d = segm_label[..., 0]
            ys, xs = np.where(segm_2d == class_id)
            num_pts = len(xs)

            if num_pts == 0:
                continue

            for i in range(0, num_pts, skip):
                y = int(ys[i])
                x = int(xs[i])
                segm_pt = (int(x * scale), int(y * scale))

                lp = left_point[y, x]
                rp = right_point[y, x]
                cp = center_point[y, x]
                left_pt = (int(segm_pt[0] + lp[0] * scale), int(segm_pt[1] + lp[1] * scale))
                right_pt = (int(segm_pt[0] + rp[0] * scale), int(segm_pt[1] + rp[1] * scale))
                center_pt = (int(segm_pt[0] + cp[0] * scale), int(segm_pt[1] + cp[1] * scale))

                if 0 <= left_pt[0] < img_size and 0 <= left_pt[1] < img_size:
                    cv2.arrowedLine(img, center_pt, left_pt, color_bgr, 1, cv2.LINE_AA, tipLength=0.1)

                if 0 <= right_pt[0] < img_size and 0 <= right_pt[1] < img_size:
                    cv2.arrowedLine(img, center_pt, right_pt, color_bgr, 1, cv2.LINE_AA, tipLength=0.1)

        return img

    def visualize_and_show_arrow_legacy(
        self,
        segm_label: np.ndarray,
        center_point: np.ndarray,
        left_point: np.ndarray,
        right_point: np.ndarray,
        with_img: Optional[np.ndarray] = None,
        skip: int = 4,
        img_size: int = 768,
        scale: int = 4,
    ) -> np.ndarray:
        img = with_img if with_img is not None else np.zeros((768, 768, 3), dtype=np.uint8)

        segm_2d = segm_label[..., 0]
        mask = segm_label != 0
        center_point *= mask
        left_point *= mask
        right_point *= mask

        for class_id, color_bgr in self.color_map_bgr.items():
            if class_id == 0:
                continue

            segm_2d = segm_label[..., 0]
            ys, xs = np.where(segm_2d == class_id)
            num_pts = len(xs)

            if num_pts == 0:
                continue

            for i in range(0, num_pts, skip):
                y = int(ys[i])
                x = int(xs[i])

                left_pt = (left_point[y, x] * 768).astype(int)
                right_pt = (right_point[y, x] * 768).astype(int)
                center_pt = (center_point[y, x] * 768).astype(int)

                left_pt_vec = center_pt - left_pt
                right_pt_vec = center_pt - right_pt

                left_pt += 4 * left_pt_vec
                right_pt += 4 * right_pt_vec

                if 0 <= left_pt[0] < img_size and 0 <= left_pt[1] < img_size:
                    cv2.arrowedLine(img, center_pt, left_pt, color_bgr, 1, cv2.LINE_AA, tipLength=0.1)

                if 0 <= right_pt[0] < img_size and 0 <= right_pt[1] < img_size:
                    cv2.arrowedLine(img, center_pt, right_pt, color_bgr, 1, cv2.LINE_AA, tipLength=0.1)

        return img
