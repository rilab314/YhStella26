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
        self.color_map_bgr = {info['id']: tuple(info['color'])[::-1] for info in class_ids}

    def visualize(self, output: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], with_img: np.ndarray = None) -> np.ndarray:
        """
        모델 출력과 GT를 각각 시각화하여 좌우로 병합한 이미지를 생성합니다.
        """
        target_img = self.create_visualization_panel(target, 'target', with_img)
        output_img = self.create_visualization_panel(output, 'output', with_img)
        return cv2.hconcat([target_img, output_img])

    def visualize_and_show_arrow(
        self,
        merged: torch.Tensor,
        base_image: Optional[np.ndarray] = None,
        skip: int = 4,
        window_name: str = "arrow_viz",
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
        merged = merged.detach().cpu().numpy()
        H, W, _ = merged.shape
        canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8) if base_image is None else base_image.copy()
        if canvas.shape[0] != img_size or canvas.shape[1] != img_size:
            canvas = cv2.resize(canvas, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

        segm_id   = merged[..., 0].astype(np.int32)
        left_end  = merged[..., 1].astype(np.float32)
        right_end = merged[..., 2].astype(np.float32)
        center    = merged[..., 3:5].astype(np.float32)
        left_pt   = merged[..., 5:7].astype(np.float32)
        right_pt  = merged[..., 7:9].astype(np.float32)

        for y in range(0, H, max(1, int(skip))):
            for x in range(0, W, max(1, int(skip))):
                cls_id = int(segm_id[y, x])
                color = self.color_map_bgr.get(cls_id, (255, 255, 255))
                c_xy_px = (center[y, x] * img_size).astype(np.float32)
                l_xy_px = (left_pt[y, x] * img_size).astype(np.float32)
                r_xy_px = (right_pt[y, x] * img_size).astype(np.float32)
                cls_id = int(segm_id[y, x])
                if cls_id in self.color_map_bgr and cls_id != 0:
                    color = self.color_map_bgr[cls_id]
                    p0 = (int(round(c_xy_px[0])), int(round(c_xy_px[1])))

                    p1 = (int(round(l_xy_px[0])), int(round(l_xy_px[1])))
                    cv2.arrowedLine(canvas, p0, p1, color, 1, tipLength=0.25)

                    p1 = (int(round(r_xy_px[0])), int(round(r_xy_px[1])))
                    cv2.arrowedLine(canvas, p0, p1, color, 1, tipLength=0.25)

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, canvas)
        k = cv2.waitKey(0) & 0xFF
        if k == ord('x'):
            exit(0)
        cv2.destroyAllWindows()
        return canvas

    def create_visualization_panel(self, data: Dict[str, torch.Tensor], mode: str, with_img: Optional[np.ndarray] = None) -> np.ndarray:
        """
        단일 패널(타겟 또는 출력)을 생성합니다.
        """
        if mode == 'output':
            segm_logit = data['segm_logit'].detach().cpu()
            # probs = torch.softmax(segm_logit, dim=-1)
            # max_prob, hard_label = torch.max(probs, dim=-1, keepdim=True)
            # segm_label = hard_label.clone()
            # tau = 0.75
            # segm_label[max_prob < tau] = 0
            segm_label = torch.argmax(segm_logit, dim=-1, keepdim=True)
            left_end_mask = (data['left_end_logit'].detach().cpu() >= 0).numpy()
            right_end_mask = (data['right_end_logit'].detach().cpu() >= 0).numpy()
        else:
            segm_label = data['segm_label'].detach().cpu()
            left_end_mask = (data['left_end'].detach().cpu() == 1).numpy()
            right_end_mask = (data['right_end'].detach().cpu() == 1).numpy()
        return self._draw_segmentation_panel(segm_label, left_end_mask, right_end_mask, with_img)

    def _draw_segmentation_panel(self, segm_label: torch.Tensor, left_end_mask: np.ndarray, right_end_mask: np.ndarray, with_img: Optional[np.ndarray] = None) -> np.ndarray:
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
        segm_up = cv2.resize(segm_label_np.astype(np.uint8), (vis_w, vis_h), interpolation=cv2.INTER_NEAREST)
        for class_id, color_bgr in self.color_map_bgr.items():
            if class_id == 0:
                continue
            vis_img[segm_up == class_id] = color_bgr
        # left_coords = np.argwhere(left_end_mask.squeeze())
        # for y, x in left_coords:
        #     cv2.circle(vis_img, center=(int(x * scale), int(y * scale)), radius=2, color=(255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
        # right_coords = np.argwhere(right_end_mask.squeeze())
        # for y, x in right_coords:
        #     cv2.drawMarker(vis_img, position=(int(x * scale), int(y * scale)), color=(0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=6, thickness=1)
        return vis_img

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

    def _follow_and_draw(self, img: np.ndarray, arr: np.ndarray, start_idx: int, next_lookup: Dict[Tuple[int, int], int], skip: int, img_size: int):
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

def main():
    cls_ids = [
                dict(category_id='000', id=0, priority=11, name='ignore', color=(0, 0, 0)),
                dict(category_id='501', id=1, priority=10, name='center_line', color=(77, 77, 255)),
                dict(category_id='502', id=2, priority=6, name='u_turn_zone_line', color=(77, 178, 255)),
                dict(category_id='503', id=3, priority=7, name='lane_line', color=(77, 255, 77)),
                dict(category_id='504', id=4, priority=3, name='bus_only_lane', color=(255, 153, 77)),
                dict(category_id='505', id=5, priority=8, name='edge_line', color=(255, 77, 77)),
                dict(category_id='506', id=6, priority=4, name='path_change_restriction_line', color=(178, 77, 255)),
                dict(category_id='515', id=7, priority=5, name='no_parking_stopping_line', color=(77, 255, 178)),
                dict(category_id='525', id=8, priority=9, name='guiding_line', color=(255, 178, 77)),
                dict(category_id='530', id=9, priority=0, name='stop_line', color=(77, 102, 255)),
                dict(category_id='531', id=10, priority=1, name='safety_zone', color=(255, 77, 128)),
                dict(category_id='535', id=11, priority=2, name='bicycle_lane', color=(128, 255, 77))
                ]
    loaded = np.load('127.0619,37.5426.npz')
    data = {k: torch.from_numpy(loaded[k]) for k in loaded.files}

    data['segm_logit'] = np.argmax(data['segm_logit'], axis=-1, keepdims=True)
    merged = np.concatenate(list(data.values()), axis=-1)
    merged_tensor = torch.from_numpy(merged).float()

    TLV = TargetLogitVisualizer(cls_ids)
    img = cv2.imread('/workspace/SatelliteDet/dataset/satellite_lane/validation/image/127.0619,37.5426.png')
    if type(img) != np.ndarray:
        print('img is not image')
        exit()

    TLV.visualize_and_show_arrow(merged_tensor, img)

if __name__ == '__main__':
    main()