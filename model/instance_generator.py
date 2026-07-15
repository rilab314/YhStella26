import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os, sys
import json
from typing import Any, Dict, List
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from configs.config import CfgNode

class GeneratePolylineInstances:
    def build_from_cfg(cfg):
        instance_generator = GeneratePolylineInstances(cfg.dataset.labels)
        return instance_generator

    def __init__(self, class_ids=None, conf_threshold=0.5, scale=4):
        self.color_map_bgr = {class_dict['id']: class_dict['color'] for class_dict in class_ids}
        self.conf_threshold = conf_threshold
        self.scale = scale
        self.peak_masks = []

    def __call__(self, outputs: List[Dict[str, torch.Tensor]]) -> List[List[Dict[str, Any]]]:
        """
        outputs: List[Dict[str, torch.Tensor]]
        """
        if not isinstance(outputs, list):
            raise TypeError(f"GeneratePolylineInstances expects List[Dict[str, Tensor]], got {type(outputs)}")

        batch_results = []
        batch_size = len(outputs)
        self.peak_masks = []

        for b in range(batch_size):
            img_outputs = outputs[b]
            img_outputs = {k: v.detach().cpu().numpy() for k, v in img_outputs.items()}
            img_outputs['segm_prob'] = self._softmax(img_outputs['segm_logit'])
            img_outputs['left_end_prob'] = self._sigmoid(img_outputs['left_end_logit'])
            img_outputs['right_end_prob'] = self._sigmoid(img_outputs['right_end_logit'])
            
            segm_prob = img_outputs['segm_prob']
            segm_prob_max = np.max(segm_prob, axis=-1) # (gh, gw)
            segm_class = np.argmax(segm_prob, axis=-1) # (gh, gw)
            img_outputs['segm_class'] = segm_class
            
            prob_tensor = torch.from_numpy(segm_prob_max).unsqueeze(0).unsqueeze(0)
            max_pool = F.max_pool2d(prob_tensor, kernel_size=3, stride=1, padding=1)
            is_local_max = (prob_tensor == max_pool).squeeze().numpy() & (segm_class > 0) 
            mask = (segm_prob_max > self.conf_threshold) & is_local_max
            ys, xs = np.nonzero(mask)
            self.peak_masks.append(mask)
            if len(ys) == 0:
                batch_results.append([])
                continue
            # peak point들을 확률순으로 정렬
            probs = segm_prob_max[ys, xs]
            sort_idx = np.argsort(probs)[::-1]
            ys, xs = ys[sort_idx], xs[sort_idx]
            
            polylines = []
            visited_map = np.zeros_like(mask, dtype=bool)

            for i, (y, x) in enumerate(zip(ys, xs)):
                if visited_map[y, x]:
                    continue
                
                category = int(segm_class[y, x])
                start_pt = np.array([x, y]) 
                polyline = self.find_graph(start_pt, category, img_outputs, visited_map)

                if len(polyline) > 0:
                    gh, gw = mask.shape
                    pts_grid = polyline * np.array([gw, gh])
                    pts_int = np.rint(pts_grid).astype(np.int32)
                    valid_mask = (
                        (pts_int[:, 0] >= 0) & (pts_int[:, 0] < gw) &
                        (pts_int[:, 1] >= 0) & (pts_int[:, 1] < gh)
                    )
                    valid_pts = pts_int[valid_mask]
                    if valid_pts.shape[0] == 1:
                        visited_map[valid_pts[0, 1], valid_pts[0, 0]] = True
                    elif valid_pts.shape[0] > 1:
                        line_mask = np.zeros((gh, gw), dtype=np.uint8)
                        cv2.polylines(line_mask, [valid_pts.reshape(-1, 1, 2)], False, 1, thickness=1, lineType=cv2.LINE_8)
                        visited_map |= (line_mask > 0)

                polylines.append({'label': category, 'points': polyline})

            batch_results.append(polylines)

        return batch_results

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def find_graph(self, point, category, outputs, visited_map):
        x, y = point
        
        center_pt = point + outputs['center_point'][y, x]
        left_pt = center_pt + outputs['left_point'][y, x]
        right_pt = center_pt + outputs['right_point'][y, x]
        
        left_line = [center_pt, left_pt]
        right_line = [center_pt, right_pt]
        left_line = self.trace_graph(left_line, outputs, visited_map, category)
        right_line = self.trace_graph(right_line, outputs, visited_map, category)
        
        final_line = left_line[1:][::-1] + right_line
        final_line = np.array(final_line, dtype=np.float32)
        
        gh, gw = outputs['segm_prob'].shape[:2]
        final_line_norm = final_line / np.array([gw, gh], dtype=np.float32)
        
        return final_line_norm

    def trace_graph(self, polyline, outputs, visited_map, category):
        if len(polyline) > 500:
            return polyline
            
        curr_tip = polyline[-1]
        prev_tip = polyline[-2]
        cx, cy = int(curr_tip[0]), int(curr_tip[1])
        gh, gw = outputs['segm_prob'].shape[:2]
        if not (0 <= cx < gw and 0 <= cy < gh):
            return polyline

        move_vec = curr_tip - prev_tip
        move_norm = np.linalg.norm(move_vec)
        if move_norm < 1e-9:
            return polyline
        move_vec = move_vec / move_norm
        left_rel_point = outputs['left_point'][cy, cx]
        left_norm = np.linalg.norm(left_rel_point)
        if left_norm < 1e-9:
            left_rel_point = np.zeros_like(left_rel_point, dtype=np.float32)
        else:
            left_rel_point = left_rel_point / left_norm
        right_rel_point = outputs['right_point'][cy, cx]
        right_norm = np.linalg.norm(right_rel_point)
        if right_norm < 1e-9:
            right_rel_point = np.zeros_like(right_rel_point, dtype=np.float32)
        else:
            right_rel_point = right_rel_point / right_norm
        
        cos_left = np.dot(move_vec, left_rel_point)
        cos_right = np.dot(move_vec, right_rel_point)
        if cos_left > cos_right:
            target_vec = left_rel_point
            end_prob = outputs['left_end_prob'][cy, cx]
        else:
            target_vec = right_rel_point
            end_prob = outputs['right_end_prob'][cy, cx]

        # 5x5 이웃(자기 자신 제외)에서 같은 category를 찾고
        # target_vec와의 cosine similarity가 가장 높은 픽셀을 next 후보로 사용
        x0, x1 = max(0, cx - 2), min(gw, cx + 3)
        y0, y1 = max(0, cy - 2), min(gh, cy + 3)
        win_h, win_w = (y1 - y0), (x1 - x0)
        ys, xs = np.indices((win_h, win_w), dtype=np.int32)
        ys = ys + y0
        xs = xs + x0
        coords = np.stack([xs, ys], axis=-1).astype(np.int32)  # (H, W, 2) with (x, y)

        cat_mask = (outputs['segm_class'][y0:y1, x0:x1] == category)
        not_self_mask = ~((coords[..., 0] == cx) & (coords[..., 1] == cy))
        cand_mask = cat_mask & not_self_mask
        next_points = coords.astype(np.float32) + outputs['center_point'][y0:y1, x0:x1]
        in_range = (
            (next_points[..., 0] >= 0.0) & (next_points[..., 0] < float(gw)) &
            (next_points[..., 1] >= 0.0) & (next_points[..., 1] < float(gh))
        )
        valid = cand_mask & in_range

        best_xy = None
        if np.any(valid):
            rel_vec = coords.astype(np.float32) - np.array([cx, cy], dtype=np.float32)
            rel_norm = np.linalg.norm(rel_vec, axis=-1) + 1e-9
            unit_vec = np.zeros_like(rel_vec, dtype=np.float32)
            unit_vec[valid] = rel_vec[valid] / rel_norm[valid, None]
            cos_map = np.sum(unit_vec * target_vec.astype(np.float32), axis=-1)
            valid = valid & (cos_map > 1/np.sqrt(2))
            
            if np.any(valid):
                cos_map[~valid] = -1e9
                flat_idx = int(np.argmax(cos_map))
                by, bx = np.unravel_index(flat_idx, cos_map.shape)
                best_xy = (int(coords[by, bx, 0]), int(coords[by, bx, 1]))

        if best_xy is not None:
            nx, ny = best_xy
            next_point = np.array([nx, ny], dtype=np.float32) + outputs['center_point'][ny, nx]
            if end_prob > 0.5:
                polyline.append(next_point)
                return polyline
            if self._is_valid_point(next_point, outputs, visited_map, category):
                polyline.append(next_point)
                return self.trace_graph(polyline, outputs, visited_map, category)
        
        return polyline
    
    def _is_within_image(self, next_pt, outputs):
        gh, gw = outputs['segm_prob'].shape[:2]
        if not (0 <= next_pt[0] < gw and 0 <= next_pt[1] < gh):
            return False
        return True
        
    def _is_valid_point(self, pt, outputs, visited_map, category):
        """Check point validity(bounds, visited status, class matching)"""
        nx, ny = int(pt[0]), int(pt[1])
        
        if not self._is_within_image(pt, outputs):
            return False
        if visited_map[ny, nx]:
            return False
        if outputs['segm_class'][ny, nx] != category:
            return False
        return True

    def save_points_to_json(self, data: List[Dict[str, Any]], save_path: str) -> None:
        records = data

        fixed = []
        for rec in records:
            fixed.append({
                "label": int(rec["label"]),
                "points": np.asarray(rec["points"], dtype=np.float32).tolist(),
            })

        with open(save_path, "w", encoding="utf-8") as f:
            f.write("[\n")
            for i, rec in enumerate(fixed):
                f.write("  {\n")
                f.write(f'    "label": {rec["label"]},\n')
                f.write(f'    "points": {json.dumps(rec["points"], ensure_ascii=False, separators=(", ", ": "))}\n')
                f.write("  }")
                f.write(",\n" if i < len(fixed) - 1 else "\n")
            f.write("]\n")

    def visualize_process(
        self,
        image: np.ndarray,
        labels_tensor: np.ndarray,
        pred_output: Dict[str, Any],
        polylines: Any,
        batch_index: int = 0,
        end_thr: float = 0.5,
    ) -> np.ndarray:
        """
        1) GT map(이미지 위 오버레이)
        2) segmentation + peak(filled) + end(empty) (이미지 위 오버레이)
        3) instance result(기존 visualize_result 재사용)
        를 생성하여 좌우로 연결한다.
        """
        h, w = image.shape[:2]

        # ---------- panel 1: GT overlay ----------
        gt_panel = image.copy()
        if labels_tensor.ndim == 3 and labels_tensor.shape[-1] >= 9:
            segm_label = labels_tensor[:, :, 8]
            segm_up = cv2.resize(segm_label.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)
            gt_color = np.zeros((h, w, 3), dtype=np.uint8)
            for cid, color in self.color_map_bgr.items():
                if cid == 0:
                    continue
                gt_color[segm_up == cid] = color
            gt_panel = cv2.addWeighted(gt_panel, 0.5, gt_color, 0.5, 0)
        cv2.putText(gt_panel, "GT MAP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        # ---------- panel 2: SEG + PEAK + END overlay ----------
        seg_panel = image.copy()
        out = {}
        for k, v in pred_output.items():
            if torch.is_tensor(v):
                out[k] = v.detach().cpu().numpy()
            else:
                out[k] = v

        segm_prob = np.exp(out["segm_logit"] - np.max(out["segm_logit"], axis=-1, keepdims=True))
        segm_prob = segm_prob / np.sum(segm_prob, axis=-1, keepdims=True)
        segm_class = np.argmax(segm_prob, axis=-1).astype(np.int32)
        gh, gw = segm_class.shape[:2]
        scale_x = w / float(gw)
        scale_y = h / float(gh)

        seg_color = np.zeros((h, w, 3), dtype=np.uint8)
        segm_up = cv2.resize(segm_class, (w, h), interpolation=cv2.INTER_NEAREST)
        for cid, color in self.color_map_bgr.items():
            if cid == 0:
                continue
            seg_color[segm_up == cid] = color
        seg_panel = cv2.addWeighted(seg_panel, 0.5, seg_color, 0.5, 0)

        center_point = out["center_point"]
        left_point = out["left_point"]
        right_point = out["right_point"]

        # visualize left/right direction arrows on even (row, col) only, excluding ignore class(0)
        for y0 in range(0, gh, 2):
            for x0 in range(0, gw, 2):
                if int(segm_class[y0, x0]) == 0:
                    continue
                cx = int(round((x0 + float(center_point[y0, x0, 0])) * scale_x))
                cy = int(round((y0 + float(center_point[y0, x0, 1])) * scale_y))
                lx = int(round(cx + float(left_point[y0, x0, 0]) * scale_x))
                ly = int(round(cy + float(left_point[y0, x0, 1]) * scale_y))
                rx = int(round(cx + float(right_point[y0, x0, 0]) * scale_x))
                ry = int(round(cy + float(right_point[y0, x0, 1]) * scale_y))

                if 0 <= cx < w and 0 <= cy < h:
                    lx = int(round(cx + 3.0 * (lx - cx)))
                    ly = int(round(cy + 3.0 * (ly - cy)))
                    rx = int(round(cx + 3.0 * (rx - cx)))
                    ry = int(round(cy + 3.0 * (ry - cy)))
                    if 0 <= lx < w and 0 <= ly < h:
                        cv2.arrowedLine(seg_panel, (cx, cy), (lx, ly), (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.25)
                    if 0 <= rx < w and 0 <= ry < h:
                        cv2.arrowedLine(seg_panel, (cx, cy), (rx, ry), (0, 0, 0), 1, cv2.LINE_AA, tipLength=0.25)

        peak_mask = None
        if len(self.peak_masks) > batch_index:
            peak_mask = self.peak_masks[batch_index]
        if peak_mask is None:
            peak_mask = np.zeros((gh, gw), dtype=bool)

        left_end_prob = 1.0 / (1.0 + np.exp(-out["left_end_logit"]))
        right_end_prob = 1.0 / (1.0 + np.exp(-out["right_end_logit"]))

        ys, xs = np.where(peak_mask)
        for y0, x0 in zip(ys.tolist(), xs.tolist()):
            cx = int(round((x0 + float(center_point[y0, x0, 0])) * scale_x))
            cy = int(round((y0 + float(center_point[y0, x0, 1])) * scale_y))
            if 0 <= cx < w and 0 <= cy < h:
                cv2.circle(seg_panel, (cx, cy), 3, (255, 255, 255), -1)  # peak: filled

        ys, xs = np.where(left_end_prob[...,0] > end_thr)
        for y0, x0 in zip(ys.tolist(), xs.tolist()):
            cx = int(round((x0 + float(center_point[y0, x0, 0])) * scale_x))
            cy = int(round((y0 + float(center_point[y0, x0, 1])) * scale_y))
            ex = int(round(cx + float(left_point[y0, x0, 0]) * scale_x))
            ey = int(round(cy + float(left_point[y0, x0, 1]) * scale_y))
            if 0 <= ex < w and 0 <= ey < h:
                cv2.circle(seg_panel, (ex, ey), 5, (0, 255, 0), 1)  # end: empty
        
        ys, xs = np.where(right_end_prob[...,0] > end_thr)
        for y0, x0 in zip(ys.tolist(), xs.tolist()):
            cx = int(round((x0 + float(center_point[y0, x0, 0])) * scale_x))
            cy = int(round((y0 + float(center_point[y0, x0, 1])) * scale_y))
            ex = int(round(cx + float(right_point[y0, x0, 0]) * scale_x))
            ey = int(round(cy + float(right_point[y0, x0, 1]) * scale_y))
            if 0 <= ex < w and 0 <= ey < h:
                cv2.circle(seg_panel, (ex, ey), 5, (255, 0, 0), 1)  # end: empty

        cv2.putText(seg_panel, "SEG+PEAK+END", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        # ---------- panel 3: instance result ----------
        poly = polylines
        if isinstance(polylines, list) and len(polylines) > 0 and isinstance(polylines[0], list):
            poly = polylines[0]
        inst_panel = self.visualize_result(image, poly)
        cv2.putText(inst_panel, "INSTANCE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        return cv2.hconcat([gt_panel, seg_panel, inst_panel])

    def visualize_result(self, image: np.ndarray, polylines):
        vis_img = image.copy()
        W, H, _ = vis_img.shape

        for poly_info in polylines:
            cat = poly_info['label']
            pts = poly_info['points'] # (N, 2) Normalized Coords (0~1)
            
            pts_pixel = (pts * np.array([W, H])).astype(np.int32)
            if len(pts_pixel) < 30:
                continue
            
            color = self.color_map_bgr.get(cat, (255, 255, 255))
            cv2.polylines(vis_img, [pts_pixel], isClosed=False, color=color, thickness=2)
            
            for p in pts_pixel:
                cv2.circle(vis_img, tuple(p), 2, color, -1)
            
            if len(pts_pixel) > 0:
                cv2.circle(vis_img, tuple(pts_pixel[0]), 5, (0, 255, 0), -1)
                cv2.circle(vis_img, tuple(pts_pixel[-1]), 5, (0, 0, 255), -1)

        return vis_img


def main():
    torch_path = '/home/gorilla/kyh_workspace/project/results/log_260209_0750/checkpoints/last_pt'
    gt_path = '/home/gorilla/kyh_workspace/project/dataset/satellite_lane/validation/label'
    img_path = '/home/gorilla/kyh_workspace/project/dataset/satellite_lane/validation/image'
    save_path = torch_path.replace('_pt', '_instance')
    os.makedirs(save_path, exist_ok=True)
    torch_list = os.listdir(torch_path)
    cfg = CfgNode.from_file("stella_cfg")
    instance_generator = GeneratePolylineInstances.build_from_cfg(cfg)

    for torch_name in torch_list:
        pred_data = torch.load(os.path.join(torch_path, torch_name))
        img_name = os.path.basename(torch_name).replace('.pt', '.png')
        img = cv2.imread(os.path.join(img_path, img_name))
        label_name = os.path.basename(torch_name).replace('.pt', '.npy')
        label = np.load(os.path.join(gt_path, label_name))

        polylines = instance_generator([pred_data])
        # instance_generator.save_points_to_json(polylines[0], os.path.join(save_path, torch_name).replace('.pt', '.json'))
        process_img = instance_generator.visualize_process(img, label, pred_data, polylines)
        cv2.imshow('d', process_img)
        k = cv2.waitKey()
        if k == 120:
            exit()





if __name__ == '__main__':
    from tqdm import tqdm
    main()
