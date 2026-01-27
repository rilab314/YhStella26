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

    def __call__(self, outputs):
        """
        outputs: Dict[str: torch.Tensor]
        """
        batch_results = []
        batch_size = len(outputs)

        for b in range(batch_size):
            img_outputs = outputs[b] if type(outputs) == list else outputs # TODO: 정리 필요
            img_outputs = {k: v.detach().cpu().numpy() for k, v in img_outputs.items()}
            img_outputs['segm_prob'] = self._softmax(img_outputs['segm_logit'])
            img_outputs['left_end_prob'] = self._sigmoid(img_outputs['left_end_logit'])
            img_outputs['right_end_prob'] = self._sigmoid(img_outputs['right_end_logit'])
            
            segm_prob = img_outputs['segm_prob']
            segm_prob_max = np.max(segm_prob, axis=-1) # (gh, gw)
            segm_class = np.argmax(segm_prob, axis=-1) # (gh, gw)
            
            prob_tensor = torch.from_numpy(segm_prob_max).unsqueeze(0).unsqueeze(0)
            max_pool = F.max_pool2d(prob_tensor, kernel_size=3, stride=1, padding=1)
            is_local_max = (prob_tensor == max_pool).squeeze().numpy() & (segm_class > 0) 
            mask = (segm_prob_max > self.conf_threshold) & is_local_max
            ys, xs = np.nonzero(mask)
            
            if len(ys) == 0:
                batch_results.append([])
                continue
            
            probs = segm_prob_max[ys, xs]
            sort_idx = np.argsort(probs)[::-1]
            ys, xs = ys[sort_idx], xs[sort_idx]
            
            polylines = []
            visited_map = np.zeros_like(mask, dtype=bool)
            
            img_outputs['segm_class'] = segm_class

            for y, x in zip(ys, xs):
                if visited_map[y, x]:
                    continue
                
                category = segm_class[y, x]
                start_pt = np.array([x, y]) 
                polyline = self.find_graph(start_pt, category, img_outputs, visited_map)

                if len(polyline) > 0:
                    gh, gw = mask.shape
                    pts_grid = polyline * np.array([gw, gh])
                    pts_int = pts_grid.astype(np.int32)

                    valid_mask = (pts_int[:, 0] >= 0) & (pts_int[:, 0] < gw) & \
                                 (pts_int[:, 1] >= 0) & (pts_int[:, 1] < gh)
                    valid_pts = pts_int[valid_mask]
                    visited_map[valid_pts[:, 1], valid_pts[:, 0]] = True

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
        left_pt = point + outputs['left_point'][y, x]
        right_pt = point + outputs['right_point'][y, x]
        
        left_line = [center_pt, left_pt]
        right_line = [center_pt, right_pt]

        left_line = self.trace_graph(left_line, outputs, visited_map, category)
        right_line = self.trace_graph(right_line, outputs, visited_map, category)
        
        final_line = left_line[1:][::-1] + right_line
        final_line = np.array(final_line, dtype=np.float32)
        
        gh, gw = outputs['segm_prob'].shape[:2]
        final_line_norm = final_line / np.array([gw, gh], dtype=np.float32)
        
        return final_line_norm

    def _is_valid_point(self, pt, outputs, visited_map, category):
        """Check point validity(bounds, visited status, class matching)"""
        nx, ny = int(pt[0]), int(pt[1])
        gh, gw = outputs['segm_prob'].shape[:2]
        
        if not (0 <= nx < gw and 0 <= ny < gh):
            return False
        if visited_map[ny, nx]:
            return False
        if outputs['segm_class'][ny, nx] != category:
            return False
        return True

    def trace_graph(self, polyline, outputs, visited_map, category):
        if len(polyline) > 100:
            return polyline
            
        curr_tip = polyline[-1]
        prev_tip = polyline[-2]
        
        cx, cy = int(curr_tip[0]), int(curr_tip[1])
        gh, gw = outputs['segm_prob'].shape[:2]

        if not (0 <= cx < gw and 0 <= cy < gh):
            return polyline

        move_vec = curr_tip - prev_tip
        
        cand_left = np.array([cx, cy]) + outputs['left_point'][cy, cx]
        cand_right = np.array([cx, cy]) + outputs['right_point'][cy, cx]
        
        vec_to_left = cand_left - curr_tip
        vec_to_right = cand_right - curr_tip
        
        norm_move = np.linalg.norm(move_vec) + 1e-6
        norm_left = np.linalg.norm(vec_to_left) + 1e-6
        norm_right = np.linalg.norm(vec_to_right) + 1e-6
        
        cos_left = np.dot(move_vec, vec_to_left) / (norm_move * norm_left)
        cos_right = np.dot(move_vec, vec_to_right) / (norm_move * norm_right)
        
        if cos_left > cos_right:
            next_pt = cand_left
            target_vec = vec_to_left
            end_prob = outputs['left_end_prob'][cy, cx]
        else:
            next_pt = cand_right
            target_vec = vec_to_right
            end_prob = outputs['right_end_prob'][cy, cx]

        if end_prob > 0.5:
            return polyline

        if self._is_valid_point(next_pt, outputs, visited_map, category):
            polyline.append(next_pt)
            return self.trace_graph(polyline, outputs, visited_map, category)
        
        else:
            next_pt_far = next_pt + target_vec
            if self._is_valid_point(next_pt_far, outputs, visited_map, category):
                polyline.append(next_pt_far)
                return self.trace_graph(polyline, outputs, visited_map, category)
        
        return polyline

    def save_points_to_json(self, data: List[Dict[str, Any]], save_path: str) -> None:
        records = data[0] # TODO: 현재 리스트 하나로 고정 여러 개 저장할 수 있도록 수정 필요

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

def visualize_result(image: np.ndarray, polylines):
    vis_img = image.copy()
    W, H, _ = vis_img.shape
    
    colors = {0: (0, 0, 0), 1: (77, 77, 255), 2: (77, 178, 255), 3: (77, 255, 77), 4: (255, 153, 77),
                5: (255, 77, 77), 6: (178, 77, 255), 7: (77, 255, 178), 8: (255, 178, 77),
                9: (77, 102, 255), 10: (255, 77, 128), 11: (128, 255, 77)}

    # print(f"Detected {len(polylines)} polylines.")

    for poly_info in polylines:
        cat = poly_info['label']
        pts = poly_info['points'] # (N, 2) Normalized Coords (0~1)
        
        pts_pixel = (pts * np.array([W, H])).astype(np.int32)
        if len(pts_pixel) < 30:
            continue
        
        color = colors.get(cat, (255, 255, 255))
        cv2.polylines(vis_img, [pts_pixel], isClosed=False, color=color, thickness=2)
        
        for p in pts_pixel:
            cv2.circle(vis_img, tuple(p), 2, color, -1)
        
        if len(pts_pixel) > 0:
            cv2.circle(vis_img, tuple(pts_pixel[0]), 5, (0, 255, 0), -1)
            cv2.circle(vis_img, tuple(pts_pixel[-1]), 5, (0, 0, 255), -1)

    return vis_img


def main():
    img_path = '/home/gorilla/kyh_workspace/project/dataset/satellite_lane/validation/image'
    json_path = '/home/gorilla/kyh_workspace/project/results/tblog_260117_2012/checkpoints/last_instance'
    img_list = [os.path.join(img_path, i) for i in os.listdir(img_path)]
    json_list = [os.path.join(json_path, i) for i in os.listdir(json_path)]
    save_path = json_path.replace('_instance', '_intstance_img')
    os.makedirs(save_path, exist_ok=True)
    img_list.sort()
    json_list.sort()
    for img_name, json_name in zip(img_list, json_list):
        with open(json_name, 'r') as f:
            polyline = json.load(f)
        img = cv2.imread(img_name)
        result = visualize_result(img, polyline)
        cv2.imwrite(os.path.join(save_path, os.path.basename(img_name)), result)
    return

    torch_path = '/home/gorilla/kyh_workspace/project/results/tblog_260117_2012/checkpoints/epoch=16_pt'
    save_path = '/home/gorilla/kyh_workspace/project/results/tblog_260117_2012/checkpoints/epoch=16_instance'
    os.makedirs(save_path, exist_ok=True)
    torch_list = os.listdir(torch_path)
    cfg = CfgNode.from_file("satellite_detr")
    instance_generator = GeneratePolylineInstances.build_from_cfg(cfg)

    for torch_name in torch_list:
        data = torch.load(os.path.join(torch_path, torch_name))
        polylines = instance_generator([data])
        save_points_to_json(polylines[0], os.path.join(save_path, torch_name).replace('.pt', '.json'))




if __name__ == '__main__':
    from tqdm import tqdm
    main()