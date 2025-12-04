import json
from tqdm import tqdm
import os
import os.path as op
from typing import List
import shutil
import cv2
import numpy as np
import pandas as pd

import settings
from configs.config import CfgNode


MAP_SCALE = 4


def generate_dataset():
    cfg = CfgNode.from_file('satellite_detr')
    dataset_path = cfg.dataset.path
    raw_data_path = op.join(os.path.dirname(dataset_path), 'satellite_images')
    with open(op.join(raw_data_path, 'dataset.json'), 'r', encoding='utf-8') as file:
        splits = json.load(file)
    for split in ['train', 'validation', 'test']:
        create_dataset(split, splits[split], raw_data_path, dataset_path, cfg)


def create_dataset(split, file_list: List[str], src_path, dst_path, cfg):
    split_path = os.path.join(dst_path, split)
    os.makedirs(os.path.join(split_path, 'image'), exist_ok=True)
    os.makedirs(os.path.join(split_path, 'label'), exist_ok=True)
    for file in tqdm(file_list):
        img_file = op.join(src_path, 'image', file + '.png')
        if not op.exists(img_file):
            print('image file does not exist:', img_file)
            exit()
        shutil.copy(img_file, os.path.join(split_path, 'image'))
        label_file = op.join(src_path, 'label', file + '.json')
        with open(label_file, 'r', encoding='utf-8') as f:
            label = json.load(f)
        
        label_map = draw_label_map(img_file, label, cfg)
        label_map_file = os.path.join(split_path, 'label', file + '.npy')
        np.save(label_map_file, label_map)


def draw_label_map(img_file, labels, cfg):
    image = cv2.imread(img_file)
    category_list = [label['name'] for label in cfg.dataset.labels]
    line_strings = [label for label in labels if label['class'] == 'RoadObject' and 'category' in label and label['category'] in category_list]
    label_map_accum = None
    for line in line_strings:
        category = category_list.index(line['category'])
        line_map_src = draw_line_string(image, line, 1)
        line_map_rdc = draw_line_string(image, line, MAP_SCALE)
        points_src = sample_points(line_map_src)
        label_map = make_label_map(image, points_src, line_map_rdc, category)
        if label_map is None:
            continue
        label_map_accum = accumulate_label_map(label_map_accum, label_map)
    return label_map_accum


def draw_line_string(image, line, scale: int):
    h, w, _ = image.shape
    mask = np.zeros((h//scale, w//scale), dtype=np.uint8)
    points = np.array(line['image_points'])
    scaled_points = (points / scale).astype(np.int32)
    cv2.polylines(mask, [scaled_points], isClosed=False, color=(255), thickness=1)
    return mask


def sample_points(line_map):
    nonzero_y, nonzero_x = np.nonzero(line_map)
    original_points = np.column_stack((nonzero_x, nonzero_y)).astype(np.int32)
    if original_points.shape[0] == 0:
        return np.array([])

    scaled_points = (original_points / MAP_SCALE).astype(int)
    df = pd.DataFrame({
        'ox': original_points[:, 0], # 원본 x
        'oy': original_points[:, 1], # 원본 y
        'sx': scaled_points[:, 0],   # 축소된 x
        'sy': scaled_points[:, 1]    # 축소된 y
    })
    sampled_df = df.groupby(['sx', 'sy']).mean()
    result_points = sampled_df[['ox', 'oy']].values
    return result_points


def make_label_map(image, points, map_mask, category):
    if points.shape[0] < 4:
        return None
    map_coords = (points / MAP_SCALE).astype(np.int32)
    mask_value = map_mask[map_coords[:, 1], map_coords[:, 0]]
    valid_indices = mask_value > 0
    points = points[valid_indices]
    map_coords = map_coords[valid_indices]

    if points.shape[0] < 4:
        return None

    h, w, _ = image.shape
    grid_h, grid_w = h // MAP_SCALE, w // MAP_SCALE
    grid_origins = map_coords * MAP_SCALE
    curr_origins = grid_origins[1:-1]

    norm_curr = (points[1:-1] - curr_origins) / MAP_SCALE
    norm_prev = (points[:-2] - curr_origins) / MAP_SCALE
    norm_next = (points[2:] - curr_origins) / MAP_SCALE

    point_map = np.zeros((grid_h, grid_w, 2), dtype=np.float32)
    prev_point_map = np.zeros((grid_h, grid_w, 2), dtype=np.float32)
    next_point_map = np.zeros((grid_h, grid_w, 2), dtype=np.float32)
    
    rows = map_coords[1:-1, 1]
    cols = map_coords[1:-1, 0]
    point_map[rows, cols] = norm_curr
    prev_point_map[rows, cols] = norm_prev
    next_point_map[rows, cols] = norm_next
    class_map = np.zeros((grid_h, grid_w, 3), dtype=np.float32)
    class_map[rows[0], cols[0], 0] = 1
    class_map[rows[-1], cols[-1], 1] = 1
    class_map[rows, cols, 2] = category
    label_map = np.concatenate([point_map, prev_point_map, next_point_map, class_map], axis=2)
    
    return label_map


def visualize_label_map(image, label_map, points):
    if label_map is None:
        return
    vis_img = image.copy()
    rows, cols = np.nonzero(label_map[:, :, 8] > 0)
    if len(rows) == 0:
        return vis_img
    valid_data = label_map[rows, cols, :]  # (N, 9)
    origins = np.stack([cols, rows], axis=1) * MAP_SCALE

    curr_pts = (valid_data[:, 0:2] * MAP_SCALE + origins).astype(np.int32)
    prev_pts = (valid_data[:, 2:4] * MAP_SCALE + origins).astype(np.int32)
    next_pts = (valid_data[:, 4:6] * MAP_SCALE + origins).astype(np.int32)

    line_color = (0, 255, 255) # Yellow
    pt_color = (0, 0, 255)     # Red    
    for i in range(len(rows)):
        p_curr = tuple(curr_pts[i])
        p_prev = tuple(prev_pts[i])
        p_next = tuple(next_pts[i])
        cv2.line(vis_img, p_prev, p_curr, line_color, 2)
        cv2.line(vis_img, p_curr, p_next, line_color, 2)
        cv2.circle(vis_img, p_curr, 2, pt_color, -1)

    cv2.imshow('labelmap', vis_img)
    cv2.waitKey(0)
    return vis_img


def accumulate_label_map(label_map_accum, label_map):
    if label_map_accum is None:
        label_map_accum = label_map.copy()
    mask = label_map[:, :, 0] != 0
    label_map_accum[mask] = label_map[mask]
    return label_map_accum


if __name__ == "__main__":
    generate_dataset()
