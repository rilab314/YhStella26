import os
import sys
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from timm.data import resolve_model_data_config
import json

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from dataset.composer_factory import Composer


class SatelliteImagesDataset(Dataset):
    @staticmethod
    def build_from_cfg(cfg, split):
        augment = Composer(cfg, split)
        dataset = SatelliteImagesDataset(cfg.dataset.path, cfg.dataset.num_classes, cfg.runtime.device, split=split, augment=augment)
        return dataset

    def __init__(self, dataset_path, num_classes, device, split: str, augment):
        self.root_path = dataset_path
        self.split = split
        self.device = torch.device(device)
        self.image_dir = str(os.path.join(dataset_path, self.split, 'image'))
        self.label_dir = str(os.path.join(dataset_path, self.split, 'label'))
        self.label_inst_dir = str(os.path.join(dataset_path, self.split, 'json'))
        label_insts = sorted(os.listdir(self.label_inst_dir))
        self.label_insts = [file for file in label_insts if file.endswith(('.json'))]
        image_files = sorted(os.listdir(self.image_dir))
        self.image_files = [file for file in image_files if file.endswith(('.png'))]
        self.augment = augment
        self.num_classes = num_classes
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_filename = self.image_files[idx]
        image = self.load_image(idx)
        labels = self.load_numpy_labels(idx)        
        if self.augment:
            transformed = self.augment(image, labels)
            image, labels = transformed['image'], transformed['labels']
        
        if isinstance(image, np.ndarray):
            image = self.transform(image)
        labels_tensor = torch.as_tensor(labels, dtype=torch.float32, device=self.device)

        target_dict = {
            "center_point": labels_tensor[:, :, :2],
            "left_point": labels_tensor[:, :, 2:4],
            "right_point": labels_tensor[:, :, 4:6],
            "left_end": labels_tensor[:, :, 6:7],
            "right_end": labels_tensor[:, :, 7:8],
            "segm_label": labels_tensor[:, :, 8:]
        }
        inst_target_dict = self.load_inst_label(idx)
        height, width = image.shape[1], image.shape[2]
        return {
            'image': image,
            'targets': target_dict,
            'inst_targets': inst_target_dict,
            'height': height,
            'width': width,
            'filename': os.path.join(self.image_dir, image_filename)
        }

    def load_image(self, idx):
        image_filename = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_filename)
        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def load_inst_label(self, idx):
        label_inst_filename = self.label_insts[idx]
        json_path = os.path.join(self.label_inst_dir, label_inst_filename)
        with open(json_path, 'r') as f:
            inst_label = json.load(f)
        
        return inst_label

    def load_numpy_labels(self, idx):
        """
        9차원 넘파이 라벨 파일을 로드합니다.
        라벨 차원 설명
        0, 1: 현재 블록의 좌표 정보 (x, y)
        2, 3: 이전 블록의 좌표 정보 (prev_x, prev_y)
        4, 5: 다음 블록의 좌표 정보 (next_x, next_y)
        6: 현재 블록이 라인의 시작 블록인지 여부 (is_start)
        7: 다음 블록이 라인의 끝 블록인지 여부 (is_end)
        8: 카테고리 정보 (category_id)

        Returns:
            labels (np.ndarray): 라벨 정보 (N x 9 배열)
        """
        image_filename = self.image_files[idx]
        label_filename = os.path.splitext(image_filename)[0] + '.npy'
        label_path = os.path.join(self.label_dir, label_filename)
        
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")
        
        return np.load(label_path)

    def apply_augmentation(self, image, labels):
        transformed = self.augment(
            image=image,
            labels=labels,
        )
        image = transformed['image']
        labels = transformed['labels']
        return image, labels



from configs.config import CfgNode
from util.misc import build_instance

class DatasetVisualizer:
    def __init__(self):
        self.cfg = CfgNode.from_file('stella_cfg')
        self.dataset = build_instance(
            self.cfg.dataset.module_name, 
            self.cfg.dataset.class_name, 
            self.cfg, 
            split='train'
        )
    
    def show(self, idx):
        # 1. 데이터 로드
        data = self.dataset[idx]
        image_tensor = data['image']       # (3, Img_H, Img_W)
        targets = data['targets']
        segm_label = targets['segm_label'] # (Label_H, Label_W, 1), Integer Class ID
        
        # 2. 이미지 전처리 (Tensor -> Numpy uint8 BGR)
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).clip(0, 255).astype(np.uint8)
        original_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # 원본 이미지 크기
        img_h, img_w = original_bgr.shape[:2]
        
        # 3. 라벨 전처리
        if isinstance(segm_label, torch.Tensor):
            segm_label = segm_label.cpu().numpy()
            
        # (H, W, 1) -> (H, W) 로 차원 축소 및 정수형 변환
        segm_label = segm_label.squeeze(-1).astype(int)
        
        # 라벨 크기가 원본과 다르면 리사이즈 (Nearest Neighbor 사용 필수)
        if segm_label.shape[:2] != (img_h, img_w):
            segm_label = cv2.resize(segm_label, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            
        # 4. 반투명 오버레이 생성
        vis_img = original_bgr.copy()
        alpha = 0.5
        
        # 현재 이미지에 존재하는 클래스만 찾아서 순회 (0번 배경이 제외 필요시 조건 추가)
        unique_classes = np.unique(segm_label)
        
        for cls_idx in unique_classes:
            # 배경(0)이나 정의되지 않은 클래스는 건너뛰기 (Dataset 설정에 따라 조정)
            if cls_idx == 0: 
                continue 
            
            # Config 범위 내에 있는지 확인
            if cls_idx < len(self.cfg.dataset.labels):
                # 해당 클래스의 마스크 추출
                mask = (segm_label == cls_idx)
                
                # 색상 가져오기 (BGR)
                color = self.cfg.dataset.labels[cls_idx]["color"]
                
                # 마스크 영역에 색상 입히기
                # vis_img[mask] = original * (1-alpha) + color * alpha
                roi = vis_img[mask]
                colored_roi = (roi.astype(float) * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
                vis_img[mask] = colored_roi
                
        # 5. 결과 연결 및 출력
        combined_img = np.hstack([original_bgr, vis_img])
        
        window_name = f'Dataset Visualization'
        cv2.imshow(window_name, combined_img)
        
        print(f"Showing index {idx}. Press any key to close window.")
        cv2.waitKey(0)


if __name__ == "__main__":
    visualizer = DatasetVisualizer()
    
    for i in range(10):
        print(visualizer.dataset[i]['inst_targets'])
