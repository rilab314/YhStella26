import torch
from typing import List
from util.misc import nested_tensor_from_tensor_list
from torch.utils.data import DataLoader
from dataset import build_dataset
from dataset.satellite_lane.satellite_images import SatelliteImagesDataset


def custom_collate_fn(batch: List[dict]):
    """
    SatelliteImagesDataset을 위한 collate 함수
    
    batch는 Dataset에서 반환한 dict들의 리스트:
    [
        {
            'image': Tensor(3, H, W),
            'targets': { 
                'center_point': Tensor(H, W, 2)
                'left_point': Tensor(H, W, 2)
                'right_point': Tensor(H, W, 2)
                'left_end': Tensor(H, W, 1)
                'right_end': Tensor(H, W, 1)
                'segm_label': Tensor(H, W, 1)
            },
            'height': int,
            'width': int,
            'filename': str
        },
        ...
    ]
    
    이걸 Deformable DETR가 원하는 형태로 맞춘다:
        samples = NestedTensor(batch_images, batch_masks)
        targets = List[Dict], 각 이미지별 line_block, label, size 등
    """
    # NestedTensor 생성
    images = [item['image'] for item in batch]
    samples = nested_tensor_from_tensor_list(images)

    targets = []
    for i, item in enumerate(batch):
        t = {}
        t["center_point"] = item["targets"]["center_point"]
        t["left_point"] = item["targets"]["left_point"]
        t["right_point"] = item["targets"]["right_point"]
        t["left_end"] = item["targets"]["left_end"]
        t["right_end"] = item["targets"]["right_end"]
        t["segm_label"] = item["targets"]["segm_label"]
        height = item["height"] 
        width = item["width"]
        t["size"] = torch.tensor([height, width])
        t["image_id"] = torch.tensor([i])
        t["filename"] = item["filename"]
        targets.append(t)

    return samples, targets


def create_dataloader(cfg, dataset, split='train', persistent_workers=True):
    """
    위성 이미지 데이터셋을 위한 특화된 데이터로더 생성 함수
    """
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True if split == 'train' else False,
        num_workers=cfg.training.num_workers,
        collate_fn=custom_collate_fn,
        drop_last=True,
        persistent_workers=persistent_workers,
    )
    return dataloader
