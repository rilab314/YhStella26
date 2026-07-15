import numpy as np
import random


class Composer:
    def __init__(self, cfg, split):
        self.transforms = []
        if cfg.dataset[split].augmentation:
            if hasattr(cfg.dataset.augmentation, 'horizontal_flip'):
                self.transforms.append(HorizontalFlip(p=cfg.dataset.augmentation.horizontal_flip.p))
            if hasattr(cfg.dataset.augmentation, 'vertical_flip'):
                self.transforms.append(VerticalFlip(p=cfg.dataset.augmentation.vertical_flip.p))

    def __call__(self, image, labels):
        for transform in self.transforms:
            image, labels = transform(image, labels)
        return {'image': image, 'labels': labels}


class HorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, labels):
        if random.random() < self.p:
            # 이미지 좌우 반전 (H, W, C)
            image = image[:, ::-1, :].copy()
            # 라벨 배열 순서 좌우 반전 (H/4, W/4, 9)
            labels = labels[:, ::-1, :].copy()
            # center x는 좌표 반전, left/right x는 단위벡터 성분 부호 반전
            labels[:, :, 0] = 1.0 - labels[:, :, 0]
            labels[:, :, 2] = -labels[:, :, 2]
            labels[:, :, 4] = -labels[:, :, 4]
        return image, labels


class VerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, labels):
        if random.random() < self.p:
            # 이미지 상하 반전
            image = image[::-1, :, :].copy()
            # 라벨 배열 순서 상하 반전
            labels = labels[::-1, :, :].copy()
            # center y는 좌표 반전, left/right y는 단위벡터 성분 부호 반전
            labels[:, :, 1] = 1.0 - labels[:, :, 1]
            labels[:, :, 3] = -labels[:, :, 3]
            labels[:, :, 5] = -labels[:, :, 5]
        return image, labels
