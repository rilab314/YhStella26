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
            # 0, 2, 4번 채널(x 좌표) 변환: x_new = 1.0 - x_old
            labels[:, :, [0, 2, 4]] = 1.0 - labels[:, :, [0, 2, 4]]
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
            # 1, 3, 5번 채널(y 좌표) 변환: y_new = 1.0 - y_old
            labels[:, :, [1, 3, 5]] = 1.0 - labels[:, :, [1, 3, 5]]
        return image, labels
