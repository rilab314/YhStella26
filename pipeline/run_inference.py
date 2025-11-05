import os
import cv2
import torch
torch.set_float32_matmul_precision('medium')

from configs.config import CfgNode
from model.predictor import Predictor

if __name__ == '__main__':
    cfg = CfgNode.from_file('satellite_detr')

    img_dir_path = '/workspace/SatelliteDet/dataset/satellite_lane/train/image'
    ckp_dir = '/workspace/SatelliteDet/tblog/checkpoints'
    ckp_list = ['epoch=18-val_loss=0.0000.ckpt', 'last.ckpt']

    for ckp_name in ckp_list:
        ckp_path = os.path.join(ckp_dir, ckp_name)
        predictor = Predictor(ckp_path, cfg)
        print('making predictor fin')
        img_list = [os.path.join(img_dir_path, i) for i in os.listdir(img_dir_path)]
        print(ckp_name, 'pred_start')

        for i, img_path in enumerate(img_list):
            os.makedirs(ckp_path[:-5], exist_ok=True)
            image_data = cv2.imread(img_path)
            result = predictor(image_data)
            cv2.imwrite(os.path.join(ckp_path[:-5], f'{i}th pred.png'), result)