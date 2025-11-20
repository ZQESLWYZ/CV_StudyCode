import os
import cv2
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import read_yaml
from utils.helpers import show_tensor_image
from data.transforms import get_transform
from data.dataset import IP102_Classifier
from data.dataloader import get_dataloader


if __name__ == "__main__":
    config = read_yaml(r'CV_Torch/II.IP102_Classification/configs/train_config.yaml')
    dataset = IP102_Classifier(config, 'val')
    
    print(f"数据集大小: {len(dataset)}")
    print(f"第一张图像路径: {dataset.imgs[0]}")
    
    # 测试1: 展示单张图像
    img, label = dataset[10]
    print(f"\n图像形状: {img.shape}, 标签: {label}")
    show_tensor_image(img, title=f"label: {label}", save_path="test_single.jpg")
    
    # 测试dataloader
    train_loader, val_loader = get_dataloader(config)
    
    print(len(train_loader),len(val_loader))