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

class IP102_Classifier(Dataset):
    """ The Dataset Class of IP102 dataset """
    
    def __init__(self, config):
        """
        Args:
            image_dir: path of images dir
            txt_file: path of images_txt
            transform: images transformation
        """
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.txt_dir = config['data']['dataset_path']
        self.img_root_dir = os.path.join(self.project_root,
                                         'II.IP102_Classification', 
                                         'data', 'raw')
        self.imgs, self.labels = self.read_txt()
        self.transform = get_transform(config)

        
    def read_txt(self):
        img_path_temp = []
        img_label_temp = []
        
        with open(self.txt_dir, 'r', encoding='utf-8') as f:
            while content:= f.readline():
                
                content = content.strip()
                img_name, img_label = content.split(' ')
                
                img_path_temp.append(os.path.join(self.project_root, 
                                                  'II.IP102_Classification',
                                                'data', 'raw', 'images', img_name))
                img_label_temp.append(int(img_label))
                
        return img_path_temp, img_label_temp
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = self.labels[index]
        
        try:
            img = Image.open(img_path).convert('RGB')
            
            if self.transform:
                img = self.transform(img)
            
            return img, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"加载图像失败: {img_path}, 错误: {str(e)}")
            raise
    

    
