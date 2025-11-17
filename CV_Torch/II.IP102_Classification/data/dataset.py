import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class IP102_Classifier(Dataset):
    """ The Dataset Class of IP102 dataset """
    def __init__(self):
        """
        Args:
            image_dir: path of images dir
            txt_file: path of images_txt
            transform: images transformation
            class_name: the name of class(optional)
        """
        pass
        
