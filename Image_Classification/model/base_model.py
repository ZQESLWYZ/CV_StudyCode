"""
# File       : model/base_model.py
# Author     : ZQESLWYZ
# version    : python 3.10
# Software   : VSCode
# Date       : 2025-12-05
"""
import torch.nn as nn

from .AlexNet import AlexNet


class BaseModel(nn.Module):
    def __init__(self, name, num_classes):
        super(BaseModel, self).__init__()
        if name == 'alexnet':
            self.base = AlexNet(num_classes)
        else:
            raise ValueError('Input model name is not supported!!!')

    def forward(self, x):
        return self.base(x)
