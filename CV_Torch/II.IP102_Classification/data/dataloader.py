import torch
import os
import sys
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import IP102_Classifier


from utils.config import read_yaml

def get_dataloader(config: dict):
    """return the dataset's Dataloader

    Args:
        config (dict): the train_config yaml
    """

    train_dataset = IP102_Classifier(config, split='train')
    val_dataset = IP102_Classifier(config, split='val')
    
    batch_size = config['data']['batch_size']
    num_worker = config['data']['num_workers']
    pin_memory = config['data']['pin_memory']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_worker,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_worker,
        pin_memory=pin_memory
    )
    
    # 打印数据加载器信息
    print(f"训练集: {len(train_dataset)} 个样本, {len(train_loader)} 个batch")
    print(f"验证集: {len(val_dataset)} 个样本, {len(val_loader)} 个batch")
    print(f"Batch size: {batch_size}")
    
    return train_loader, val_loader
