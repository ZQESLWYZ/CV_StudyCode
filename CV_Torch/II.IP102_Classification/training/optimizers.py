import torch.optim as optim
from torch.optim import Adam, SGD, AdamW
import torch.nn as nn

def get_optimizer(model, config):
    """
    根据配置创建优化器
    """
    optimizer_config = config['train']
    optimizer_name = optimizer_config['optimizer_name']
    learning_rate = optimizer_config['learning_rate']
    weight_decay = optimizer_config.get('weight_decay', 0.0)
    
    if optimizer_name.lower() == 'adam':
        optimizer = Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=optimizer_config.get('betas', (0.9, 0.999))
        )
    elif optimizer_name.lower() == 'sgd':
        optimizer = SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=optimizer_config.get('momentum', 0.9),
            nesterov=optimizer_config.get('nesterov', True)
        )
    elif optimizer_name.lower() == 'adamw':
        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=optimizer_config.get('betas', (0.9, 0.999))
        )
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")
    
    return optimizer