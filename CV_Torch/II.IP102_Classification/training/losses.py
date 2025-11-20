import torch.nn as nn

def get_criterion(config):
    """
    根据配置创建损失函数
    """
    loss_config = config['train']
    loss_name = loss_config['loss_name']
    
    if loss_name.lower() == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(
            weight=loss_config.get('weight', None),
            label_smoothing=loss_config.get('label_smoothing', 0.0)
        )
    elif loss_name.lower() == 'focal':
        # 需要安装 pip install focal-loss-torch
        from focal_loss.focal_loss import FocalLoss
        criterion = FocalLoss(
            alpha=loss_config.get('alpha', 1.0),
            gamma=loss_config.get('gamma', 2.0)
        )
    elif loss_name.lower() == 'smooth_l1':
        criterion = nn.SmoothL1Loss(
            beta=loss_config.get('beta', 1.0)
        )
    else:
        raise ValueError(f"不支持的损失函数: {loss_name}")
    
    return criterion