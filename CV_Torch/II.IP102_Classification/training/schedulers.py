from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, CosineAnnealingLR, 
    CosineAnnealingWarmRestarts, ReduceLROnPlateau,
    ExponentialLR, OneCycleLR
)

def get_scheduler(optimizer, config, train_loader=None):
    """
    根据配置创建学习率调度器
    """
    scheduler_config = config['train']
    
    if not scheduler_config:
        return None
    
    scheduler_name = scheduler_config['scheduler_name']
    scheduler = None
    
    if scheduler_name.lower() == 'cosineannealing':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config['train']['num_epochs'],  # 最大迭代次数
        )

    
    return scheduler