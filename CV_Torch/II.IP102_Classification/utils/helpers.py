import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple


def show_tensor_image(tensor_img: torch.Tensor, 
                     title: str = "Image",
                     denormalize: bool = True,
                     mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                     std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
                     save_path: Optional[str] = None,
                     figsize: Tuple[int, int] = (8, 8)):
    """
    展示PyTorch tensor格式的图像
    
    Args:
        tensor_img: 形状为 (C, H, W) 的tensor图像
        title: 图像标题
        denormalize: 是否进行反标准化
        mean: 标准化时使用的均值（RGB顺序）
        std: 标准化时使用的标准差（RGB顺序）
        save_path: 保存路径，如果为None则不保存
        figsize: 图像显示大小
        
    Example:
        >>> img, label = dataset[0]
        >>> show_tensor_image(img, title=f"Label: {label}")
    """
    # 复制tensor避免修改原始数据
    img = tensor_img.clone().detach()
    
    # 确保tensor在CPU上
    if img.is_cuda:
        img = img.cpu()
    
    # 反标准化
    if denormalize:
        mean = torch.tensor(mean).view(3, 1, 1)
        std = torch.tensor(std).view(3, 1, 1)
        img = img * std + mean
    
    # 裁剪到[0, 1]范围
    img = torch.clamp(img, 0, 1)
    
    # 转换为numpy并调整通道顺序: (C, H, W) -> (H, W, C)
    img_np = img.permute(1, 2, 0).numpy()
    
    # 显示图像
    plt.figure(figsize=figsize)
    plt.imshow(img_np)
    plt.title(title, fontsize=14)
    plt.axis('off')
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        print(f"图像已保存到: {save_path}")
    
    plt.show()

